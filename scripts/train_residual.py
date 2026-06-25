import importlib
import os
import sys
from copy import deepcopy
from datetime import datetime

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaacgym import gymutil
import isaacgym

from configs import *  # noqa: F401,F403
from global_config import ROOT_DIR
from modules.model_loader import load_actor_critic_checkpoint
from modules.residual_expert_actor_critic import ResidualExpertActorCritic
from runner.residual_policy_runner import ResidualPolicyRunner
from tasks import register_all_tasks
from utils.helpers import class_to_dict, update_cfg_from_args
from utils.task_registry import task_registry


def get_residual_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "d1h_moe_disc", "help": "Task name."},
        {"name": "--base_task", "type": str, "default": "d1h_moe_base", "help": "Task name used to build the frozen base policy."},
        {"name": "--resume", "action": "store_true", "default": False, "help": "Resume training from a checkpoint"},
        {"name": "--experiment_name", "type": str, "help": "Override experiment name."},
        {"name": "--run_name", "type": str, "help": "Override run name."},
        {"name": "--load_run", "type": str, "help": "Run name to load when resume=True."},
        {"name": "--checkpoint", "type": int, "help": "Checkpoint id to load when resume=True."},
        {"name": "--headless", "action": "store_true", "default": False, "help": "Force display off at all times"},
        {"name": "--horovod", "action": "store_true", "default": False, "help": "Use horovod for multi-gpu training"},
        {"name": "--rl_device", "type": str, "default": "cuda:0", "help": "Device used by the RL algorithm."},
        {"name": "--num_envs", "type": int, "help": "Override number of environments."},
        {"name": "--seed", "type": int, "help": "Override random seed."},
        {"name": "--max_iterations", "type": int, "help": "Override max learning iterations."},
        {"name": "--base_ckpt", "type": str, "default": None, "help": "Checkpoint path for the frozen base policy."},
        # residual_alpha:
        #   residual expert 输出动作修正量的最终缩放系数。
        #   最终动作近似为：a_final = a_base + residual_alpha * a_residual。
        #   数值越大，expert 对动作影响越强；数值越小，base 越占主导。
        #   disc 第一版建议 0.35~0.50，避免 residual 直接覆盖 base。
        # residual_alpha_warmup_steps:
        #   residual_alpha 从较小值逐渐升到目标值所用的训练迭代数。
        #   作用是避免训练初期 residual 随机输出过大，破坏 frozen base 的稳定动作。
        # residual_alpha_warmup_min:
        #   warmup 初期 alpha 的比例。
        #   例如 residual_alpha=0.45, warmup_min=0.15，则初始 alpha≈0.0675。
        #   后续随训练迭代逐渐升到 0.45。
        # residual_delta_clip:
        #   对已经乘过 alpha 的 residual 修正量 delta 做逐维硬截断。
        #   即 delta = clamp(alpha * residual_mean, -clip, +clip)。
        #   作用是防止 expert 输出过大修正，直接覆盖 base 动作。
        # residual_std_scale:
        #   residual actor 探索噪声 std 的缩放系数。
        #   默认为 1.0。减小它可以降低探索动作的抖动和激进程度。
        # residual_std_min / residual_std_max:
        #   最终动作分布 std 的下限和上限。
        #   std 太小探索不足，std 太大容易动作抖动、弹跳或摔倒。
        #   disc residual 第一版建议 min=0.20, max=0.65。
        # reset_residual_std:
        #   resume residual 训练时强行重置 residual actor 的 std。
        #   用于 std 已经过大或过小时重新控制探索强度。
        #   fresh training 一般不用。
        {"name": "--residual_alpha", "type": float, "default": 0.60, "help": "Final scale factor for the residual expert mean."},
        {"name": "--residual_alpha_warmup_steps", "type": int, "default": 3000, "help": "Iterations used to ramp residual alpha to its final value."},
        {"name": "--residual_alpha_warmup_min", "type": float, "default": 0.25, "help": "Initial alpha fraction during warmup."},
        {"name": "--residual_delta_clip", "type": float, "default": 0.65, "help": "Per-action clamp for alpha-scaled residual mean. Set <=0 to disable."},
        {"name": "--residual_std_scale", "type": float, "default": None, "help": "Optional scale factor for residual exploration std. Defaults to 1.0 when omitted."},
        {"name": "--residual_std_min", "type": float, "default": 0.45, "help": "Lower clamp for the final executed action std."},
        {"name": "--residual_std_max", "type": float, "default": 1.10, "help": "Upper clamp for the final executed action std."},
        {"name": "--reset_residual_std", "type": float, "default": None, "help": "If set, overwrite residual std after loading a resume checkpoint."},
        {"name": "--stair_ff_scale", "type": float, "default": None, "help": "Fix stair feedforward at this scale for the whole run. Omit to use annealing."},
        {"name": "--stair_ff_anneal_iter_offset", "type": float, "default": None, "help": "Start the stair feedforward anneal schedule at this local iteration."},
    ]

    args = gymutil.parse_arguments(description="Train residual expert policy.", custom_parameters=custom_parameters)
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args


def build_actor_critic(module_name, class_name, env, policy_cfg):
    actor_critic_module = importlib.import_module(module_name)
    actor_critic_class = getattr(actor_critic_module, class_name)
    return actor_critic_class(
        env.cfg.env.n_proprio,
        env.cfg.env.n_scan,
        env.num_obs,
        env.cfg.env.n_priv_latent,
        env.cfg.env.history_len,
        env.num_actions,
        **deepcopy(policy_cfg),
    )


def build_log_dir(train_cfg):
    log_root = os.path.join(ROOT_DIR, "logs", train_cfg.runner.experiment_name)
    return os.path.join(log_root, datetime.now().strftime("%b%d_%H-%M-%S") + "_" + train_cfg.runner.run_name)


def train(args):
    if not args.resume and not args.base_ckpt:
        raise ValueError("Fresh residual training requires --base_ckpt.")

    env, _ = task_registry.make_env(name=args.task, args=args)
    if hasattr(env.cfg, "control"):
        control_cfg = env.cfg.control
        if args.stair_ff_scale is not None:
            if hasattr(control_cfg, "stair_ff_anneal_override_scale"):
                control_cfg.stair_ff_anneal_override_scale = float(args.stair_ff_scale)
        if args.stair_ff_anneal_iter_offset is not None:
            if hasattr(control_cfg, "stair_ff_anneal_iter_offset"):
                control_cfg.stair_ff_anneal_iter_offset = float(args.stair_ff_anneal_iter_offset)
    if hasattr(env.cfg, "rewards"):
        diagnostic_clip = float(args.residual_delta_clip) if args.residual_delta_clip > 0.0 else 0.55
        env.cfg.rewards.residual_delta_clip_for_diagnostics = diagnostic_clip

    _, train_cfg = task_registry.get_cfgs(args.task)
    _, base_train_cfg = task_registry.get_cfgs(args.base_task)
    _, train_cfg = update_cfg_from_args(None, train_cfg, args)
    train_cfg_dict = class_to_dict(train_cfg)
    base_train_cfg_dict = class_to_dict(base_train_cfg)

    policy_class_name = train_cfg_dict["runner"]["policy_class_name"]
    base_policy_class_name = base_train_cfg_dict["runner"]["policy_class_name"]
    policy_cfg = train_cfg_dict["policy"]
    base_policy_cfg = base_train_cfg_dict["policy"]

    base_actor_critic = build_actor_critic("modules", base_policy_class_name, env, base_policy_cfg)
    residual_actor_critic = build_actor_critic("modules", policy_class_name, env, policy_cfg)

    if args.base_ckpt:
        load_actor_critic_checkpoint(base_actor_critic, args.base_ckpt, args.rl_device)

    actor_critic = ResidualExpertActorCritic(
        base_actor_critic=base_actor_critic,
        residual_actor_critic=residual_actor_critic,
        alpha=args.residual_alpha,
        freeze_base=True,
        residual_std_scale=args.residual_std_scale,
        min_policy_std=args.residual_std_min,
        max_policy_std=args.residual_std_max,
        residual_delta_clip=args.residual_delta_clip,
        alpha_warmup_steps=args.residual_alpha_warmup_steps,
        alpha_warmup_min=args.residual_alpha_warmup_min,
    )

    log_dir = build_log_dir(train_cfg)
    print("[train_residual] task           =", args.task)
    print("[train_residual] base_task      =", args.base_task)
    print("[train_residual] base_ckpt      =", args.base_ckpt)
    print("[train_residual] residual_alpha =", args.residual_alpha)
    print("[train_residual] residual_alpha_warmup_steps =", args.residual_alpha_warmup_steps)
    print("[train_residual] residual_alpha_warmup_min =", args.residual_alpha_warmup_min)
    print("[train_residual] residual_delta_clip =", args.residual_delta_clip)
    print("[train_residual] residual_std_scale =", actor_critic.residual_std_scale)
    print("[train_residual] residual_std_range =", (actor_critic.min_policy_std, actor_critic.max_policy_std))
    print("[train_residual] reset_residual_std =", args.reset_residual_std)
    control_cfg = getattr(env.cfg, "control", None)
    print("[train_residual] stair_ff_scale =", getattr(control_cfg, "stair_ff_anneal_override_scale", None))
    print("[train_residual] stair_ff_anneal_iter_offset =", getattr(control_cfg, "stair_ff_anneal_iter_offset", 0.0))
    print("[train_residual] log_dir        =", log_dir)

    runner = ResidualPolicyRunner(
        env,
        train_cfg_dict,
        actor_critic,
        log_dir=log_dir,
        device=args.rl_device,
        reset_residual_std=args.reset_residual_std,
    )
    runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    register_all_tasks()
    args = get_residual_args()
    train(args)
