import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaacgym import gymutil
import isaacgym  # noqa: F401

from tasks import register_all_tasks
from utils.helpers import class_to_dict
from utils.task_registry import task_registry


def get_args():
    custom_parameters = [
        {"name": "--task", "type": str, "default": "d1h_moe_gate"},
        {"name": "--base_task", "type": str, "default": "d1h_base"},
        {"name": "--stair_task", "type": str, "default": "d1h_disc_residual"},
        {"name": "--slip_task", "type": str, "default": "d1h_slip_residual"},
        {"name": "--recovery_task", "type": str, "default": "d1h_recovery_residual"},
        {"name": "--base_ckpt", "type": str, "required": True},
        {"name": "--stair_ckpt", "type": str, "required": True},
        {"name": "--slip_ckpt", "type": str, "required": True},
        {"name": "--recovery_ckpt", "type": str, "required": True},
        {"name": "--estimator_ckpt", "type": str, "required": True},
        {"name": "--num_envs", "type": int, "default": 4096},
        {"name": "--max_iterations", "type": int, "default": 4000},
        {"name": "--run_name", "type": str, "default": "gate_top2_v1"},
        {"name": "--residual_alpha", "type": float, "default": 0.60},
        {"name": "--residual_delta_clip", "type": float, "default": 0.0},
        {"name": "--gate_top_k", "type": int, "default": 2},
        {"name": "--gate_temperature", "type": float, "default": 1.0},
        {"name": "--gate_init_weight", "type": float, "default": 0.05},
        {"name": "--gate_aux_coef", "type": float, "default": 0.10},
        {"name": "--record_video", "action": "store_true", "default": False},
        {"name": "--headless", "action": "store_true", "default": False},
        {"name": "--rl_device", "type": str, "default": "cuda:0"},
    ]
    # update_cfg_from_args expects the standard runner fields.
    runner_defaults = {
        "seed": None,
        "resume": False,
        "experiment_name": None,
        "load_run": None,
        "checkpoint": None,
    }
    args = gymutil.parse_arguments(
        description="Train the D1H independent-sigmoid residual MoE gate.",
        custom_parameters=custom_parameters,
    )
    for field, default in runner_defaults.items():
        if not hasattr(args, field):
            setattr(args, field, default)
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args


def train(args):
    env_cfg, train_cfg = task_registry.get_cfgs(args.task)
    _, base_train_cfg = task_registry.get_cfgs(args.base_task)
    _, stair_train_cfg = task_registry.get_cfgs(args.stair_task)
    _, slip_train_cfg = task_registry.get_cfgs(args.slip_task)
    _, recovery_train_cfg = task_registry.get_cfgs(args.recovery_task)

    expert_cfgs = {
        "base": class_to_dict(base_train_cfg),
        "stair": class_to_dict(stair_train_cfg),
        "slip": class_to_dict(slip_train_cfg),
        "recovery": class_to_dict(recovery_train_cfg),
    }
    for tag, config in expert_cfgs.items():
        setattr(train_cfg.policy, f"{tag}_policy_cfg", config["policy"])
        setattr(train_cfg.policy, f"{tag}_policy_class_name", config["runner"]["policy_class_name"])
        setattr(train_cfg.policy, f"{tag}_ckpt", getattr(args, f"{tag}_ckpt"))

    train_cfg.policy.estimator_ckpt = args.estimator_ckpt
    train_cfg.policy.residual_alpha = args.residual_alpha
    train_cfg.policy.residual_delta_clip = args.residual_delta_clip
    train_cfg.policy.gate_top_k = args.gate_top_k
    train_cfg.policy.gate_temperature = args.gate_temperature
    train_cfg.policy.gate_init_weight = args.gate_init_weight
    train_cfg.algorithm.gate_aux_coef = args.gate_aux_coef
    train_cfg.runner.max_iterations = args.max_iterations
    train_cfg.runner.run_name = args.run_name
    train_cfg.runner.record_video = args.record_video
    env_cfg.env.num_envs = args.num_envs

    env, env_cfg = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    runner, train_cfg = task_registry.make_alg_runner(
        env=env, name=args.task, args=args, train_cfg=train_cfg
    )
    task_registry.save_cfgs(name=args.task, env_cfg=env_cfg, train_cfg=train_cfg)
    runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations,
        init_at_random_ep_len=True,
    )


if __name__ == "__main__":
    register_all_tasks()
    train(get_args())
