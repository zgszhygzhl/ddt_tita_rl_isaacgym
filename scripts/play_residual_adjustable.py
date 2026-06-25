```python
"""Adjustable inference / video recording for base + residual expert policy.

Example:

python scripts/play_residual_adjustable.py \
  --task=d1h_disc_residual \
  --base_task=d1h_base \
  --base_ckpt logs/d1h_base/Junxx_xx-xx-xx_d1h_base/checkpoints/model_8000.pt \
  --load_run Junxx_xx-xx-xx_d1h_disc_residual \
  --checkpoint 8000 \
  --headless \
  --play_terrain stairs_up \
  --play_stair_height 0.15 \
  --play_vx 0.35 \
  --play_duration 20 \
  --play_output disc_residual_play.mp4

也可以直接指定 residual checkpoint：

python scripts/play_residual_adjustable.py \
  --task=d1h_disc_residual \
  --base_task=d1h_base \
  --base_ckpt logs/d1h_base/xxx/checkpoints/model_8000.pt \
  --residual_ckpt logs/d1h_disc_residual/xxx/checkpoints/model_8000.pt \
  --headless \
  --play_terrain stairs_up \
  --play_stair_height 0.15 \
  --play_vx 0.35
"""

import importlib
import os
import sys
from copy import deepcopy

# scripts/ 下运行时，把项目根目录加入 import path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

import numpy as np
import torch
from isaacgym import gymapi, gymutil
import isaacgym  # noqa: F401

from global_config import ROOT_DIR
from modules.model_loader import load_actor_critic_checkpoint
from modules.residual_expert_actor_critic import ResidualExpertActorCritic
from tasks import register_all_tasks
from utils.helpers import class_to_dict, get_load_path
from utils.task_registry import task_registry
from utils.video_recorder import FfmpegVideoWriter

# 复用现有单策略推理脚本里的可调地形、视频、metrics 工具函数。
from play_climb_adjustable import (  # noqa: E402
    PLAY_TILE_ROWS,
    PLAY_TILE_COLS,
    PLAY_TILE_WIDTH,
    PLAY_TILE_HEIGHT,
    PLAY_VIDEO_FPS,
    set_eval_terrain,
    select_env_ids,
    create_cameras,
    capture_mosaic_frame,
    collect_metrics_step,
    save_metrics_plot,
    _resolve_metrics_output,
)


def get_residual_play_args():
    custom_parameters = [
        # residual task / checkpoint
        {"name": "--task", "type": str, "default": "d1h_disc_residual",
         "help": "Residual task name, e.g. d1h_disc_residual."},
        {"name": "--base_task", "type": str, "default": "d1h_base",
         "help": "Task name used to build the frozen base policy."},
        {"name": "--base_ckpt", "type": str, "default": None,
         "help": "Checkpoint path for the frozen base policy."},
        {"name": "--residual_ckpt", "type": str, "default": None,
         "help": "Direct checkpoint path for residual wrapper/expert. If omitted, use --load_run/--checkpoint."},

        # normal checkpoint resolving for residual run
        {"name": "--load_run", "type": str, "default": None,
         "help": "Residual run folder under logs/<experiment_name>/."},
        {"name": "--checkpoint", "type": int, "default": -1,
         "help": "Residual checkpoint id. -1 means latest model_*.pt."},

        # residual combination parameters
        {"name": "--residual_alpha", "type": float, "default": 0.45,
         "help": "Scale of residual action at inference."},
        {"name": "--residual_delta_clip", "type": float, "default": 0.45,
         "help": "Per-action clamp for alpha-scaled residual delta. <=0 disables residual delta clamp."},
        {"name": "--residual_std_min", "type": float, "default": 0.20,
         "help": "Unused for deterministic inference, kept for wrapper construction."},
        {"name": "--residual_std_max", "type": float, "default": 0.65,
         "help": "Unused for deterministic inference, kept for wrapper construction."},

        # play controls
        {"name": "--play_vx", "type": float, "default": 0.35,
         "help": "Fixed command vx during play."},
        {"name": "--play_vy", "type": float, "default": 0.0,
         "help": "Fixed command vy during play."},
        {"name": "--play_yaw", "type": float, "default": 0.0,
         "help": "Fixed yaw-rate command during play."},
        {"name": "--play_duration", "type": float, "default": 20.0,
         "help": "Play duration in seconds."},
        {"name": "--play_num_envs", "type": int, "default": 16,
         "help": "Number of simulation envs for play."},
        {"name": "--play_video_num_envs", "type": int, "default": 4,
         "help": "Number of envs shown in the mosaic video."},
        {"name": "--play_terrain", "type": str, "default": "stairs_up",
         "help": "One of: stairs_up, stairs_down, slope, rough_slope, mixed."},
        {"name": "--play_stair_height", "type": float, "default": 0.15,
         "help": "Fixed stair height in meters."},
        {"name": "--play_step_width", "type": float, "default": 0.55,
         "help": "Fixed stair step width in meters."},
        {"name": "--play_slope", "type": float, "default": 0.0,
         "help": "Fixed slope for slope/rough_slope play terrain."},
        {"name": "--play_seed", "type": int, "default": 0,
         "help": "Seed for selecting video env ids."},
        {"name": "--play_output", "type": str, "default": "play_residual_adjustable.mp4",
         "help": "Output mp4 name or path."},
        {"name": "--play_metrics_output", "type": str, "default": "",
         "help": "Output metrics PNG. Default: <play_output>_metrics.png."},
        {"name": "--play_metrics_env_id", "type": int, "default": -1,
         "help": "Environment id used for metrics plotting. Default: first selected video env."},
        {"name": "--play_metrics_max_points", "type": int, "default": 2500,
         "help": "Maximum points drawn in metrics PNG."},
    ]

    args = gymutil.parse_arguments(
        description="Play base + residual expert policy.",
        custom_parameters=custom_parameters,
    )

    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"

    return args


def _resolve_path(path):
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(ROOT_DIR, path)


def build_actor_critic(policy_class_name, env, policy_cfg):
    actor_critic_module = importlib.import_module("modules")
    actor_critic_class = getattr(actor_critic_module, policy_class_name)

    return actor_critic_class(
        env.cfg.env.n_proprio,
        env.cfg.env.n_scan,
        env.num_obs,
        env.cfg.env.n_priv_latent,
        env.cfg.env.history_len,
        env.num_actions,
        **deepcopy(policy_cfg),
    )


def resolve_residual_checkpoint(args, train_cfg):
    if args.residual_ckpt is not None:
        ckpt = _resolve_path(args.residual_ckpt)
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Residual checkpoint does not exist: {ckpt}")
        return ckpt

    log_root = os.path.join(ROOT_DIR, "logs", train_cfg.runner.experiment_name)
    load_run = args.load_run if args.load_run is not None else -1
    checkpoint = args.checkpoint if args.checkpoint is not None else -1
    return get_load_path(log_root, load_run=load_run, checkpoint=checkpoint)


def _extract_state_dict(loaded_obj):
    if isinstance(loaded_obj, dict):
        if "model_state_dict" in loaded_obj:
            return loaded_obj["model_state_dict"]
        if "actor_critic_state_dict" in loaded_obj:
            return loaded_obj["actor_critic_state_dict"]
    return loaded_obj


def _strip_prefix(state_dict, prefix):
    prefix_len = len(prefix)
    return {
        key[prefix_len:]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


def _print_incompatible_keys(prefix, result):
    # PyTorch IncompatibleKeys has missing_keys/unexpected_keys.
    missing = getattr(result, "missing_keys", [])
    unexpected = getattr(result, "unexpected_keys", [])

    if len(missing) > 0:
        print(f"[{prefix}] missing keys ({len(missing)}):")
        for key in missing[:20]:
            print(f"  missing: {key}")
        if len(missing) > 20:
            print(f"  ... {len(missing) - 20} more")

    if len(unexpected) > 0:
        print(f"[{prefix}] unexpected keys ({len(unexpected)}):")
        for key in unexpected[:20]:
            print(f"  unexpected: {key}")
        if len(unexpected) > 20:
            print(f"  ... {len(unexpected) - 20} more")


def load_residual_checkpoint(wrapper, ckpt_path, device):
    """
    residual 训练保存的是 ResidualExpertActorCritic 的 state_dict，
    key 通常长这样：
        base_actor_critic.xxx
        residual_actor_critic.xxx

    推理时 base 由 --base_ckpt 显式加载。
    这里优先只加载 residual_actor_critic.xxx，避免 residual checkpoint 覆盖 base。
    """
    loaded = torch.load(ckpt_path, map_location=device)
    state_dict = _extract_state_dict(loaded)

    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported residual checkpoint format: {type(state_dict)}")

    residual_state = _strip_prefix(state_dict, "residual_actor_critic.")

    if len(residual_state) > 0:
        result = wrapper.residual_actor_critic.load_state_dict(residual_state, strict=False)
        _print_incompatible_keys("residual_actor_critic", result)
        print(f"[play_residual] loaded residual_actor_critic from wrapper checkpoint: {ckpt_path}")
        return

    # 兼容一种可能：checkpoint 本身就是 residual actor 的 state_dict。
    result = wrapper.residual_actor_critic.load_state_dict(state_dict, strict=False)
    _print_incompatible_keys("residual_actor_critic", result)
    print(f"[play_residual] loaded residual actor state_dict directly: {ckpt_path}")


def make_residual_policy(env, args, train_cfg, base_train_cfg):
    train_cfg_dict = class_to_dict(train_cfg)
    base_train_cfg_dict = class_to_dict(base_train_cfg)

    residual_policy_class_name = train_cfg_dict["runner"]["policy_class_name"]
    base_policy_class_name = base_train_cfg_dict["runner"]["policy_class_name"]

    residual_policy_cfg = train_cfg_dict["policy"]
    base_policy_cfg = base_train_cfg_dict["policy"]

    base_actor_critic = build_actor_critic(base_policy_class_name, env, base_policy_cfg)
    residual_actor_critic = build_actor_critic(residual_policy_class_name, env, residual_policy_cfg)

    base_ckpt = _resolve_path(args.base_ckpt)
    if base_ckpt is None:
        raise ValueError("--base_ckpt is required for base + residual inference.")
    if not os.path.exists(base_ckpt):
        raise FileNotFoundError(f"Base checkpoint does not exist: {base_ckpt}")

    load_actor_critic_checkpoint(base_actor_critic, base_ckpt, env.device)
    print(f"[play_residual] loaded base checkpoint: {base_ckpt}")

    wrapper = ResidualExpertActorCritic(
        base_actor_critic=base_actor_critic,
        residual_actor_critic=residual_actor_critic,
        alpha=float(args.residual_alpha),
        freeze_base=True,
        residual_std_scale=1.0,
        min_policy_std=float(args.residual_std_min),
        max_policy_std=float(args.residual_std_max),
        residual_delta_clip=float(args.residual_delta_clip),
        alpha_warmup_steps=0,
        alpha_warmup_min=1.0,
        zero_init_residual=True,
    )

    residual_ckpt = resolve_residual_checkpoint(args, train_cfg)
    load_residual_checkpoint(wrapper, residual_ckpt, env.device)

    wrapper.eval().to(env.device)

    print("[play_residual] residual_alpha      =", args.residual_alpha)
    print("[play_residual] residual_delta_clip =", args.residual_delta_clip)
    print("[play_residual] residual checkpoint =", residual_ckpt)

    return wrapper


def print_residual_summary(policy, delta_norms, saturation_ratios):
    print("[play_residual] residual summary:")

    if len(delta_norms) > 0:
        arr = np.asarray(delta_norms, dtype=np.float32)
        print(f"  delta_norm mean={arr.mean():.4f}, max={arr.max():.4f}, min={arr.min():.4f}")
    else:
        print("  delta_norm: no data")

    if len(saturation_ratios) > 0:
        arr = np.asarray(saturation_ratios, dtype=np.float32)
        print(f"  action_saturation_ratio mean={arr.mean():.4f}, max={arr.max():.4f}")
    else:
        print("  action_saturation_ratio: no data")

    if hasattr(policy, "last_current_alpha"):
        print("  current_alpha =", float(policy.last_current_alpha.detach().cpu().item()))


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    _, base_train_cfg = task_registry.get_cfgs(name=args.base_task)

    # 复用 play_climb_adjustable.py 的可调地形逻辑
    set_eval_terrain(env_cfg, args)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.reset()
    obs = env.get_observations()

    policy = make_residual_policy(env, args, train_cfg, base_train_cfg)

    video_env_ids = select_env_ids(env.num_envs, args.play_video_num_envs, seed=args.play_seed)
    cam_handles = create_cameras(env, video_env_ids)
    print(f"[play_residual] selected video envs: {video_env_ids}")

    if int(args.play_metrics_env_id) >= 0:
        metric_env_id = min(int(args.play_metrics_env_id), env.num_envs - 1)
    else:
        metric_env_id = int(video_env_ids[0]) if len(video_env_ids) > 0 else 0
    print(f"[play_residual] metrics env: {metric_env_id}")

    metrics = {"t": [], "torques": [], "cmd": [], "cmd_given": [], "vel": []}
    delta_norms = []
    saturation_ratios = []

    video_width = PLAY_TILE_COLS * PLAY_TILE_WIDTH
    video_height = PLAY_TILE_ROWS * PLAY_TILE_HEIGHT

    out_path = args.play_output
    if not os.path.isabs(out_path):
        out_path = os.path.join(ROOT_DIR, "logs", train_cfg.runner.experiment_name, out_path)

    metrics_out_path = _resolve_metrics_output(args, out_path)

    video = FfmpegVideoWriter(out_path, video_width, video_height, PLAY_VIDEO_FPS)
    print(f"[play_residual] recording video: {out_path}")

    num_steps = int(args.play_duration / env.dt)
    record_every = max(1, int(round(1.0 / (PLAY_VIDEO_FPS * env.dt))))

    try:
        for i in range(num_steps):
            env.commands[:, 0] = float(args.play_vx)
            env.commands[:, 1] = float(args.play_vy)
            env.commands[:, 2] = float(args.play_yaw)
            env.commands[:, 3] = 0.0

            with torch.no_grad():
                actions = policy.act_inference(obs)

            if hasattr(policy, "last_delta_norm"):
                delta_norms.append(float(policy.last_delta_norm.detach().cpu().item()))
            if hasattr(policy, "last_saturation_ratio"):
                saturation_ratios.append(float(policy.last_saturation_ratio.detach().cpu().item()))

            obs, privileged_obs, rewards, costs, dones, infos = env.step(actions)

            collect_metrics_step(env, metric_env_id, i * env.dt, metrics)

            if i % record_every == 0:
                frame = capture_mosaic_frame(env, video_env_ids, cam_handles, i, args)
                video.write(frame)

    finally:
        video.close()
        print(f"[play_residual] video saved: {out_path}")
        save_metrics_plot(env, metrics, metric_env_id, metrics_out_path, args)
        print_residual_summary(policy, delta_norms, saturation_ratios)


if __name__ == "__main__":
    register_all_tasks()
    args = get_residual_play_args()
    play(args)
```
