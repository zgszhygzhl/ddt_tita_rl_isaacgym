"""Adjustable inference / video recording for D1H/Y1v0h stair climbing.

Usage example:
python play_climb_adjustable.py \
  --task=d1h_evt1_climb \
  --load_run Jun24_00-31-58_d1h_evt1_climb \
  --checkpoint 12000 \
  --headless \
  --play_vx 0.35 \
  --play_vy 0.0 \
  --play_yaw 0.0 \
  --play_terrain stairs_up \
  --play_stair_height 0.08 \
  --play_num_envs 16 \
  --play_duration 20
"""
from isaacgym import gymapi
import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont

from envs import LeggedRobot
from modules import *  # noqa: F401,F403
from utils import get_args, get_load_path, task_registry
from utils.helpers import class_to_dict
from utils.video_recorder import FfmpegVideoWriter
from global_config import ROOT_DIR

# Robust imports for the two layouts used in this project.
try:
    from configs.y1v0h_evt1.y1v0h_evt1_climb_config import (
        Y1v0hEvt1Climb,
        Y1v0hEvt1ClimbCfg,
        Y1v0hEvt1ClimbCfgPPO,
    )
except Exception:
    from configs.y1v0h_evt1_climb_config import (  # type: ignore
        Y1v0hEvt1Climb,
        Y1v0hEvt1ClimbCfg,
        Y1v0hEvt1ClimbCfgPPO,
    )


PLAY_TILE_ROWS = 2
PLAY_TILE_COLS = 2
PLAY_TILE_WIDTH = 640
PLAY_TILE_HEIGHT = 360
PLAY_VIDEO_FPS = 30


def parse_adjustable_args():
    """Parse play-only args first, then let the repo's get_args parse the rest."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--play_vx", type=float, default=0.35)
    parser.add_argument("--play_vy", type=float, default=0.0)
    parser.add_argument("--play_yaw", type=float, default=0.0)
    parser.add_argument("--play_duration", type=float, default=20.0)
    parser.add_argument("--play_num_envs", type=int, default=16)
    parser.add_argument("--play_video_num_envs", type=int, default=4)
    parser.add_argument(
        "--play_terrain",
        type=str,
        default="stairs_up",
        choices=["stairs_up", "stairs_down", "slope", "rough_slope", "mixed"],
    )
    parser.add_argument(
        "--play_stair_height",
        type=float,
        default=0.08,
        help="Fixed stair height in meters. Applies to both step_height and stairs_max_height when available.",
    )
    parser.add_argument("--play_step_width", type=float, default=0.55)
    parser.add_argument("--play_slope", type=float, default=0.0)
    parser.add_argument("--play_seed", type=int, default=0)
    parser.add_argument("--play_output", type=str, default="play_adjustable.mp4")
    parser.add_argument(
        "--play_metrics_output",
        type=str,
        default="",
        help="PNG path for torque and command-tracking curves. Default: <play_output>_metrics.png",
    )
    parser.add_argument(
        "--play_metrics_env_id",
        type=int,
        default=-1,
        help="Environment id used for metrics plotting. Default: first selected video env.",
    )
    parser.add_argument(
        "--play_metrics_max_points",
        type=int,
        default=2500,
        help="Maximum points drawn in the PNG; data are downsampled only for plotting.",
    )

    play_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining
    base_args = get_args()
    return base_args, play_args


def register_tasks():
    task_registry.register(
        "d1h_evt1_climb",
        Y1v0hEvt1Climb,
        Y1v0hEvt1ClimbCfg(),
        Y1v0hEvt1ClimbCfgPPO(),
    )


def set_attr_if_exists(obj, name, value):
    if hasattr(obj, name):
        setattr(obj, name, value)


def set_eval_terrain(env_cfg, play_args):
    """Make terrain deterministic and adjustable for inference."""
    env_cfg.env.num_envs = min(int(env_cfg.env.num_envs), int(play_args.play_num_envs))

    # Disable training-time randomization/noise for clean inference comparison.
    if hasattr(env_cfg, "noise"):
        env_cfg.noise.add_noise = False

    if hasattr(env_cfg, "domain_rand"):
        for name in [
            "push_robots",
            "randomize_friction",
            "randomize_restitution",
            "randomize_base_com",
            "randomize_base_mass",
            "randomize_motor",
            "randomize_kpkd",
            "randomize_lag_timesteps",
            "disturbance",
        ]:
            set_attr_if_exists(env_cfg.domain_rand, name, False)

    # Disable command curriculum during inference; commands are fixed in the loop.
    if hasattr(env_cfg, "commands"):
        set_attr_if_exists(env_cfg.commands, "curriculum", False)
        if hasattr(env_cfg.commands, "ranges"):
            env_cfg.commands.ranges.lin_vel_x = [play_args.play_vx, play_args.play_vx]
            env_cfg.commands.ranges.lin_vel_y = [play_args.play_vy, play_args.play_vy]
            env_cfg.commands.ranges.ang_vel_yaw = [play_args.play_yaw, play_args.play_yaw]

    if not hasattr(env_cfg, "terrain"):
        return

    terrain = env_cfg.terrain
    set_attr_if_exists(terrain, "mesh_type", "trimesh")
    set_attr_if_exists(terrain, "curriculum", False)
    set_attr_if_exists(terrain, "selected", False)
    set_attr_if_exists(terrain, "measure_heights", True)
    set_attr_if_exists(terrain, "num_rows", 1)
    set_attr_if_exists(terrain, "num_cols", max(4, int(play_args.play_num_envs)))
    set_attr_if_exists(terrain, "max_init_terrain_level", 0)

    # Fixed terrain height/shape. Set both names because different branches use different names.
    h = float(play_args.play_stair_height)
    set_attr_if_exists(terrain, "step_height", [h, h])
    set_attr_if_exists(terrain, "stairs_max_height", h)

    if hasattr(terrain, "step_width_range"):
        terrain.step_width_range = [float(play_args.play_step_width), float(play_args.play_step_width)]
    if hasattr(terrain, "step_width"):
        terrain.step_width = float(play_args.play_step_width)

    if hasattr(terrain, "slope"):
        terrain.slope = [float(play_args.play_slope), float(play_args.play_slope)]

    # Match utils/terrain.py branch order:
    # [slope, rough_slope, first stair branch, second stair branch, obstacles, stones, gap, pit, ...]
    # In many IsaacGym stair generators the sign of step_height may swap visual up/down.
    # If you find the direction reversed, swap stairs_up and stairs_down below.
    if play_args.play_terrain == "slope":
        terrain.terrain_proportions = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    elif play_args.play_terrain == "rough_slope":
        terrain.terrain_proportions = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    elif play_args.play_terrain == "stairs_down":
        terrain.terrain_proportions = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    elif play_args.play_terrain == "stairs_up":
        terrain.terrain_proportions = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    else:
        # Keep the training distribution.
        pass


def get_policy_checkpoint(args, train_cfg):
    load_run = args.load_run if args.load_run is not None else getattr(train_cfg.runner, "load_run", -1)
    checkpoint = args.checkpoint if args.checkpoint is not None else getattr(train_cfg.runner, "checkpoint", -1)
    log_root = os.path.join(ROOT_DIR, "logs", train_cfg.runner.experiment_name)
    return get_load_path(log_root, load_run=load_run, checkpoint=checkpoint)


def select_env_ids(num_envs, num_selected, seed=0):
    num_selected = min(int(num_selected), int(num_envs))
    if num_selected <= 0:
        return []
    rng = np.random.default_rng(seed)
    return rng.choice(num_envs, size=num_selected, replace=False).astype(int).tolist()


def create_cameras(env, env_ids):
    camera_props = gymapi.CameraProperties()
    camera_props.width = PLAY_TILE_WIDTH
    camera_props.height = PLAY_TILE_HEIGHT
    cam_handles = []
    for env_id in env_ids:
        cam_handle = env.gym.create_camera_sensor(env.envs[env_id], camera_props)
        cam_handles.append(cam_handle)
    update_camera_locations(env, env_ids, cam_handles)
    return cam_handles


def update_camera_locations(env, env_ids, cam_handles):
    for env_id, cam_handle in zip(env_ids, cam_handles):
        origin = env.env_origins[env_id].detach().cpu().numpy()
        cam_pos = gymapi.Vec3(float(origin[0] + 2.8), float(origin[1] - 4.2), float(origin[2] + 1.8))
        cam_target = gymapi.Vec3(float(origin[0] + 0.45), float(origin[1]), float(origin[2] + 0.65))
        env.gym.set_camera_location(cam_handle, env.envs[env_id], cam_pos, cam_target)


def terrain_text(env, env_id, fixed_height=None):
    if not hasattr(env, "terrain_levels"):
        return "terrain=plane"
    level = int(env.terrain_levels[env_id].item())
    nrows = int(getattr(env.cfg.terrain, "num_rows", 1))
    ncols = int(getattr(env.cfg.terrain, "num_cols", 1))
    diff = level / max(float(nrows), 1.0)
    terrain_type = int(env.terrain_types[env_id].item()) if hasattr(env, "terrain_types") else -1

    h_cm = None
    if fixed_height is not None:
        h_cm = 100.0 * float(fixed_height)
    elif hasattr(env.cfg.terrain, "step_height"):
        step_min = float(env.cfg.terrain.step_height[0])
        step_max = float(env.cfg.terrain.step_height[1])
        h_cm = 100.0 * (step_min + (step_max - step_min) * diff)
    elif hasattr(env.cfg.terrain, "stairs_max_height"):
        h_cm = 100.0 * float(env.cfg.terrain.stairs_max_height) * diff

    max_level = max(nrows - 1, 0)
    if h_cm is None:
        return f"L={level}/{max_level} D={diff:.2f} T={terrain_type}/{ncols}"
    return f"L={level}/{max_level} D={diff:.2f} H={h_cm:.1f}cm T={terrain_type}/{ncols}"


def draw_overlay(frame, lines, font_size=16):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size)
    except Exception:
        font = ImageFont.load_default()

    padding = 8
    gap = 4
    bboxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
    box_w = max(b[2] - b[0] for b in bboxes) + 2 * padding
    box_h = sum(b[3] - b[1] for b in bboxes) + gap * (len(lines) - 1) + 2 * padding
    draw.rectangle([0, 0, box_w, box_h], fill=(0, 0, 0, 170))
    y = padding
    for line, b in zip(lines, bboxes):
        draw.text((padding, y), line, font=font, fill=(255, 255, 255, 255))
        y += (b[3] - b[1]) + gap
    return np.asarray(image, dtype=np.uint8)


def annotate_tile(frame, env, env_id, step_i, play_args):
    cmd = env.commands[env_id, :3].detach().cpu().numpy()
    cmd_given = env.commands_given[env_id, :3].detach().cpu().numpy() if hasattr(env, "commands_given") else cmd
    vel_x = float(env.base_lin_vel[env_id, 0].item())
    vel_y = float(env.base_lin_vel[env_id, 1].item())
    yaw = float(env.base_ang_vel[env_id, 2].item())
    lines = [
        f"env={env_id} step={step_i}",
        f"cmd=({cmd[0]:+.2f},{cmd[1]:+.2f},{cmd[2]:+.2f})",
        f"given=({cmd_given[0]:+.2f},{cmd_given[1]:+.2f},{cmd_given[2]:+.2f})",
        f"vel=({vel_x:+.2f},{vel_y:+.2f},{yaw:+.2f})",
        f"{play_args.play_terrain} {terrain_text(env, env_id, fixed_height=play_args.play_stair_height)}",
    ]
    return draw_overlay(frame, lines)


def capture_mosaic_frame(env, env_ids, cam_handles, step_i, play_args):
    update_camera_locations(env, env_ids, cam_handles)
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)

    total_tiles = PLAY_TILE_ROWS * PLAY_TILE_COLS
    black_tile = np.zeros((PLAY_TILE_HEIGHT, PLAY_TILE_WIDTH, 3), dtype=np.uint8)
    tiles = []
    for env_id, cam_handle in zip(env_ids, cam_handles):
        image = env.gym.get_camera_image(env.sim, env.envs[env_id], cam_handle, gymapi.IMAGE_COLOR)
        frame = np.asarray(image, dtype=np.uint8).reshape((PLAY_TILE_HEIGHT, PLAY_TILE_WIDTH, 4))[:, :, :3].copy()
        frame = annotate_tile(frame, env, env_id, step_i, play_args)
        tiles.append(frame)

    while len(tiles) < total_tiles:
        tiles.append(black_tile)

    rows = []
    for row_idx in range(PLAY_TILE_ROWS):
        start = row_idx * PLAY_TILE_COLS
        rows.append(np.concatenate(tiles[start:start + PLAY_TILE_COLS], axis=1))
    return np.concatenate(rows, axis=0)



def _safe_dof_names(env):
    """Return DOF names if the environment exposes them; otherwise use joint_0..."""
    names = getattr(env, "dof_names", None)
    if names is None:
        num_dof = int(getattr(env, "num_dof", 0))
        return [f"joint_{i}" for i in range(num_dof)]
    return [str(x) for x in names]


def _resolve_metrics_output(play_args, video_out_path):
    """Resolve metric PNG path from CLI arg or video output path."""
    if play_args.play_metrics_output:
        out_path = play_args.play_metrics_output
        if not os.path.isabs(out_path):
            out_path = os.path.join(ROOT_DIR, "logs", out_path)
        return out_path
    root, _ = os.path.splitext(video_out_path)
    return root + "_metrics.png"


def _downsample_for_plot(arr, max_points):
    """Uniformly downsample a 1D/2D numpy array for readable plotting."""
    arr = np.asarray(arr)
    n = arr.shape[0]
    max_points = int(max_points)
    if max_points <= 0 or n <= max_points:
        return arr, np.arange(n)
    idx = np.linspace(0, n - 1, max_points).astype(np.int64)
    return arr[idx], idx


def collect_metrics_step(env, metric_env_id, t_value, storage):
    """Collect one timestep of torques and command tracking for one env."""
    eid = int(metric_env_id)
    storage["t"].append(float(t_value))

    torque = env.torques[eid].detach().cpu().numpy().copy() if hasattr(env, "torques") else np.zeros(int(getattr(env, "num_actions", 0)))
    storage["torques"].append(torque)

    cmd = env.commands[eid, :3].detach().cpu().numpy().copy() if hasattr(env, "commands") else np.zeros(3)
    if hasattr(env, "commands_given"):
        cmd_given = env.commands_given[eid, :3].detach().cpu().numpy().copy()
    else:
        cmd_given = cmd.copy()

    vel = np.array([
        float(env.base_lin_vel[eid, 0].item()),
        float(env.base_lin_vel[eid, 1].item()),
        float(env.base_ang_vel[eid, 2].item()),
    ], dtype=np.float32)

    storage["cmd"].append(cmd)
    storage["cmd_given"].append(cmd_given)
    storage["vel"].append(vel)


def save_metrics_plot(env, storage, metric_env_id, out_png, play_args):
    """Save one PNG dashboard: joint torques + vx/vy/yaw command tracking."""
    if len(storage["t"]) == 0:
        print("[play] no metrics collected; skip metrics plot")
        return

    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    t = np.asarray(storage["t"], dtype=np.float32)
    torques = np.asarray(storage["torques"], dtype=np.float32)
    cmd = np.asarray(storage["cmd"], dtype=np.float32)
    cmd_given = np.asarray(storage["cmd_given"], dtype=np.float32)
    vel = np.asarray(storage["vel"], dtype=np.float32)

    max_points = int(getattr(play_args, "play_metrics_max_points", 2500))
    t_ds, idx = _downsample_for_plot(t, max_points)
    torques_ds = torques[idx]
    cmd_ds = cmd[idx]
    cmd_given_ds = cmd_given[idx]
    vel_ds = vel[idx]

    dof_names = _safe_dof_names(env)
    if len(dof_names) != torques.shape[1]:
        dof_names = [f"joint_{i}" for i in range(torques.shape[1])]

    torque_limits = None
    if hasattr(env, "torque_limits"):
        torque_limits = env.torque_limits.detach().cpu().numpy().astype(np.float32)
        if torque_limits.shape[0] != torques.shape[1]:
            torque_limits = None

    abs_torque = np.abs(torques)
    max_abs = abs_torque.max(axis=0)
    rms = np.sqrt(np.mean(np.square(torques), axis=0))
    if torque_limits is not None:
        eps = 1e-6
        sat = abs_torque >= (0.98 * torque_limits[None, :] - eps)
        sat_ratio = 100.0 * sat.mean(axis=0)
    else:
        sat_ratio = np.zeros_like(max_abs)

    fig, axes = plt.subplots(4, 1, figsize=(15, 14), sharex=True)

    ax = axes[0]
    for j in range(torques_ds.shape[1]):
        label = dof_names[j]
        if torque_limits is not None:
            label = f"{label} max={max_abs[j]:.1f}/{torque_limits[j]:.1f}Nm sat={sat_ratio[j]:.1f}%"
        else:
            label = f"{label} max={max_abs[j]:.1f}Nm"
        ax.plot(t_ds, torques_ds[:, j], linewidth=1.0, label=label)
    if torque_limits is not None and len(np.unique(np.round(torque_limits, 4))) == 1:
        lim = float(torque_limits[0])
        ax.axhline(lim, linestyle="--", linewidth=0.8)
        ax.axhline(-lim, linestyle="--", linewidth=0.8)
    ax.set_ylabel("torque [Nm]")
    ax.set_title(
        f"Joint torques and command tracking | env={metric_env_id} | "
        f"terrain={play_args.play_terrain} | duration={t[-1]:.2f}s"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=7, ncol=2)

    labels = [
        ("vx", "linear x [m/s]", 0),
        ("vy", "linear y [m/s]", 1),
        ("yaw", "yaw rate [rad/s]", 2),
    ]
    for ax, (name, ylabel, k) in zip(axes[1:], labels):
        ax.plot(t_ds, cmd_ds[:, k], linewidth=1.0, label=f"cmd {name}")
        ax.plot(t_ds, cmd_given_ds[:, k], linewidth=1.0, label=f"given {name}")
        ax.plot(t_ds, vel_ds[:, k], linewidth=1.0, label=f"actual {name}")
        err = vel[:, k] - cmd_given[:, k]
        mae = float(np.mean(np.abs(err)))
        rmse = float(np.sqrt(np.mean(np.square(err))))
        ax.set_ylabel(ylabel)
        ax.set_title(f"{name} tracking: MAE={mae:.3f}, RMSE={rmse:.3f}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("time [s]")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    print(f"[play] metrics plot saved: {out_png}")

    # Also print a compact torque summary for quick terminal inspection.
    print("[play] torque summary for plotted env:")
    for j, name in enumerate(dof_names):
        if torque_limits is not None:
            print(f"  {j:02d} {name:>16s}: max_abs={max_abs[j]:7.2f} Nm  rms={rms[j]:7.2f} Nm  limit={torque_limits[j]:7.2f} Nm  sat={sat_ratio[j]:5.1f}%")
        else:
            print(f"  {j:02d} {name:>16s}: max_abs={max_abs[j]:7.2f} Nm  rms={rms[j]:7.2f} Nm")


def play(args, play_args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    set_eval_terrain(env_cfg, play_args)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.reset()
    obs = env.get_observations()

    policy_cfg_dict = class_to_dict(train_cfg.policy)
    runner_cfg_dict = class_to_dict(train_cfg.runner)
    actor_critic_class = eval(runner_cfg_dict["policy_class_name"])
    policy = actor_critic_class(
        env.cfg.env.n_proprio,
        env.cfg.env.n_scan,
        env.num_obs,
        env.cfg.env.n_priv_latent,
        env.cfg.env.history_len,
        env.num_actions,
        **policy_cfg_dict,
    )

    ckpt = get_policy_checkpoint(args, train_cfg)
    model = torch.load(ckpt, map_location=env.device)
    policy.load_state_dict(model["model_state_dict"])
    policy.eval().to(env.device)
    print(f"[play] loaded checkpoint: {ckpt}")

    video_env_ids = select_env_ids(env.num_envs, play_args.play_video_num_envs, seed=play_args.play_seed)
    cam_handles = create_cameras(env, video_env_ids)
    print(f"[play] selected video envs: {video_env_ids}")

    if int(play_args.play_metrics_env_id) >= 0:
        metric_env_id = min(int(play_args.play_metrics_env_id), env.num_envs - 1)
    else:
        metric_env_id = int(video_env_ids[0]) if len(video_env_ids) > 0 else 0
    print(f"[play] metrics env: {metric_env_id}")
    metrics = {"t": [], "torques": [], "cmd": [], "cmd_given": [], "vel": []}

    video_width = PLAY_TILE_COLS * PLAY_TILE_WIDTH
    video_height = PLAY_TILE_ROWS * PLAY_TILE_HEIGHT
    out_path = play_args.play_output
    if not os.path.isabs(out_path):
        out_path = os.path.join(ROOT_DIR, "logs", train_cfg.runner.experiment_name, out_path)
    metrics_out_path = _resolve_metrics_output(play_args, out_path)
    video = FfmpegVideoWriter(out_path, video_width, video_height, PLAY_VIDEO_FPS)
    print(f"[play] recording video: {out_path}")

    num_steps = int(play_args.play_duration / env.dt)
    record_every = max(1, int(round(1.0 / (PLAY_VIDEO_FPS * env.dt))))

    try:
        for i in range(num_steps):
            env.commands[:, 0] = float(play_args.play_vx)
            env.commands[:, 1] = float(play_args.play_vy)
            env.commands[:, 2] = float(play_args.play_yaw)
            env.commands[:, 3] = 0.0

            with torch.no_grad():
                if hasattr(policy, "act_teacher"):
                    actions = policy.act_teacher(obs)
                else:
                    actions = policy.act_inference(obs)
            obs, privileged_obs, rewards, costs, dones, infos = env.step(actions)
            collect_metrics_step(env, metric_env_id, i * env.dt, metrics)

            if i % record_every == 0:
                frame = capture_mosaic_frame(env, video_env_ids, cam_handles, i, play_args)
                video.write(frame)
    finally:
        video.close()
        print(f"[play] video saved: {out_path}")
        save_metrics_plot(env, metrics, metric_env_id, metrics_out_path, play_args)


if __name__ == "__main__":
    register_tasks()
    args, play_args = parse_adjustable_args()
    play(args, play_args)
