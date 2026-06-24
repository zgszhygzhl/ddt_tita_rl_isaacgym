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

    video_width = PLAY_TILE_COLS * PLAY_TILE_WIDTH
    video_height = PLAY_TILE_ROWS * PLAY_TILE_HEIGHT
    out_path = play_args.play_output
    if not os.path.isabs(out_path):
        out_path = os.path.join(ROOT_DIR, "logs", train_cfg.runner.experiment_name, out_path)
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

            if i % record_every == 0:
                frame = capture_mosaic_frame(env, video_env_ids, cam_handles, i, play_args)
                video.write(frame)
    finally:
        video.close()
        print(f"[play] video saved: {out_path}")


if __name__ == "__main__":
    register_tasks()
    args, play_args = parse_adjustable_args()
    play(args, play_args)
