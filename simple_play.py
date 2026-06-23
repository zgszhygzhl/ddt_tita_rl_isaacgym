import os

import numpy as np
import torch
from isaacgym import gymapi

from configs.tita_constraint_config import TitaConstraintRoughCfg, TitaConstraintRoughCfgPPO
from configs.d1h_constraint_config import D1HConstraintRoughCfg, D1HConstraintRoughCfgPPO
from configs.y1v0h_evt1_climb_config import (
    Y1v0hEvt1Climb,
    Y1v0hEvt1ClimbCfg,
    Y1v0hEvt1ClimbCfgPPO,
)
from envs import LeggedRobot
from modules import *
from utils import get_args, get_load_path, task_registry
from utils.helpers import class_to_dict
from utils.video_recorder import FfmpegVideoWriter
from global_config import ROOT_DIR


# Play/evaluation video layout.
# Create 24 candidate environments, then randomly select 4 for a 2x2 mosaic.
PLAY_NUM_ENVS = 24
PLAY_VIDEO_NUM_ENVS = 4
PLAY_TILE_ROWS = 2
PLAY_TILE_COLS = 2
PLAY_TILE_WIDTH = 640
PLAY_TILE_HEIGHT = 360
PLAY_VIDEO_DURATION = 20.0
PLAY_VIDEO_FPS = 30


def register_tasks():
    task_registry.register(
        "tita_constraint",
        LeggedRobot,
        TitaConstraintRoughCfg(),
        TitaConstraintRoughCfgPPO(),
    )
    task_registry.register(
        "d1h_constraint",
        LeggedRobot,
        D1HConstraintRoughCfg(),
        D1HConstraintRoughCfgPPO(),
    )
    task_registry.register(
        "d1h_evt1_climb",
        Y1v0hEvt1Climb,
        Y1v0hEvt1ClimbCfg(),
        Y1v0hEvt1ClimbCfgPPO(),
    )


def _get_policy_checkpoint(args, train_cfg):
    load_run = args.load_run if args.load_run is not None else getattr(train_cfg.runner, "load_run", -1)
    checkpoint = args.checkpoint if args.checkpoint is not None else getattr(train_cfg.runner, "checkpoint", -1)
    log_root = os.path.join(ROOT_DIR, "logs", train_cfg.runner.experiment_name)
    return get_load_path(log_root, load_run=load_run, checkpoint=checkpoint)


def _disable_eval_randomization(env_cfg):
    env_cfg.noise.add_noise = False

    if hasattr(env_cfg.terrain, "curriculum"):
        env_cfg.terrain.curriculum = False
    if hasattr(env_cfg.terrain, "num_rows"):
        env_cfg.terrain.num_rows = 5
    if hasattr(env_cfg.terrain, "num_cols"):
        env_cfg.terrain.num_cols = 5

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
        if hasattr(env_cfg.domain_rand, name):
            setattr(env_cfg.domain_rand, name, False)

    if hasattr(env_cfg.control, "use_filter"):
        env_cfg.control.use_filter = True


def _select_random_env_ids(num_envs, num_selected, seed=None):
    num_selected = min(int(num_selected), int(num_envs))
    if num_selected <= 0:
        return []
    rng = np.random.default_rng(seed)
    return rng.choice(num_envs, size=num_selected, replace=False).astype(int).tolist()


def _create_play_cameras(env, env_ids):
    camera_props = gymapi.CameraProperties()
    camera_props.width = PLAY_TILE_WIDTH
    camera_props.height = PLAY_TILE_HEIGHT

    cam_handles = []
    for env_id in env_ids:
        cam_handle = env.gym.create_camera_sensor(env.envs[env_id], camera_props)
        cam_handles.append(cam_handle)
    _update_play_camera_locations(env, env_ids, cam_handles)
    return cam_handles


def _update_play_camera_locations(env, env_ids, cam_handles):
    for env_id, cam_handle in zip(env_ids, cam_handles):
        origin = env.env_origins[env_id].detach().cpu().numpy()
        cam_pos = gymapi.Vec3(
            float(origin[0] + 2.8),
            float(origin[1] - 4.2),
            float(origin[2] + 1.8),
        )
        cam_target = gymapi.Vec3(
            float(origin[0] + 0.45),
            float(origin[1] + 0.0),
            float(origin[2] + 0.65),
        )
        env.gym.set_camera_location(cam_handle, env.envs[env_id], cam_pos, cam_target)


def _capture_mosaic_frame(env, env_ids, cam_handles):
    _update_play_camera_locations(env, env_ids, cam_handles)
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)

    total_tiles = PLAY_TILE_ROWS * PLAY_TILE_COLS
    black_tile = np.zeros((PLAY_TILE_HEIGHT, PLAY_TILE_WIDTH, 3), dtype=np.uint8)
    tiles = []

    for env_id, cam_handle in zip(env_ids, cam_handles):
        image = env.gym.get_camera_image(
            env.sim,
            env.envs[env_id],
            cam_handle,
            gymapi.IMAGE_COLOR,
        )
        frame = np.asarray(image, dtype=np.uint8).reshape(
            (PLAY_TILE_HEIGHT, PLAY_TILE_WIDTH, 4)
        )[:, :, :3]
        tiles.append(frame)

    while len(tiles) < total_tiles:
        tiles.append(black_tile)

    rows = []
    for row_index in range(PLAY_TILE_ROWS):
        row_start = row_index * PLAY_TILE_COLS
        row_tiles = tiles[row_start:row_start + PLAY_TILE_COLS]
        rows.append(np.concatenate(row_tiles, axis=1))
    return np.concatenate(rows, axis=0)


def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    env_cfg.env.num_envs = min(env_cfg.env.num_envs, PLAY_NUM_ENVS)
    _disable_eval_randomization(env_cfg)

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

    resume_path = _get_policy_checkpoint(args, train_cfg)
    model_dict = torch.load(resume_path, map_location=env.device)
    policy.load_state_dict(model_dict["model_state_dict"])
    policy.eval()
    policy = policy.to(env.device)
    print(f"[play] loaded checkpoint: {resume_path}")

    seed = args.seed if getattr(args, "seed", None) is not None else None
    video_env_ids = _select_random_env_ids(env.num_envs, PLAY_VIDEO_NUM_ENVS, seed=seed)
    cam_handles = _create_play_cameras(env, video_env_ids)
    print(f"[play] selected video envs: {video_env_ids}")

    video_width = PLAY_TILE_COLS * PLAY_TILE_WIDTH
    video_height = PLAY_TILE_ROWS * PLAY_TILE_HEIGHT
    video_path = os.path.join(
        ROOT_DIR,
        "logs",
        train_cfg.runner.experiment_name,
        "play_record.mp4",
    )
    video = FfmpegVideoWriter(video_path, video_width, video_height, PLAY_VIDEO_FPS)
    print(f"[play] recording 2x2 video: {video_path}")

    num_steps = int(PLAY_VIDEO_DURATION / env.dt)
    record_every = max(1, int(round(1.0 / (PLAY_VIDEO_FPS * env.dt))))
    status_every = max(1, int(round(1.0 / env.dt)))

    try:
        for i in range(num_steps):
            # Fixed inference command for video. Change this if you want another test.
            env.commands[:, 0] = 0.4
            env.commands[:, 1] = 0.0
            env.commands[:, 2] = 0.0
            env.commands[:, 3] = 0.0

            with torch.no_grad():
                if hasattr(policy, "act_teacher"):
                    actions = policy.act_teacher(obs)
                else:
                    actions = policy.act_inference(obs)

            obs, privileged_obs, rewards, costs, dones, infos = env.step(actions)

            if i % record_every == 0:
                frame = _capture_mosaic_frame(env, video_env_ids, cam_handles)
                video.write(frame)

            if i % status_every == 0:
                robot_index = video_env_ids[0] if video_env_ids else 0
                print(
                    f"step={i:05d} "
                    f"env={robot_index} "
                    f"cmd=({env.commands[robot_index, 0].item():+.2f}, "
                    f"{env.commands[robot_index, 1].item():+.2f}, "
                    f"{env.commands[robot_index, 2].item():+.2f}) "
                    f"vel=({env.base_lin_vel[robot_index, 0].item():+.2f}, "
                    f"{env.base_lin_vel[robot_index, 1].item():+.2f}, "
                    f"{env.base_ang_vel[robot_index, 2].item():+.2f})"
                )
    finally:
        video.close()
        print(f"[play] video saved: {video_path}")


if __name__ == "__main__":
    register_tasks()
    args = get_args()
    play(args)
