from isaacgym import gymutil
import isaacgym  # noqa: F401

import os
import sys
from copy import deepcopy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

import torch

from global_config import ROOT_DIR
from tasks import register_all_tasks
from utils.helpers import class_to_dict
from utils.task_registry import task_registry
from modules.model_loader import load_actor_critic_checkpoint

from play_climb_adjustable import (
    set_eval_terrain,
    select_env_ids,
    create_cameras,
    capture_mosaic_frame,
    PLAY_TILE_ROWS,
    PLAY_TILE_COLS,
    PLAY_TILE_WIDTH,
    PLAY_TILE_HEIGHT,
    PLAY_VIDEO_FPS,
)
from utils.video_recorder import FfmpegVideoWriter
from play_residual_adjustable import (
    build_actor_critic,
    make_residual_policy,
)


SCENARIO_IDS = {
    "normal": 0,
    "stair": 1,
    "slip": 2,
    "recovery": 3,
    "stair_slip": 4,
}


def get_args():
    custom_parameters = [
        {"name": "--headless", "action": "store_true", "default": False,
         "help": "Run without viewer."},

        {"name": "--scenario", "type": str, "default": "normal",
         "help": "normal, stair, slip, recovery, stair_slip"},

        {"name": "--controller", "type": str, "default": "base",
         "help": "base or residual"},

        {"name": "--task", "type": str, "default": "d1h_base",
         "help": "Environment task used for data collection."},

        {"name": "--base_task", "type": str, "default": "d1h_base",
         "help": "Base policy task name."},

        {"name": "--base_ckpt", "type": str, "default": None,
         "help": "Base policy checkpoint."},

        {"name": "--residual_ckpt", "type": str, "default": None,
         "help": "Residual expert checkpoint. Required if controller=residual."},

        {"name": "--load_run", "type": str, "default": None,
         "help": "Residual run folder if residual_ckpt is omitted."},

        {"name": "--checkpoint", "type": int, "default": -1,
         "help": "Residual checkpoint id if residual_ckpt is omitted."},

        {"name": "--residual_alpha", "type": float, "default": 0.45},
        {"name": "--residual_delta_clip", "type": float, "default": 0.55},
        {"name": "--residual_std_min", "type": float, "default": 0.20},
        {"name": "--residual_std_max", "type": float, "default": 0.65},

        {"name": "--num_envs", "type": int, "default": 256},
        {"name": "--steps", "type": int, "default": 2000},
        {"name": "--sample_every", "type": int, "default": 4},

        {"name": "--play_vx", "type": float, "default": 0.35},
        {"name": "--play_vy", "type": float, "default": 0.0},
        {"name": "--play_yaw", "type": float, "default": 0.0},
        {"name": "--vx_min", "type": float, "default": None},
        {"name": "--vx_max", "type": float, "default": None},
        {"name": "--vy_min", "type": float, "default": None},
        {"name": "--vy_max", "type": float, "default": None},
        {"name": "--yaw_min", "type": float, "default": None},
        {"name": "--yaw_max", "type": float, "default": None},

        {"name": "--cmd_resample_steps", "type": int, "default": 250,
         "help": "Resample commands every N policy steps during dataset collection."},

        {"name": "--stand_prob", "type": float, "default": 0.10,
         "help": "Probability of setting vx, vy, yaw to zero when resampling commands."},

        {"name": "--play_stair_height", "type": float, "default": 0.12},
        {"name": "--play_step_width", "type": float, "default": 0.55},
        {"name": "--play_slope", "type": float, "default": 0.03},

        {"name": "--label_window", "type": int, "default": 5,
         "help": "Causal past-window max smoothing for dynamic scores."},

        {"name": "--add_noise", "action": "store_true", "default": False,
         "help": "Keep observation noise on. Default false for cleaner first dataset."},

        {"name": "--output", "type": str, "default": "data/moe_terrain/normal_000.pt"},

        {"name": "--record_video", "action": "store_true", "default": False,
         "help": "Record a 2x2 debug video during dataset collection."},

        {"name": "--video_output", "type": str, "default": "",
         "help": "Output mp4 path. Default: <output>.mp4"},

        {"name": "--video_num_envs", "type": int, "default": 4,
         "help": "Number of envs to show in the debug video."},

        {"name": "--video_every", "type": int, "default": 4,
         "help": "Write one video frame every N policy steps."},

        {"name": "--video_fps", "type": int, "default": 0,
         "help": "Video fps. 0 means auto from env.dt and video_every."},
    ]

    args = gymutil.parse_arguments(
        description="Collect proprioceptive terrain estimator dataset.",
        custom_parameters=custom_parameters,
    )

    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"

    return args


def resolve_path(path):
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(ROOT_DIR, path)


def set_attr_if_exists(obj, name, value):
    if hasattr(obj, name):
        setattr(obj, name, value)


def set_terrain_proportions(env_cfg, proportions):
    if hasattr(env_cfg, "terrain"):
        env_cfg.terrain.terrain_proportions = proportions


def apply_scenario_cfg(env_cfg, args):
    """
    Reuse play_climb_adjustable.set_eval_terrain first, then override scenario-specific settings.
    """
    args.play_num_envs = int(args.num_envs)

    if args.scenario == "normal":
        args.play_terrain = "mixed"
    elif args.scenario == "stair":
        args.play_terrain = "stairs_up"
    elif args.scenario == "slip":
        args.play_terrain = "slope"
    elif args.scenario == "recovery":
        args.play_terrain = "mixed"
    elif args.scenario == "stair_slip":
        args.play_terrain = "stairs_up"
    else:
        raise ValueError(f"Unknown scenario: {args.scenario}")

    set_eval_terrain(env_cfg, args)

    env_cfg.env.num_envs = int(args.num_envs)

    if hasattr(env_cfg, "terrain"):
        env_cfg.terrain.measure_heights = True
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.num_rows = 1
        env_cfg.terrain.num_cols = max(4, int(args.num_envs))

    if hasattr(env_cfg, "noise"):
        env_cfg.noise.add_noise = bool(args.add_noise)

    if hasattr(env_cfg, "commands"):
        env_cfg.commands.curriculum = False
        if hasattr(env_cfg.commands, "ranges"):
            vx_min, vx_max, vy_min, vy_max, yaw_min, yaw_max = get_command_ranges(args)
            env_cfg.commands.ranges.lin_vel_x = [vx_min, vx_max]
            env_cfg.commands.ranges.lin_vel_y = [vy_min, vy_max]
            env_cfg.commands.ranges.ang_vel_yaw = [yaw_min, yaw_max]

    # Scenario-specific terrain distribution.
    # Terrain branch order used by play_climb_adjustable:
    # [slope, rough_slope, stair_down_branch, stair_up_branch, obstacles, stones, gap, pit, ...]
    if args.scenario == "normal":
        set_terrain_proportions(env_cfg, [0.55, 0.45, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    elif args.scenario == "stair":
        # Terrain order:
        # [slope, rough_slope, stairs_up, stairs_down, obstacles, stones, gap, pit, ...]
        set_terrain_proportions(env_cfg, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    elif args.scenario == "slip":
        set_terrain_proportions(env_cfg, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    elif args.scenario == "recovery":
        set_terrain_proportions(env_cfg, [0.50, 0.50, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    elif args.scenario == "stair_slip":
        # Terrain order:
        # [slope, rough_slope, stairs_up, stairs_down, obstacles, stones, gap, pit, ...]
        set_terrain_proportions(env_cfg, [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # Friction settings.
    if hasattr(env_cfg, "domain_rand"):
        if args.scenario in ["slip", "stair_slip"]:
            set_attr_if_exists(env_cfg.domain_rand, "randomize_friction", True)
            set_attr_if_exists(env_cfg.domain_rand, "friction_range", [0.08, 0.55])
        else:
            set_attr_if_exists(env_cfg.domain_rand, "randomize_friction", True)
            set_attr_if_exists(env_cfg.domain_rand, "friction_range", [0.60, 1.80])

        if args.scenario == "recovery":
            set_attr_if_exists(env_cfg.domain_rand, "push_robots", True)
            set_attr_if_exists(env_cfg.domain_rand, "push_interval_s", 2.0)
            set_attr_if_exists(env_cfg.domain_rand, "max_push_vel_xy", 0.6)

            set_attr_if_exists(env_cfg.domain_rand, "disturbance", True)
            set_attr_if_exists(env_cfg.domain_rand, "disturbance_range", [-35.0, 35.0])
            set_attr_if_exists(env_cfg.domain_rand, "disturbance_interval", 100)
        else:
            set_attr_if_exists(env_cfg.domain_rand, "push_robots", False)
            set_attr_if_exists(env_cfg.domain_rand, "disturbance", False)

    # Recovery reset ranges if the task supports them.
    if args.scenario == "recovery" and hasattr(env_cfg, "init_state"):
        set_attr_if_exists(env_cfg.init_state, "recovery_roll_range", [-0.60, 0.60])
        set_attr_if_exists(env_cfg.init_state, "recovery_pitch_range", [-0.60, 0.60])
        set_attr_if_exists(env_cfg.init_state, "recovery_yaw_range", [-0.30, 0.30])
        set_attr_if_exists(env_cfg.init_state, "recovery_init_z_range", [0.28, 0.56])
        set_attr_if_exists(env_cfg.init_state, "recovery_folded_prob", 0.20)

def default_command_ranges_for_scenario(scenario):
    if scenario == "normal":
        return {
            "vx": [-0.20, 0.70],
            "vy": [-0.12, 0.12],
            "yaw": [-0.25, 0.25],
        }

    if scenario == "stair":
        return {
            "vx": [0.15, 0.50],
            "vy": [-0.05, 0.05],
            "yaw": [-0.12, 0.12],
        }

    if scenario == "slip":
        return {
            "vx": [0.10, 0.55],
            "vy": [-0.05, 0.05],
            "yaw": [-0.15, 0.15],
        }

    if scenario == "recovery":
        return {
            "vx": [0.00, 0.30],
            "vy": [-0.04, 0.04],
            "yaw": [-0.10, 0.10],
        }

    if scenario == "stair_slip":
        return {
            "vx": [0.15, 0.40],
            "vy": [-0.04, 0.04],
            "yaw": [-0.10, 0.10],
        }

    raise ValueError(f"Unknown scenario: {scenario}")


def get_command_ranges(args):
    defaults = default_command_ranges_for_scenario(args.scenario)

    vx_min = defaults["vx"][0] if args.vx_min is None else float(args.vx_min)
    vx_max = defaults["vx"][1] if args.vx_max is None else float(args.vx_max)

    vy_min = defaults["vy"][0] if args.vy_min is None else float(args.vy_min)
    vy_max = defaults["vy"][1] if args.vy_max is None else float(args.vy_max)

    yaw_min = defaults["yaw"][0] if args.yaw_min is None else float(args.yaw_min)
    yaw_max = defaults["yaw"][1] if args.yaw_max is None else float(args.yaw_max)

    return vx_min, vx_max, vy_min, vy_max, yaw_min, yaw_max


def resample_commands(env, args):
    vx_min, vx_max, vy_min, vy_max, yaw_min, yaw_max = get_command_ranges(args)

    B = env.num_envs
    device = env.device

    env.commands[:, 0] = torch.empty(B, device=device).uniform_(vx_min, vx_max)
    env.commands[:, 1] = torch.empty(B, device=device).uniform_(vy_min, vy_max)
    env.commands[:, 2] = torch.empty(B, device=device).uniform_(yaw_min, yaw_max)
    env.commands[:, 3] = 0.0

    stand_mask = torch.rand(B, device=device) < float(args.stand_prob)
    env.commands[stand_mask, 0] = 0.0
    env.commands[stand_mask, 1] = 0.0
    env.commands[stand_mask, 2] = 0.0

def make_base_policy(env, base_train_cfg, base_ckpt):
    train_cfg_dict = class_to_dict(base_train_cfg)
    policy_class_name = train_cfg_dict["runner"]["policy_class_name"]
    policy_cfg = train_cfg_dict["policy"]

    policy = build_actor_critic(policy_class_name, env, policy_cfg)

    ckpt = resolve_path(base_ckpt)
    if ckpt is None:
        raise ValueError("--base_ckpt is required.")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"Base checkpoint does not exist: {ckpt}")

    load_actor_critic_checkpoint(policy, ckpt, env.device)
    policy.eval().to(env.device)
    print(f"[collect] loaded base checkpoint: {ckpt}")

    return policy


def policy_act(policy, obs):
    if hasattr(policy, "act_inference"):
        return policy.act_inference(obs)
    if hasattr(policy, "act_student"):
        return policy.act_student(obs)
    if hasattr(policy, "act_teacher"):
        return policy.act_teacher(obs)
    raise RuntimeError(f"Policy has no inference method: {type(policy)}")


def print_terrain_distribution(env):
    print("\n[collect] ===== actual terrain info =====")

    if hasattr(env, "cfg") and hasattr(env.cfg, "terrain"):
        terrain = env.cfg.terrain
        print("[collect] mesh_type:", getattr(terrain, "mesh_type", None))
        print("[collect] terrain_proportions:", getattr(terrain, "terrain_proportions", None))
        print("[collect] num_rows:", getattr(terrain, "num_rows", None))
        print("[collect] num_cols:", getattr(terrain, "num_cols", None))

    if hasattr(env, "terrain_types"):
        tt = env.terrain_types.detach().cpu().long()
        unique, counts = torch.unique(tt, return_counts=True)
        print("[collect] terrain_types count:")
        for u, c in zip(unique.tolist(), counts.tolist()):
            print(f"  type={u}: count={c}")
    else:
        print("[collect] env has no terrain_types")

    if hasattr(env, "terrain_levels"):
        lv = env.terrain_levels.detach().cpu().long()
        unique, counts = torch.unique(lv, return_counts=True)
        print("[collect] terrain_levels count:")
        for u, c in zip(unique.tolist(), counts.tolist()):
            print(f"  level={u}: count={c}")

    print("[collect] =================================\n")


def safe_mean(heights, mask):
    if mask.sum().item() == 0:
        return torch.zeros(heights.shape[0], device=heights.device)
    return heights[:, mask].mean(dim=1)


def safe_max(heights, mask):
    if mask.sum().item() == 0:
        return torch.zeros(heights.shape[0], device=heights.device)
    return heights[:, mask].max(dim=1).values


def compute_raw_labels(env):
    """
    Returns:
        vel_label:  [B, 3]
        gray_label: [B, 5]
            0 step_up_score
            1 slope_score
            2 traction_loss_score
            3 instability_score
            4 stall_score
        debug: dict
    """
    device = env.device
    B = env.num_envs

    if hasattr(env, "measured_heights") and torch.is_tensor(env.measured_heights):
        heights = env.measured_heights
    else:
        heights = torch.zeros(B, 1, device=device)

    if hasattr(env, "height_points") and torch.is_tensor(env.height_points):
        hp = env.height_points[0]
        x = hp[:, 0]
        y = hp[:, 1]
    else:
        x = torch.zeros(heights.shape[1], device=device)
        y = torch.zeros(heights.shape[1], device=device)

    near_mask = (torch.abs(x) < 0.10) & (torch.abs(y) < 0.25)
    front_mask = (x > 0.25) & (x < 0.80) & (torch.abs(y) < 0.30)

    front_slope_mask = (x > 0.35) & (x < 0.80) & (torch.abs(y) < 0.35)
    back_slope_mask = (x < -0.35) & (x > -0.80) & (torch.abs(y) < 0.35)

    center_mask = (torch.abs(x) < 0.20) & (torch.abs(y) < 0.20)

    near_h = safe_mean(heights, near_mask)
    front_h = safe_max(heights, front_mask)

    dh_front = front_h - near_h

    step_up_score = torch.clamp(
        (dh_front - 0.025) / 0.12,
        0.0,
        1.0,
    )

    front_mean = safe_mean(heights, front_slope_mask)
    back_mean = safe_mean(heights, back_slope_mask)

    slope_raw = front_mean - back_mean
    slope_score = torch.clamp(
        slope_raw / 0.25,
        -1.0,
        1.0,
    )

    base_vx = env.base_lin_vel[:, 0]
    base_vy = env.base_lin_vel[:, 1]
    cmd_x = env.commands[:, 0]

    if hasattr(env, "friction_coeffs_tensor") and torch.is_tensor(env.friction_coeffs_tensor):
        mu = env.friction_coeffs_tensor.view(B, -1)[:, 0]
    else:
        mu = torch.ones(B, device=device)

    mu_loss = torch.clamp(
        (0.65 - mu) / 0.55,
        0.0,
        1.0,
    )

    side_slip = torch.clamp(
        (torch.abs(base_vy) - 0.06) / 0.20,
        0.0,
        1.0,
    )

    vx_overspeed = torch.clamp(
        (base_vx - cmd_x - 0.08) / 0.40,
        0.0,
        1.0,
    )

    wheel_speed = torch.abs(env.dof_vel[:, [3, 7]]).mean(dim=1)

    wheel_spin = torch.clamp(
        (wheel_speed - 8.0) / 20.0,
        0.0,
        1.0,
    )

    observed_slip = torch.maximum(
        torch.maximum(side_slip, vx_overspeed),
        wheel_spin,
    )

    motion_gate = (
        (torch.abs(cmd_x) > 0.10)
        | (wheel_speed > 5.0)
    ).float()

    traction_loss_score = torch.maximum(
        observed_slip,
        0.6 * mu_loss * motion_gate,
    )

    upright = -env.projected_gravity[:, 2]

    tilt_bad = torch.clamp(
        (0.90 - upright) / 0.25,
        0.0,
        1.0,
    )

    ground_center = safe_mean(heights, center_mask)
    base_height = env.root_states[:, 2] - ground_center

    height_bad = torch.clamp(
        (0.38 - base_height) / 0.16,
        0.0,
        1.0,
    )

    angvel_xy = torch.norm(env.base_ang_vel[:, :2], dim=1)

    angvel_bad = torch.clamp(
        (angvel_xy - 0.8) / 2.0,
        0.0,
        1.0,
    )

    instability_score = torch.maximum(
        torch.maximum(tilt_bad, height_bad),
        angvel_bad,
    )

    forward_cmd = (cmd_x > 0.12).float()

    progress_error = torch.clamp(
        (cmd_x - base_vx - 0.10) / 0.35,
        0.0,
        1.0,
    )

    wheel_active = torch.clamp(
        (wheel_speed - 4.0) / 12.0,
        0.0,
        1.0,
    )

    stall_score = forward_cmd * progress_error * wheel_active * (
        1.0 - 0.7 * traction_loss_score
    )
    stall_score = torch.clamp(stall_score, 0.0, 1.0)

    vel_label = env.base_lin_vel.clone()

    gray_label = torch.stack(
        [
            step_up_score,
            slope_score,
            traction_loss_score,
            instability_score,
            stall_score,
        ],
        dim=-1,
    )

    debug = {
        "dh_front": dh_front,
        "mu": mu,
        "cmd_x": cmd_x,
        "base_vx": base_vx,
        "base_vy": base_vy,
        "wheel_speed": wheel_speed,
        "base_height": base_height,

        "mu_loss": mu_loss,
        "side_slip": side_slip,
        "vx_overspeed": vx_overspeed,
        "wheel_spin": wheel_spin,
        "tilt_bad": tilt_bad,
        "height_bad": height_bad,
        "angvel_bad": angvel_bad,
    }

    return vel_label, gray_label, debug


def get_obs_hist(env):
    """
    Use causal proprioceptive history maintained by env.
    Shape: [B, history_len, n_proprio]
    """
    return env.obs_history_buf.clone()


def main():
    register_all_tasks()
    args = get_args()

    if args.scenario not in SCENARIO_IDS:
        raise ValueError(f"--scenario must be one of {list(SCENARIO_IDS.keys())}")

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    _, base_train_cfg = task_registry.get_cfgs(name=args.base_task)

    apply_scenario_cfg(env_cfg, args)

    args.num_envs = int(args.num_envs)
    env_cfg.env.num_envs = int(args.num_envs)

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)

    print_terrain_distribution(env)

    env.reset()
    obs = env.get_observations()

    video_writer = None
    video_env_ids = []
    video_cam_handles = []

    if bool(args.record_video):
        video_env_ids = select_env_ids(env.num_envs, int(args.video_num_envs), seed=0)
        video_cam_handles = create_cameras(env, video_env_ids)

        video_output = args.video_output
        if not video_output:
            root, _ = os.path.splitext(args.output)
            video_output = root + ".mp4"
        if not os.path.isabs(video_output):
            video_output = os.path.join(ROOT_DIR, video_output)

        if int(args.video_fps) > 0:
            video_fps = int(args.video_fps)
        else:
            env_dt = float(getattr(env, "dt", 0.01))
            video_fps = max(1, int(round(1.0 / (env_dt * max(1, int(args.video_every))))))

        video_writer = FfmpegVideoWriter(
            video_output,
            PLAY_TILE_COLS * PLAY_TILE_WIDTH,
            PLAY_TILE_ROWS * PLAY_TILE_HEIGHT,
            video_fps,
        )
        print("[collect] recording video:", video_output)
        print("[collect] video_fps:", video_fps)

    if args.controller == "base":
        policy = make_base_policy(env, base_train_cfg, args.base_ckpt)
    elif args.controller == "residual":
        policy = make_residual_policy(env, args, train_cfg, base_train_cfg)
    else:
        raise ValueError("--controller must be base or residual")

    label_window = max(1, int(args.label_window))
    label_hist = torch.zeros(
        env.num_envs,
        label_window,
        5,
        device=env.device,
        dtype=torch.float,
    )

    obs_hist_list = []
    vel_label_list = []
    gray_label_list = []
    scenario_id_list = []
    debug_list = []

    scenario_id = SCENARIO_IDS[args.scenario]
    scenario_tensor = torch.full(
        (env.num_envs,),
        int(scenario_id),
        device=env.device,
        dtype=torch.long,
    )

    print("[collect] scenario     =", args.scenario)
    print("[collect] controller   =", args.controller)
    print("[collect] task         =", args.task)
    print("[collect] num_envs     =", env.num_envs)
    print("[collect] steps        =", args.steps)
    print("[collect] sample_every =", args.sample_every)

    with torch.no_grad():
        for i in range(int(args.steps)):
            if i == 0 or i % int(args.cmd_resample_steps) == 0:
                resample_commands(env, args)

            actions = policy_act(policy, obs)

            obs, privileged_obs, rewards, costs, dones, infos = env.step(actions)

            if video_writer is not None and (i % int(args.video_every) == 0):
                frame = capture_mosaic_frame(env, video_env_ids, video_cam_handles, i, args)
                video_writer.write(frame)

            vel_label, raw_gray_label, debug = compute_raw_labels(env)

            label_hist = torch.cat(
                [
                    label_hist[:, 1:, :],
                    raw_gray_label.unsqueeze(1),
                ],
                dim=1,
            )

            gray_label = raw_gray_label.clone()

            # Causal past-window smoothing only for dynamic states.
            # order: step_up, slope, traction, instability, stall
            gray_label[:, 2:5] = label_hist[:, :, 2:5].max(dim=1).values

            if i % int(args.sample_every) == 0:
                obs_hist_list.append(get_obs_hist(env).detach().cpu().to(torch.float16))
                vel_label_list.append(vel_label.detach().cpu().to(torch.float32))
                gray_label_list.append(gray_label.detach().cpu().to(torch.float32))
                scenario_id_list.append(scenario_tensor.detach().cpu())

                debug_list.append(
                    torch.stack(
                        [
                            debug["dh_front"],
                            debug["mu"],
                            debug["cmd_x"],
                            debug["base_vx"],
                            debug["base_vy"],
                            debug["wheel_speed"],
                            debug["base_height"],

                            debug["mu_loss"],
                            debug["side_slip"],
                            debug["vx_overspeed"],
                            debug["wheel_spin"],
                            debug["tilt_bad"],
                            debug["height_bad"],
                            debug["angvel_bad"],
                        ],
                        dim=-1,
                    ).detach().cpu().to(torch.float32)
                )

            if (i + 1) % 200 == 0:
                n = sum(x.shape[0] for x in gray_label_list)
                print(f"[collect] step {i + 1}/{args.steps}, samples={n}")

    obs_hist = torch.cat(obs_hist_list, dim=0)
    vel_label = torch.cat(vel_label_list, dim=0)
    gray_label = torch.cat(gray_label_list, dim=0)
    scenario_ids = torch.cat(scenario_id_list, dim=0)
    debug = torch.cat(debug_list, dim=0)

    out_path = args.output
    if not os.path.isabs(out_path):
        out_path = os.path.join(ROOT_DIR, out_path)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    torch.save(
        {
            "obs_hist": obs_hist,
            "vel_label": vel_label,
            "gray_label": gray_label,
            "scenario_id": scenario_ids,
            "debug": debug,
            "label_names": [
                "step_up_score",
                "slope_score",
                "traction_loss_score",
                "instability_score",
                "stall_score",
            ],
            "debug_names": [
                "dh_front",
                "mu",
                "cmd_x",
                "base_vx",
                "base_vy",
                "wheel_speed",
                "base_height",
                "mu_loss",
                "side_slip",
                "vx_overspeed",
                "wheel_spin",
                "tilt_bad",
                "height_bad",
                "angvel_bad",
            ],
            "scenario": args.scenario,
        },
        out_path,
    )

    if video_writer is not None:
        video_writer.close()
        print("[collect] video closed")

    print("[collect] saved:", out_path)
    print("[collect] obs_hist:", tuple(obs_hist.shape))
    print("[collect] vel_label:", tuple(vel_label.shape))
    print("[collect] gray_label:", tuple(gray_label.shape))


if __name__ == "__main__":
    main()
