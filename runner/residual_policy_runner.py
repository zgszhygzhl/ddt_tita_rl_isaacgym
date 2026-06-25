import os

import numpy as np
import torch

from algorithm import NP3O
from runner.on_constraint_policy_runner import OnConstraintPolicyRunner


class ResidualPolicyRunner(OnConstraintPolicyRunner):
    def __init__(self, env, train_cfg, actor_critic, log_dir=None, device="cpu", reset_residual_std=None):
        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.depth_encoder_cfg = train_cfg.get("depth_encoder", {"if_depth": False})
        self.depth_encoder_cfg["if_depth"] = False
        self.if_depth = False
        self.device = device
        self.env = env
        self.current_learning_iteration = 0

        actor_critic.to(self.device)

        depth_encoder = None
        depth_actor = None

        self.alg_cfg["k_value"] = self.env.cost_k_values
        self.alg = NP3O(
            actor_critic,
            depth_encoder,
            self.depth_encoder_cfg,
            depth_actor,
            device=self.device,
            **self.alg_cfg,
        )

        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.dagger_update_freq = self.alg_cfg["dagger_update_freq"]

        if hasattr(self.env.cfg, "cost"):
            num_costs = self.env.cfg.cost.num_costs
        elif hasattr(self.env.cfg, "costs"):
            num_costs = self.env.cfg.costs.num_costs
        else:
            raise AttributeError("env.cfg must have cost or costs config with num_costs")

        self.alg.init_storage(
            self.env.num_envs,
            self.num_steps_per_env,
            [self.env.num_obs],
            [self.env.num_privileged_obs],
            [self.env.num_actions],
            [num_costs],
            self.env.cost_d_values_tensor,
        )

        self.log_dir = log_dir
        self.checkpoint_dir = None
        if self.log_dir is not None:
            self.checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0

        if self.cfg.get("resume", False):
            self._load_training_checkpoint()

        if reset_residual_std is not None:
            if not hasattr(actor_critic, "set_residual_std"):
                raise AttributeError("actor_critic does not support resetting residual std")
            actor_critic.set_residual_std(reset_residual_std)
            print("[ResidualPolicyRunner] reset residual std to:", reset_residual_std)

        self.record_video = bool(self.cfg.get("record_video", False)) and self.log_dir is not None
        self.video_interval = int(self.cfg.get("video_interval", 500))
        self.video_duration = float(self.cfg.get("video_duration", 8.0))
        self.video_fps = int(self.cfg.get("video_fps", 30))
        self.video_num_envs = int(self.cfg.get("video_num_envs", 16))
        self.video_tile_rows = int(self.cfg.get("video_tile_rows", 4))
        self.video_tile_cols = int(self.cfg.get("video_tile_cols", 4))
        self.video_tile_width = int(self.cfg.get("video_tile_width", 320))
        self.video_tile_height = int(self.cfg.get("video_tile_height", 180))
        self.video_width = self.video_tile_cols * self.video_tile_width
        self.video_height = self.video_tile_rows * self.video_tile_height
        self.video_env_ids = []
        self.video_cam_handles = []
        self.video_writer = None
        self.video_steps_left = 0
        self.video_step_count = 0
        self.video_record_every = 1
        self.video_current_iteration = 0
        self.video_black_tile = np.zeros(
            (self.video_tile_height, self.video_tile_width, 3), dtype=np.uint8
        )

        self.env.reset()
        if self.record_video:
            self._setup_train_video_camera()

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference

    def get_actor_critic(self, device=None):
        self.alg.actor_critic.eval()
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic