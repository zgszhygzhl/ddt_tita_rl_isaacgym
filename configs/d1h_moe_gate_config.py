import torch
from isaacgym.torch_utils import quat_rotate_inverse

from configs.d1h_disc_residual_config import (
    D1hDiscResidual,
    D1hDiscResidualCfg,
    D1hDiscResidualCfgPPO,
)


class D1hMoEGateCfg(D1hDiscResidualCfg):
    class terrain(D1hDiscResidualCfg.terrain):
        mesh_type = "trimesh"
        measure_heights = True
        curriculum = False
        num_rows = 8
        num_cols = 20
        # Project terrain order: slope, rough slope, stairs up, stairs down, discrete.
        terrain_proportions = [0.20, 0.15, 0.40, 0.05, 0.20, 0.0, 0.0, 0.0, 0.0]
        slope = [0.0, 0.04]
        step_height = [0.04, 0.15]
        step_width_range = [0.50, 0.62]
        discrete_obstacles_height = [0.03, 0.12]
        pit_depth = [0.0, 0.20]
        slope_treshold = 0.2
        max_init_terrain_level = 5

    class commands(D1hDiscResidualCfg.commands):
        curriculum = False
        resampling_time = 6
        commands_proportion = [0.75, 0.05, 0.05, 0.05, 0.025, 0.025, 0.025, 0.0, 0.025]
        max_lin_vel_x_change_rate = 0.5
        max_lin_vel_y_change_rate = 0.15
        max_ang_vel_change_rate = 0.25
        enable_command_buffer = True
        buffer_smoothing_factor = 0.1

        class ranges(D1hDiscResidualCfg.commands.ranges):
            lin_vel_x = [0.15, 0.55]
            lin_vel_y = [-0.08, 0.08]
            ang_vel_yaw = [-0.15, 0.15]
            heading = [-0.5, 0.5]

    class domain_rand(D1hDiscResidualCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.12, 1.80]
        randomize_restitution = True
        restitution_range = [0.0, 0.25]
        randomize_base_mass = True
        added_mass_range = [-1.0, 3.0]
        randomize_base_com = True
        added_com_range = [-0.08, 0.08]
        push_robots = True
        push_interval_s = 5.0
        max_push_vel_xy = 0.45
        randomize_motor = True
        motor_strength_range = [0.9, 1.1]
        randomize_kpkd = True
        kp_range = [0.9, 1.1]
        kd_range = [0.9, 1.1]
        randomize_lag_timesteps = True
        lag_timesteps = 3
        disturbance = True
        disturbance_range = [-25.0, 25.0]
        disturbance_interval = 6

    class rewards(D1hDiscResidualCfg.rewards):
        slip_contact_force = 3.0
        wheel_spin_deadband = 8.0
        vx_overspeed_deadband = 0.08
        vx_overspeed_max = 0.8

        class scales(D1hDiscResidualCfg.rewards.scales):
            slip_lateral = -4.0
            wheel_spin = -0.015
            base_sideslip = -2.0
            vx_overspeed = -3.0


class D1hMoEGateCfgPPO(D1hDiscResidualCfgPPO):
    class algorithm(D1hDiscResidualCfgPPO.algorithm):
        entropy_coef = 0.01
        learning_rate = 3.0e-4
        max_grad_norm = 0.5
        num_learning_epochs = 5
        num_mini_batches = 4
        cost_value_loss_coef = 0.1
        cost_viol_loss_coef = 0.1
        residual_l2_coef = 0.01
        gate_aux_coef = 0.20

    class policy(D1hDiscResidualCfgPPO.policy):
        init_noise_std = 0.35
        continue_from_last_std = False
        activation = "elu"
        num_costs = 6
        teacher_act = False
        imi_flag = False
        gate_hidden_dims = [128, 64]
        critic_hidden_dims = [256, 128, 64]
        # Kept for backward compatibility; ignored by sigmoid gate.
        gate_top_k = 2
        gate_temperature = 1.0
        gate_init_weight = 0.05
        residual_alpha = 0.60
        residual_delta_clip = 0.0
        base_ckpt = ""
        stair_ckpt = ""
        slip_ckpt = ""
        recovery_ckpt = ""
        estimator_ckpt = ""
        base_policy_class_name = "ActorCriticBarlowTwins"
        stair_policy_class_name = "ActorCriticBarlowTwins"
        slip_policy_class_name = "ActorCriticBarlowTwins"
        recovery_policy_class_name = "ActorCriticBarlowTwins"
        base_policy_cfg = {}
        stair_policy_cfg = {}
        slip_policy_cfg = {}
        recovery_policy_cfg = {}

    class runner(D1hDiscResidualCfgPPO.runner):
        run_name = "gate_top2_v1"
        experiment_name = "d1h_moe_gate"
        policy_class_name = "ActorCriticMoEGate"
        runner_class_name = "OnConstraintPolicyRunner"
        algorithm_class_name = "NP3O"
        max_iterations = 4000
        num_steps_per_env = 24
        save_interval = 200
        record_video = True
        video_interval = 500
        video_duration = 6.0
        video_fps = 30
        video_num_envs = 16
        video_tile_rows = 4
        video_tile_cols = 4
        video_tile_width = 320
        video_tile_height = 180


class D1hMoEGateStairCfg(D1hMoEGateCfg):
    class terrain(D1hDiscResidualCfg.terrain):
        mesh_type = "trimesh"
        measure_heights = True
        curriculum = True
        num_rows = 8
        num_cols = 20
        # Rollout-verified mapping: third branch is stairs_up, fourth is stairs_down.
        terrain_proportions = [0.075, 0.0, 0.85, 0.075, 0.0, 0.0, 0.0, 0.0, 0.0]
        slope = [0.0, 0.02]
        step_height = [0.02, 0.16]
        step_width_range = [0.51, 0.61]
        discrete_obstacles_height = [0.05, 0.15]
        pit_depth = [0.0, 0.3]
        slope_treshold = 0.2
        max_init_terrain_level = 4

    class commands(D1hDiscResidualCfg.commands):
        curriculum = True
        max_curriculum = 0.8
        max_curriculum_x = 1.0
        max_curriculum_y = 0.15
        min_curriculum_x = 0.0
        min_curriculum_y = -0.15
        max_curriculum_z = 0.25
        commands_proportion = [0.75, 0.05, 0.05, 0.05, 0.025, 0.025, 0.025, 0.0, 0.025]
        max_lin_vel_x_change_rate = 0.5
        max_lin_vel_y_change_rate = 0.2
        max_ang_vel_change_rate = 0.3
        enable_command_buffer = True
        buffer_smoothing_factor = 0.1

        class ranges(D1hDiscResidualCfg.commands.ranges):
            lin_vel_x = [0.35, 0.55]
            lin_vel_y = [-0.08, 0.08]
            ang_vel_yaw = [-0.10, 0.10]
            heading = [-0.5, 0.5]

    class domain_rand(D1hDiscResidualCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.6, 1.8]
        randomize_restitution = True
        restitution_range = [0.0, 0.3]
        randomize_base_mass = True
        added_mass_range = [-1.0, 3.0]
        randomize_base_com = True
        added_com_range = [-0.1, 0.1]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 0.4
        randomize_motor = True
        motor_strength_range = [0.9, 1.1]
        randomize_kpkd = True
        kp_range = [0.9, 1.1]
        kd_range = [0.9, 1.1]
        randomize_lag_timesteps = True
        lag_timesteps = 3
        disturbance = False
        disturbance_range = [-20.0, 20.0]
        disturbance_interval = 8

    class rewards(D1hDiscResidualCfg.rewards):
        slip_contact_force = 3.0
        wheel_spin_deadband = 8.0
        vx_overspeed_deadband = 0.08
        vx_overspeed_max = 0.8

        class scales(D1hDiscResidualCfg.rewards.scales):
            slip_lateral = 0.0
            wheel_spin = 0.0
            base_sideslip = 0.0
            vx_overspeed = 0.0


class D1hMoEGateStairCfgPPO(D1hMoEGateCfgPPO):
    class algorithm(D1hMoEGateCfgPPO.algorithm):
        entropy_coef = 0.01
        learning_rate = 3.0e-4
        max_grad_norm = 0.5
        num_learning_epochs = 5
        num_mini_batches = 4
        cost_value_loss_coef = 0.1
        cost_viol_loss_coef = 0.1
        residual_l2_coef = 0.01
        gate_aux_coef = 0.20

    class policy(D1hMoEGateCfgPPO.policy):
        gate_init_weight = 0.05
        residual_alpha = 1.0
        residual_delta_clip = 0.0

    class runner(D1hMoEGateCfgPPO.runner):
        run_name = "gate_stair_warmup"
        experiment_name = "d1h_moe_gate_stair"
        policy_class_name = "ActorCriticMoEGate"
        runner_class_name = "OnConstraintPolicyRunner"
        algorithm_class_name = "NP3O"
        max_iterations = 2000
        num_steps_per_env = 24
        save_interval = 200
        record_video = True
        video_interval = 500
        video_duration = 6.0
        video_fps = 30
        video_num_envs = 16
        video_tile_rows = 4
        video_tile_cols = 4
        video_tile_width = 320
        video_tile_height = 180


class D1hMoEGate(D1hDiscResidual):
    def _contact_mask(self):
        contact_force = getattr(self.cfg.rewards, "slip_contact_force", 3.0)
        return self.contact_forces[:, self.feet_indices, 2] > contact_force

    def _foot_vel_base(self):
        num_feet = self.feet_indices.shape[0]
        quat = self.base_quat.unsqueeze(1).repeat(1, num_feet, 1).reshape(-1, 4)
        vel = self.foot_velocities.reshape(-1, 3)
        return quat_rotate_inverse(quat, vel).view(self.num_envs, num_feet, 3)

    def _reward_slip_lateral(self):
        contacts = self._contact_mask().float()
        lateral_vel = self._foot_vel_base()[:, :, 1]
        return torch.sum(torch.square(lateral_vel) * contacts, dim=1)

    def _reward_wheel_spin(self):
        contacts = self._contact_mask().float()
        wheel_ids = torch.tensor([3, 7], dtype=torch.long, device=self.device)
        spin = torch.abs(self.dof_vel[:, wheel_ids])
        deadband = float(getattr(self.cfg.rewards, "wheel_spin_deadband", 8.0))
        return torch.sum(torch.square(torch.clamp(spin - deadband, min=0.0)) * contacts, dim=1)

    def _reward_base_sideslip(self):
        return torch.square(self.base_lin_vel[:, 1])

    def _reward_vx_overspeed(self):
        deadband = float(getattr(self.cfg.rewards, "vx_overspeed_deadband", 0.08))
        max_excess = float(getattr(self.cfg.rewards, "vx_overspeed_max", 0.8))
        excess = torch.clamp(
            self.base_lin_vel[:, 0] - self.commands_given[:, 0] - deadband,
            min=0.0,
            max=max_excess,
        )
        return torch.square(excess)
