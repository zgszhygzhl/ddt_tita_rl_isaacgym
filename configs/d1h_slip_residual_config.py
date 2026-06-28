import torch
from isaacgym.torch_utils import quat_rotate_inverse

from configs.d1h_base_config import (
    D1hBase,
    D1hBaseCfg,
    D1hBaseCfgPPO,
)


class D1hSlipResidualCfg(D1hBaseCfg):
    """
    Low-friction / slippery-road residual expert.

    目标：
        frozen d1h_base + residual slip expert

    重点：
        1. 低摩擦光滑路面；
        2. 抑制横滑；
        3. 抑制轮子空转；
        4. 抑制 vx 超命令；
        5. 动作更柔和，力矩不过饱和。
    """

    class env(D1hBaseCfg.env):
        num_envs = 4096

    class commands(D1hBaseCfg.commands):
        # 光滑路面上不要一开始训练太大速度，先学稳。
        commands_proportion = [0.65, 0.05, 0.05, 0.05, 0.03, 0.03, 0.04, 0.05, 0.05]

        curriculum = False

        max_lin_vel_x_change_rate = 0.25
        max_lin_vel_y_change_rate = 0.08
        max_ang_vel_change_rate = 0.15

        enable_command_buffer = True
        buffer_smoothing_factor = 0.08

        class ranges(D1hBaseCfg.commands.ranges):
            lin_vel_x = [-0.6, 0.6]
            lin_vel_y = [-0.2, 0.2]
            ang_vel_yaw = [-0.10, 0.10]
            heading = [-0.3, 0.3]

    class rewards(D1hBaseCfg.rewards):
        base_height_target = 0.45

        # 防止接近力矩极限。
        soft_torque_limit = 0.80

        # slip expert 专用参数。
        slip_contact_force = 3.0
        wheel_spin_deadband = 8.0
        vx_overspeed_deadband = 0.08
        vx_overspeed_max = 0.8

        class scales(D1hBaseCfg.rewards.scales):
            termination = -100.0

            # 速度跟踪保留，但不要太猛。
            tracking_lin_vel = 0.0
            tracking_lin_vel_x = 10.0
            tracking_lin_vel_y = 2.0
            tracking_ang_vel = 2.0

            # 姿态稳定。
            lin_vel_z = -2.0
            ang_vel_xy = -0.20
            orientation = -12.0
            base_height = -10.0

            # 低摩擦下关键：动作柔和、力矩不过饱和。
            torques = -2.0e-5
            torque_limits = -0.25
            powers = 0.0
            dof_acc = -8.0e-7
            action_rate = -0.25
            action_smoothness = 0.0

            collision = -10.0
            stand_still = -1.0

            feet_air_time = 0.0
            foot_clearance = 0.0
            stumble = 0.0

            # 光滑路面上鼓励双轮持续接触，不鼓励跳跃步态。
            no_gait = 2.0
            both_feet_air = 0.0
            upward_vel_spike = 0.0
            contact_upward_bounce = 0.0

            # 支撑几何，保持温和。
            body_pos_to_feet_x = 1.0
            body_feet_distance_x = -8.0
            body_feet_distance_y = -60.0
            body_symmetry_y = 0.2
            body_symmetry_z = 0.1

            heading = 3.0
            upward = 1.0
            head_los_distance = -15.0

            # slip 专用项。
            slip_lateral = -8.0
            wheel_spin = -0.03
            base_sideslip = -4.0
            vx_overspeed = -6.0

    class domain_rand(D1hBaseCfg.domain_rand):
        # 低摩擦范围是这个专家的核心。
        randomize_friction = True
        friction_range = [0.08, 0.55]

        randomize_restitution = True
        restitution_range = [0.0, 0.15]

        randomize_base_mass = True
        added_mass_range = [-0.5, 1.5]

        randomize_base_com = True
        added_com_range = [-0.04, 0.04]

        # 先不加外部 push，避免把“低摩擦防滑”和“抗扰动恢复”混在一起。
        push_robots = False
        push_interval_s = 10
        max_push_vel_xy = 0.2

        randomize_motor = True
        motor_strength_range = [0.90, 1.10]

        randomize_kpkd = True
        kp_range = [0.90, 1.10]
        kd_range = [0.90, 1.10]

        randomize_lag_timesteps = True
        lag_timesteps = 3

        disturbance = False
        disturbance_range = [-20.0, 20.0]
        disturbance_interval = 100

    class terrain(D1hBaseCfg.terrain):
        mesh_type = 'trimesh'
        measure_heights = True

        # 光滑路面：只用平滑坡面，不用 rough / stairs / discrete。
        # [smooth slope, rough slope, stairs down, stairs up, discrete, stones, gap, pit, ...]
        terrain_proportions = [0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.0]

        slope = [0.0, 0.05]
        step_height = [0.02, 0.04]
        step_width_range = [0.55, 0.65]
        discrete_obstacles_height = [0.02, 0.06]
        pit_depth = [0.0, 0.2]

        slope_treshold = 0.75
        max_init_terrain_level = 4
        curriculum = False


class D1hSlipResidualCfgPPO(D1hBaseCfgPPO):
    class algorithm(D1hBaseCfgPPO.algorithm):
        entropy_coef = 0.008
        learning_rate = 1.0e-3
        max_grad_norm = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        cost_value_loss_coef = 0.1
        cost_viol_loss_coef = 0.1

        # slip expert 不应该大幅改 base，只做低摩擦补偿。
        residual_l2_coef = 0.002

    class policy(D1hBaseCfgPPO.policy):
        init_noise_std = 1.0
        continue_from_last_std = True

        scan_encoder_dims = [64, 32]
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [256, 128, 64]
        priv_encoder_dims = []

        barlow_actor_hidden_dims = [256, 128, 64]
        barlow_mlp_encoder_dims = [256, 128, 64]
        barlow_latent_dim = 16
        barlow_obs_encoder_dims = [128, 64]
        barlow_num_hist = 10

        activation = 'elu'
        rnn_type = 'lstm'
        rnn_hidden_size = 256
        rnn_num_layers = 1

        tanh_encoder_output = False
        num_costs = 6

        teacher_act = True
        imi_flag = True

    class runner(D1hBaseCfgPPO.runner):
        run_name = 'd1h_slip_residual'
        experiment_name = 'd1h_slip_residual'
        policy_class_name = 'ActorCriticBarlowTwins'
        runner_class_name = 'OnConstraintPolicyRunner'
        algorithm_class_name = 'NP3O'

        max_iterations = 5000
        num_steps_per_env = 24
        save_interval = 200

        record_video = True
        video_interval = 400
        video_duration = 6.0
        video_fps = 30
        video_num_envs = 16
        video_tile_rows = 4
        video_tile_cols = 4
        video_tile_width = 320
        video_tile_height = 180


class D1hSlipResidual(D1hBase):
    """
    Slippery-road residual task.

    新增 reward：
        slip_lateral:     接触状态下足端横向速度，抑制横滑；
        wheel_spin:       接触状态下轮关节过快转动，抑制空转；
        base_sideslip:    base 横向速度，抑制车身侧滑；
        vx_overspeed:     实际 vx 明显超过命令，抑制低摩擦滑冲。
    """

    def _contact_mask(self):
        contact_force = getattr(self.cfg.rewards, "slip_contact_force", 3.0)
        return self.contact_forces[:, self.feet_indices, 2] > contact_force

    def _foot_vel_base(self):
        num_feet = self.feet_indices.shape[0]
        quat = self.base_quat.unsqueeze(1).repeat(1, num_feet, 1).reshape(-1, 4)
        vel = self.foot_velocities.reshape(-1, 3)
        vel_base = quat_rotate_inverse(quat, vel).view(self.num_envs, num_feet, 3)
        return vel_base

    def _reward_slip_lateral(self):
        # 接触时足端/轮子在 body-y 方向的速度越大，说明越在横滑。
        contacts = self._contact_mask().float()
        foot_vel_base = self._foot_vel_base()
        lateral_vel = foot_vel_base[:, :, 1]
        return torch.sum(torch.square(lateral_vel) * contacts, dim=1)

    def _reward_wheel_spin(self):
        # 接触时轮关节速度过大，视为空转倾向。
        # D1H 的轮关节索引按当前工程为 [3, 7]。
        contacts = self._contact_mask().float()
        wheel_ids = torch.tensor([3, 7], dtype=torch.long, device=self.device)

        spin = torch.abs(self.dof_vel[:, wheel_ids])
        deadband = float(getattr(self.cfg.rewards, "wheel_spin_deadband", 8.0))
        spin_excess = torch.clamp(spin - deadband, min=0.0)

        return torch.sum(torch.square(spin_excess) * contacts, dim=1)

    def _reward_base_sideslip(self):
        # 正常低摩擦前进时，不希望 base 有明显 y 方向滑移。
        return torch.square(self.base_lin_vel[:, 1])

    def _reward_vx_overspeed(self):
        # 低摩擦下常见问题：实际 vx 被滑出去，明显超过命令。
        cmd_x = self.commands_given[:, 0]
        vx = self.base_lin_vel[:, 0]

        deadband = float(getattr(self.cfg.rewards, "vx_overspeed_deadband", 0.08))
        max_excess = float(getattr(self.cfg.rewards, "vx_overspeed_max", 0.8))

        excess = torch.clamp(vx - cmd_x - deadband, min=0.0, max=max_excess)
        return torch.square(excess)
