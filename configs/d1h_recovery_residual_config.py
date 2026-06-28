import math
import torch
from isaacgym import gymtorch
from isaacgym.torch_utils import torch_rand_float, quat_from_euler_xyz

from configs.y1v0h_evt1_climb_config import (
    Y1v0hEvt1Climb,
    Y1v0hEvt1ClimbCfg,
    Y1v0hEvt1ClimbCfgPPO,
)


class D1hRecoveryResidualCfg(Y1v0hEvt1ClimbCfg):
    """
    Recovery residual expert.

    目标：
        frozen d1h_base + residual recovery expert

    重点：
        1. 随机初始姿态；
        2. 随机初始位置，包含台阶/坡面/离散障碍；
        3. 强 push / force disturbance；
        4. 小网络快速训练；
        5. 主要学强扰动后恢复站稳，而不是重新学完整行走策略。
    """

    class env(Y1v0hEvt1ClimbCfg.env):
        num_envs = 4096

    class init_state(Y1v0hEvt1ClimbCfg.init_state):
        # recovery 初始高度由 D1hRecoveryResidual._reset_root_states 随机覆盖。
        pos = [0.0, 0.0, 0.4]

        # 随机初始姿态范围，单位 rad。
        recovery_roll_range = [-0.70, 0.70]      # 约 ±40 deg
        recovery_pitch_range = [-0.70, 0.70]     # 约 ±40 deg
        recovery_yaw_range = [-0.35, 0.35]       # 约 ±20 deg

        # 初始位置随机范围，用于让机器人出生在台阶/坡面/障碍不同相位。
        recovery_init_x_range = [-2.10, 2.10]
        recovery_init_y_range = [-0.45, 0.45]
        recovery_init_z_range = [0.24, 0.54]

        # 初始速度扰动。
        recovery_init_lin_vel_xy_range = [-0.55, 0.55]
        recovery_init_lin_vel_z_range = [-0.35, 0.45]
        recovery_init_ang_vel_xy_range = [-2.00, 2.00]
        recovery_init_ang_vel_z_range = [-1.00, 1.00]

        # 关节初始扰动。
        recovery_dof_noise = 0.35
        recovery_folded_prob = 0.2

        # 折叠/半蹲起身姿态，顺序：
        # FL_hip, FL_thigh, FL_calf, FL_foot, FR_hip, FR_thigh, FR_calf, FR_foot
        recovery_folded_joint_angles = [
            0.20, 1.30, -2.75, 0.0,
           -0.20, 1.30, -2.75, 0.0,
        ]

    class commands(Y1v0hEvt1ClimbCfg.commands):
        # recovery 以站立恢复为主，少量低速前进/转向用于兼容恢复后继续走。
        # 注意：当前 _resample_commands 里 stand_still 实际由 sum(commands_proportion[:7]) 之后的概率决定。
        # 下面前 7 项合计 0.30，所以约 70% 是站立恢复。
        # 1.x 2.y 3.xy_mix 4.spot_turn 5.x_rotation 6.y_rotation 7.xy_mix_rotation 8.stand_still
        commands_proportion = [0.15, 0.02, 0.02, 0.04, 0.02, 0.02, 0.03, 0.70, 0.0]

        curriculum = False

        max_lin_vel_x_change_rate = 0.30
        max_lin_vel_y_change_rate = 0.10
        max_ang_vel_change_rate = 0.20

        enable_command_buffer = True
        buffer_smoothing_factor = 0.1

        class ranges(Y1v0hEvt1ClimbCfg.commands.ranges):
            lin_vel_x = [-0.05, 0.25]
            lin_vel_y = [-0.03, 0.03]
            ang_vel_yaw = [-0.08, 0.08]
            heading = [-0.25, 0.25]

    class rewards(Y1v0hEvt1ClimbCfg.rewards):
        base_height_target = 0.45

        # recovery 不能太鼓励跳起来，只允许必要的短暂接触丢失。
        both_feet_air_contact_force = 3.0
        both_feet_air_grace_time = 0.020
        both_feet_air_ramp_time = 0.035

        upward_vel_spike_deadband = 0.14
        upward_vel_spike_max_excess = 1.0

        contact_upward_bounce_force = 5.0
        contact_upward_bounce_deadband = 0.12
        contact_upward_bounce_max_excess = 1.0

        # 支撑线约束稍微收紧，但不要太狠，避免台阶上必要身体摆动被压死。
        head_los_forward_offset = 0.0
        head_los_deadband = 0.035
        head_los_max_distance = 0.30

        # recovery 专用连续奖励参数。
        recovery_height_sigma = 0.012
        recovery_still_sigma = 0.25

        class scales(Y1v0hEvt1ClimbCfg.rewards.scales):
            torques = -2.0e-5
            powers = -2e-5
            termination = -100.0

            # recovery 不主追速度，速度 tracking 只保持基本命令兼容。
            tracking_lin_vel = 0.0
            tracking_lin_vel_x = 3.0
            tracking_lin_vel_y = 1.5
            tracking_ang_vel = 1.5

            # 核心：恢复直立、恢复高度、压角速度。
            orientation = -25.0
            base_height = -30.0
            lin_vel_z = -5.0
            ang_vel_xy = -1.0

            # 专用正奖励，鼓励恢复到直立、高度正确、低速度状态。
            recovery_upright = 4.0
            recovery_height = 2.0
            recovery_still = 3.0
            recovery_success = 2.0

            # 动作质量，不能太大，否则恢复动作会被压死。
            powers = -2.0e-5
            dof_acc = -5.0e-7
            action_rate = -0.15
            action_smoothness = 0.0

            collision = -20.0
            stand_still = -2.0

            # recovery 不强制步态，但也不要鼓励一直双脚同步蹬。
            feet_air_time = 0.0
            foot_clearance = 0.0
            stumble = 0.0
            no_gait = 1.0

            # 防跳、防弹。
            both_feet_air = -1.0
            upward_vel_spike = -12.0
            contact_upward_bounce = -8.0

            # 几何约束保留，但 x 方向不能太狠，否则恢复时两腿不能错开。
            body_pos_to_feet_x = 0.5
            body_feet_distance_x = -5.0
            body_feet_distance_y = -50.0
            body_symmetry_y = 0.2
            body_symmetry_z = 0.03

            heading = 1.0
            upward = 1.0
            head_los_distance = -20.0

    class domain_rand(Y1v0hEvt1ClimbCfg.domain_rand):
        # recovery 要强随机化。
        randomize_friction = True
        friction_range = [0.45, 2.20]

        randomize_restitution = True
        restitution_range = [0.0, 0.35]

        randomize_base_mass = True
        added_mass_range = [-1.5, 4.0]

        randomize_base_com = True
        added_com_range = [-0.12, 0.12]

        # 强 push，用于训练被撞/被扰动后恢复。
        push_robots = True
        push_interval_s = 8
        max_push_vel_xy = 0.7

        randomize_motor = True
        motor_strength_range = [0.85, 1.15]

        randomize_kpkd = True
        kp_range = [0.85, 1.15]
        kd_range = [0.85, 1.15]

        randomize_lag_timesteps = True
        lag_timesteps = 3

        # 外力扰动打开，比普通训练强。
        disturbance = True
        disturbance_range = [-60.0, 60.0]
        disturbance_interval = 4

    class terrain(Y1v0hEvt1ClimbCfg.terrain):
        mesh_type = 'trimesh'
        measure_heights = True

        # 顺序按当前项目实际使用：
        # [slope, rough_slope, stairs_down, stairs_up, discrete, stones, gap, pit, ...]
        # recovery 要覆盖台阶上起身，因此 stairs_up / stairs_down 比例较高。
        terrain_proportions = [0.15, 0.10, 0.30, 0.30, 0.15, 0.0, 0.0, 0.0, 0.0]

        slope = [0.0, 0.04]
        step_height = [0.02, 0.16]
        step_width_range = [0.50, 0.65]
        discrete_obstacles_height = [0.03, 0.12]
        pit_depth = [0.0, 0.20]

        slope_treshold = 0.2
        max_init_terrain_level = 5


        curriculum = False


class D1hRecoveryResidualCfgPPO(Y1v0hEvt1ClimbCfgPPO):
    class algorithm(Y1v0hEvt1ClimbCfgPPO.algorithm):
        entropy_coef = 0.01
        learning_rate = 1.0e-3
        max_grad_norm = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        cost_value_loss_coef = 0.1
        cost_viol_loss_coef = 0.1

        # recovery residual 需要比 stair 稍强一点，但不能无限乱改 base。
        residual_l2_coef = 0.0015

    class policy(Y1v0hEvt1ClimbCfgPPO.policy):
        # 小网络：recovery 主要是姿态/接触恢复，不需要太大。
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

    class runner(Y1v0hEvt1ClimbCfgPPO.runner):
        run_name = 'd1h_recovery_residual'
        experiment_name = 'd1h_recovery_residual'
        policy_class_name = 'ActorCriticBarlowTwins'
        runner_class_name = 'OnConstraintPolicyRunner'
        algorithm_class_name = 'NP3O'

        max_iterations = 6000
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


class D1hRecoveryResidual(Y1v0hEvt1Climb):
    """
    Recovery expert task.

    主要改两件事：
        1. reset_root_states：随机初始位置、姿态、速度；
        2. reset_dofs：随机关节姿态，部分环境从折叠姿态起身。
    """

    def _reset_root_states(self, env_ids):
        if len(env_ids) == 0:
            return

        self.root_states[env_ids] = self.base_init_state

        if self.custom_origins:
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, 0] += torch_rand_float(
                self.cfg.init_state.recovery_init_x_range[0],
                self.cfg.init_state.recovery_init_x_range[1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)
            self.root_states[env_ids, 1] += torch_rand_float(
                self.cfg.init_state.recovery_init_y_range[0],
                self.cfg.init_state.recovery_init_y_range[1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)
            self.root_states[env_ids, 2] = self.env_origins[env_ids, 2] + torch_rand_float(
                self.cfg.init_state.recovery_init_z_range[0],
                self.cfg.init_state.recovery_init_z_range[1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)
        else:
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, 2] = torch_rand_float(
                self.cfg.init_state.recovery_init_z_range[0],
                self.cfg.init_state.recovery_init_z_range[1],
                (len(env_ids), 1),
                device=self.device,
            ).squeeze(1)

        roll = torch_rand_float(
            self.cfg.init_state.recovery_roll_range[0],
            self.cfg.init_state.recovery_roll_range[1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

        pitch = torch_rand_float(
            self.cfg.init_state.recovery_pitch_range[0],
            self.cfg.init_state.recovery_pitch_range[1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

        yaw = torch_rand_float(
            self.cfg.init_state.recovery_yaw_range[0],
            self.cfg.init_state.recovery_yaw_range[1],
            (len(env_ids), 1),
            device=self.device,
        ).squeeze(1)

        self.root_states[env_ids, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)

        self.root_states[env_ids, 7:9] = torch_rand_float(
            self.cfg.init_state.recovery_init_lin_vel_xy_range[0],
            self.cfg.init_state.recovery_init_lin_vel_xy_range[1],
            (len(env_ids), 2),
            device=self.device,
        )

        self.root_states[env_ids, 9:10] = torch_rand_float(
            self.cfg.init_state.recovery_init_lin_vel_z_range[0],
            self.cfg.init_state.recovery_init_lin_vel_z_range[1],
            (len(env_ids), 1),
            device=self.device,
        )

        self.root_states[env_ids, 10:12] = torch_rand_float(
            self.cfg.init_state.recovery_init_ang_vel_xy_range[0],
            self.cfg.init_state.recovery_init_ang_vel_xy_range[1],
            (len(env_ids), 2),
            device=self.device,
        )

        self.root_states[env_ids, 12:13] = torch_rand_float(
            self.cfg.init_state.recovery_init_ang_vel_z_range[0],
            self.cfg.init_state.recovery_init_ang_vel_z_range[1],
            (len(env_ids), 1),
            device=self.device,
        )

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_states),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _reset_dofs(self, env_ids):
        if len(env_ids) == 0:
            return

        num = len(env_ids)
        env_ids_long = env_ids.long()

        # ------------------------------------------------------------
        # 1. Build q_default robustly.
        # self.default_dof_pos can be:
        #   [num_dof]
        #   [1, num_dof]
        #   [num_envs, num_dof]
        # Different branches in this repo use different shapes.
        # ------------------------------------------------------------
        default = self.default_dof_pos

        if default.dim() == 1:
            q_default = default.unsqueeze(0).expand(num, -1).clone()

        elif default.dim() == 2:
            if default.shape[0] == self.num_envs:
                q_default = default[env_ids_long].clone()
            elif default.shape[0] == 1:
                q_default = default.expand(num, -1).clone()
            else:
                # Fallback: treat the first row as the default posture.
                q_default = default[0:1].expand(num, -1).clone()
        else:
            raise RuntimeError(
                f"Unexpected default_dof_pos shape: {tuple(default.shape)}"
            )

        if q_default.shape[1] != self.num_dof:
            raise RuntimeError(
                f"q_default shape mismatch: {tuple(q_default.shape)}, num_dof={self.num_dof}"
            )

        # ------------------------------------------------------------
        # 2. Folded posture.
        # ------------------------------------------------------------
        folded_angles = torch.as_tensor(
            self.cfg.init_state.recovery_folded_joint_angles,
            dtype=torch.float,
            device=self.device,
        ).flatten()

        if folded_angles.numel() != self.num_dof:
            raise RuntimeError(
                f"recovery_folded_joint_angles length = {folded_angles.numel()}, "
                f"but num_dof = {self.num_dof}"
            )

        q_folded = folded_angles.unsqueeze(0).expand(num, -1).clone()

        folded_mask = (
            torch.rand(num, 1, device=self.device)
            < float(self.cfg.init_state.recovery_folded_prob)
        )

        q = torch.where(folded_mask, q_folded, q_default)

        # ------------------------------------------------------------
        # 3. Add joint noise.
        # ------------------------------------------------------------
        q_noise = torch_rand_float(
            -self.cfg.init_state.recovery_dof_noise,
            self.cfg.init_state.recovery_dof_noise,
            (num, self.num_dof),
            device=self.device,
        )

        q = q + q_noise

        # Foot / wheel joints start from zero to avoid initial wheel spinning.
        wheel_ids = torch.tensor([3, 7], dtype=torch.long, device=self.device)
        q[:, wheel_ids] = 0.0

        # ------------------------------------------------------------
        # 4. Clamp to joint position limits robustly.
        # self.dof_pos_limits is normally [num_dof, 2].
        # ------------------------------------------------------------
        limits = self.dof_pos_limits

        if limits.dim() == 2 and limits.shape[0] == self.num_dof:
            lower = limits[:, 0].unsqueeze(0)
            upper = limits[:, 1].unsqueeze(0)
        elif limits.dim() == 3 and limits.shape[0] == self.num_envs:
            lower = limits[env_ids_long, :, 0]
            upper = limits[env_ids_long, :, 1]
        else:
            raise RuntimeError(
                f"Unexpected dof_pos_limits shape: {tuple(limits.shape)}"
            )

        q = torch.max(torch.min(q, upper), lower)

        # ------------------------------------------------------------
        # 5. Write dof position / velocity.
        # Avoid self.dof_vel[env_ids, [3, 7]] because that advanced indexing
        # can broadcast incorrectly. Build local dof_vel first.
        # ------------------------------------------------------------
        self.dof_pos[env_ids_long] = q

        dof_vel = torch_rand_float(
            -1.0,
            1.0,
            (num, self.num_dof),
            device=self.device,
        )
        dof_vel[:, wheel_ids] = 0.0
        self.dof_vel[env_ids_long] = dof_vel

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(env_ids_int32),
            len(env_ids_int32),
        )

    def _get_base_height(self):
        return torch.mean(
            self.root_states[:, 2].unsqueeze(1) - self.measured_heights,
            dim=1,
        )

    def _reward_recovery_upright(self):
        # upright 时 projected_gravity[:, 2] 接近 -1。
        return torch.clamp(-self.projected_gravity[:, 2], 0.0, 1.0)

    def _reward_recovery_height(self):
        base_height = self._get_base_height()
        err = torch.square(base_height - self.cfg.rewards.base_height_target)
        return torch.exp(-err / self.cfg.rewards.recovery_height_sigma)

    def _reward_recovery_still(self):
        lin_xy = torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1)
        ang_xy = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
        err = lin_xy + 0.25 * ang_xy
        return torch.exp(-err / self.cfg.rewards.recovery_still_sigma)

    def _reward_recovery_success(self):
        base_height = self._get_base_height()

        upright_ok = -self.projected_gravity[:, 2] > 0.90
        height_ok = torch.abs(base_height - self.cfg.rewards.base_height_target) < 0.08
        lin_ok = torch.norm(self.base_lin_vel[:, :2], dim=1) < 0.30
        ang_ok = torch.norm(self.base_ang_vel[:, :2], dim=1) < 0.80

        return (upright_ok & height_ok & lin_ok & ang_ok).float()

    def _randomize_recovery_terrain_origins(self, env_ids):
        """Randomize terrain level at every reset without distance-based curriculum.

        Recovery 任务不使用“走出去多远”作为地形升级标准。
        每次 reset 随机 terrain level，但保留 terrain type 分布。
        """
        if len(env_ids) == 0:
            return

        if not getattr(self, "custom_origins", False):
            return

        if not hasattr(self, "terrain_origins"):
            return

        env_ids_long = env_ids.long()

        self.terrain_levels[env_ids_long] = torch.randint(
            low=0,
            high=self.max_terrain_level,
            size=(len(env_ids_long),),
            device=self.device,
        )

        self.env_origins[env_ids_long] = self.terrain_origins[
            self.terrain_levels[env_ids_long],
            self.terrain_types[env_ids_long],
        ]

    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return

        recovery_success = self._reward_recovery_success()[env_ids].detach()

        # Recovery 不使用原来的距离型 terrain curriculum。
        # 在 super().reset_idx(env_ids) 之前更新 env_origins，
        # 因为 super().reset_idx 会调用 _reset_root_states。
        self._randomize_recovery_terrain_origins(env_ids)

        super().reset_idx(env_ids)

        if "episode" in self.extras:
            self.extras["episode"]["recovery_success_rate"] = recovery_success.mean()

            if getattr(self, "custom_origins", False):
                self.extras["episode"]["terrain_level_random"] = torch.mean(
                    self.terrain_levels.float()
                )
