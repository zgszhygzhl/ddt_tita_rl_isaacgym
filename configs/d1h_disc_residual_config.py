from configs.y1v0h_evt1_climb_config import (
    Y1v0hEvt1Climb,
    Y1v0hEvt1ClimbCfg,
    Y1v0hEvt1ClimbCfgPPO,
)


class D1hDiscResidualCfg(Y1v0hEvt1ClimbCfg):
    """
    Discrete-terrain residual expert config.

    这个任务不是重新训练完整上台阶策略，而是在 frozen d1h_base 上训练 residual 修正：
        a_final = a_base + alpha * delta_disc

    目标：
    1. 保留 d1h_evt1_climb 已经验证成功的台阶地形和奖励结构；
    2. 减小网络和训练轮次；
    3. 增大跳跃/弹跳惩罚，避免纯跳台阶；
    4. 让 expert 更像“离散地形补偿器”，而不是完整策略。
    """

    class env(Y1v0hEvt1ClimbCfg.env):
        num_envs = 4096

    class commands(Y1v0hEvt1ClimbCfg.commands):
        # residual disc expert 主要学前向上台阶，不重点训练横移/大转向。
          # 1.x 2.y 3.xy_mix 4.spot_turn 5.x_rotation 6.y_rotation 7.xy_mix_rotation 8.stand_still
        commands_proportion = [0.75, 0.05, 0.05, 0.05, 0.025, 0.025, 0.025, 0.0, 0.025]

        curriculum = True
        max_curriculum = 0.8

        max_curriculum_x = 0.8
        max_curriculum_y = 0.15
        min_curriculum_x = -0.4
        min_curriculum_y = -0.15
        max_curriculum_z = 0.25

        # 命令不要太激进，先让 residual 学稳定越障。
        max_lin_vel_x_change_rate = 0.5
        max_lin_vel_y_change_rate = 0.2
        max_ang_vel_change_rate = 0.3

        enable_command_buffer = True
        buffer_smoothing_factor = 0.1

        class ranges(Y1v0hEvt1ClimbCfg.commands.ranges):
            # 先聚焦 0.2~0.55 m/s 上台阶。
            # 后面如果稳定，再把上限推到 0.7 或 1.0。
            lin_vel_x = [0.20, 0.55]
            lin_vel_y = [-0.08, 0.08]
            ang_vel_yaw = [-0.10, 0.10]
            heading = [-0.5, 0.5]

    class rewards(Y1v0hEvt1ClimbCfg.rewards):
        # 保持 climb 的较高 base 高度目标，方便过台阶。
        base_height_target = 0.45

        # 双脚同时离地判定阈值和容忍时间。
        # grace_time 太小会惩罚接触噪声；太大又会放过真正跳跃。
        both_feet_air_contact_force = 5.0
        both_feet_air_grace_time = 0.015
        both_feet_air_ramp_time = 0.025

        # 更严格抑制向上弹跳。
        upward_vel_spike_deadband = 0.10
        upward_vel_spike_max_excess = 1.0

        contact_upward_bounce_force = 5.0
        contact_upward_bounce_deadband = 0.08
        contact_upward_bounce_max_excess = 1.0

        class scales(Y1v0hEvt1ClimbCfg.rewards.scales):
            termination = -100.0

            # 保持前向速度跟踪，但不要为了速度牺牲姿态。
            tracking_lin_vel = 0.0
            tracking_lin_vel_x = 12.0
            tracking_lin_vel_y = 2.0
            tracking_ang_vel = 2.0

            # 垂向速度和姿态稳定更重要。
            lin_vel_z = -3.0
            ang_vel_xy = -0.08
            orientation = -12.0
            base_height = -28.0

            # 动作平滑略加重，避免 residual 大幅抖动。
            powers = -2e-5
            dof_acc = -3.0e-7
            action_rate = -0.15
            action_smoothness = 0.0

            collision = -10.0
            stand_still = -1.0

            # 不使用 feet_air_time；当前实现对纯前向命令不适合直接打开。
            feet_air_time = 0.0
            foot_clearance = 0.0
            stumble = 0.0

            # 不要把 no_gait 加太大，否则会鼓励双脚一直接触，反而不利于形成类步态。
            no_gait = 0.5

            # 关键：打开双脚同时离地惩罚，抑制纯跳。
            both_feet_air = -15.0

            # 关键：加重向上弹跳惩罚。
            upward_vel_spike = -35.0
            contact_upward_bounce = -30.0

            # 支撑几何约束，防止一条腿伸太远或身体跑出支撑线。
            body_pos_to_feet_x = 1.0
            body_feet_distance_x = -20.0
            body_feet_distance_y = -100.0

            # z 对称不要太大，否则会压制一腿上台阶、一腿跟随的高度差。
            body_symmetry_y = 0.2
            body_symmetry_z = 0.03

            heading = 6.0
            upward = 1.0
            head_los_distance = -25.0

    class domain_rand(Y1v0hEvt1ClimbCfg.domain_rand):
        # residual 第一版不要太强随机化，先学会稳定上台阶。
        randomize_friction = True
        friction_range = [0.6, 1.8]

        randomize_restitution = True
        restitution_range = [0.0, 0.3]

        randomize_base_mass = True
        added_mass_range = [-0.5, 1.0]

        randomize_base_com = True
        added_com_range = [-0.03, 0.03]

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

    class terrain(Y1v0hEvt1ClimbCfg.terrain):
        mesh_type = 'trimesh'
        measure_heights = True

        # utils/terrain.py 中实际顺序：
        # [slope, rough_slope, stairs_down branch, stairs_up branch, discrete, stones, gap, pit, ...]
        # 这里主要训练 stairs_up residual。
        terrain_proportions = [0.05, 0.0, 0.85, 0.10, 0.0, 0.0, 0.0, 0.0, 0.0]

        slope = [0.0, 0.02]

        # 第一版目标到 15 cm，略低于原 climb 的 16 cm，降低 residual 学习难度。
        # 稳定后可以改到 [0.02, 0.16] 或 [0.03, 0.17]。
        step_height = [0.02, 0.15]
        step_width_range = [0.52, 0.62]

        discrete_obstacles_height = [0.05, 0.15]
        pit_depth = [0.0, 0.3]

        slope_treshold = 0.2
        max_init_terrain_level = 4


class D1hDiscResidualCfgPPO(Y1v0hEvt1ClimbCfgPPO):
    class algorithm(Y1v0hEvt1ClimbCfgPPO.algorithm):
        entropy_coef = 0.006
        learning_rate = 8.0e-4
        max_grad_norm = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        cost_value_loss_coef = 0.1
        cost_viol_loss_coef = 0.1

        # residual 修正量 L2 正则。
        # 作用：鼓励 expert 少改 base，只在台阶/冲击等必要时修正动作。
        # 过大可能导致上不去；过小则 expert 可能重新变成大幅跳跃策略。
        residual_l2_coef = 0.005

    class policy(Y1v0hEvt1ClimbCfgPPO.policy):
        # residual expert 不需要原 climb 的大网络。
        init_noise_std = 0.6
        continue_from_last_std = True

        scan_encoder_dims = [64, 32]
        actor_hidden_dims = [256, 128, 64]
        critic_hidden_dims = [256, 128, 64]
        priv_encoder_dims = []

        activation = 'elu'

        rnn_type = 'lstm'
        rnn_hidden_size = 256
        rnn_num_layers = 1

        tanh_encoder_output = False
        num_costs = 6

        teacher_act = True

        # residual wrapper 里已经关闭 imitation。
        # 这里也关掉，避免专家训练混入 Barlow imitation loss。
        imi_flag = False

    class runner(Y1v0hEvt1ClimbCfgPPO.runner):
        run_name = 'd1h_disc_residual'
        experiment_name = 'd1h_disc_residual'
        policy_class_name = 'ActorCriticBarlowTwins'
        runner_class_name = 'OnConstraintPolicyRunner'
        algorithm_class_name = 'NP3O'

        # residual expert 在 frozen base 上训练，不需要原 climb 的 40000 轮。
        max_iterations = 8000
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


class D1hDiscResidual(Y1v0hEvt1Climb):
    """
    先复用 Y1v0hEvt1Climb 的 reward/cost 函数。
    后续如果需要专门统计台阶通过率、跳跃率、双脚离地时间，
    再在这个类里补充新的 reward 或 info logging。
    """
    pass