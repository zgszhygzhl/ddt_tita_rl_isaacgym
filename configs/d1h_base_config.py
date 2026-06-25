# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from configs.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
import torch
from configs.y1v0h_evt1_command import Y1v0hEvt1Command
from utils.math import wrap_to_pi
import numpy as np

class D1hBaseCfg(LeggedRobotCfg):
    """
    Base / nominal rolling policy config.

    This task is intentionally NOT a stair-climbing task.
    It is used to train a stable default controller for flat ground,
    mild slopes, mild rough terrain, normal velocity tracking and basic standing.
    Later discrete-terrain and recovery residual experts should be trained on top of this base.
    """

    class env(LeggedRobotCfg.env):
        num_envs = 4096

        n_scan = 187
        n_priv_latent = 4 + 1 + 8 + 8 + 8 + 6 + 1 + 2 + 1 - 3
        n_proprio = 33
        history_len = 10
        num_observations = n_proprio + n_scan + history_len * n_proprio + n_priv_latent

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.46]  # x, y, z [m]
        rot = [0, 0.0, 0.0, 1]  # x, y, z, w [quat]
        lin_vel = [0.0, 0.0, 0.0]
        ang_vel = [0.0, 0.0, 0.0]

        random_ori_probability = 0.0
        random_dof_pos_probability = 0.0

        default_joint_angles = {
            'FL_hip_joint': 0.0,
            'FR_hip_joint': 0.0,

            'FL_thigh_joint': 0.8,
            'FR_thigh_joint': 0.8,

            'FL_calf_joint': -1.5,
            'FR_calf_joint': -1.5,

            'FL_foot_joint': 0.0,
            'FR_foot_joint': 0.0,
        }

        desired_feet_distance = 0.44
        feet_distance_range = [0.36, 0.50]

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {'joint': 40}
        damping = {'joint': 1.0}
        action_scale = 0.5
        decimation = 4
        hip_scale_reduction = 0.5
        use_filter = True

    class commands(LeggedRobotCfg.commands):
        # Base policy should learn normal rolling and command tracking.
        # It should not be specialized for stair crawling.
        # 1.x 2.y 3.xy_mix 4.spot_turn 5.x_rotation 6.y_rotation 7.xy_mix_rotation 8.stand_still
        commands_proportion = [0.55, 0.05, 0.10, 0.10, 0.05, 0.05, 0.05, 0.00, 0.05]

        curriculum = True
        max_curriculum = 0.8

        max_curriculum_x = 1.0
        max_curriculum_y = 0.5
        min_curriculum_x = -1.0
        min_curriculum_y = -0.5
        max_curriculum_z = 1.0

        num_commands = 4
        resampling_time = 10
        heading_command = False
        global_reference = False

        flip_same_sign_probability = 0.2

        max_lin_vel_x_change_rate = 0.6
        max_lin_vel_y_change_rate = 0.2
        max_ang_vel_change_rate = 0.5

        enable_command_buffer = True
        buffer_smoothing_factor = 0.1

        class ranges:
            lin_vel_x = [-0.3, 0.8]
            lin_vel_y = [-0.1, 0.1]
            ang_vel_yaw = [-0.5, 0.5]
            heading = [-3.14, 3.14]

    class asset(LeggedRobotCfg.asset):
        file = '{ROOT_DIR}/resources/d1h/urdf/robot.urdf'
        foot_name = "foot"
        root_name = "base"
        name = "d1h"
        penalize_contacts_on = ["calf", "thigh"]
        terminate_after_contacts_on = ["base"]
        self_collisions = 0
        flip_visual_attachments = False

    class sim2sim:
        s2s_struct = True

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.5

        # Keep these attributes for compatibility with reward functions,
        # but base task does not use the stair-specific terms.
        head_los_forward_offset = 0.0
        head_los_deadband = 0.05
        head_los_max_distance = 0.35

        both_feet_air_contact_force = 5.0
        both_feet_air_grace_time = 0.04
        both_feet_air_ramp_time = 0.08

        upward_vel_spike_deadband = 0.25
        upward_vel_spike_max_excess = 1.0
        contact_upward_bounce_force = 5.0
        contact_upward_bounce_deadband = 0.15
        contact_upward_bounce_max_excess = 1.0

        class scales(LeggedRobotCfg.rewards.scales):
            torques = 0.0
            powers = -2e-5
            termination = -100.0

            # Main base-policy tracking rewards.
            tracking_lin_vel = 0.0
            tracking_lin_vel_x = 15.0
            tracking_lin_vel_y = 5.0
            tracking_ang_vel = 5.0

            # Stability and smoothness.
            lin_vel_z = -2.0
            ang_vel_xy = -0.08
            orientation = -10.0
            base_height = -15.0

            dof_vel = 0.0
            dof_acc = -2.5e-7
            action_rate = -0.1
            action_smoothness = 0.0

            collision = -10.0
            stand_still = -1.5

            feet_air_time = 0.0
            foot_clearance = 0.0
            stumble = 0.0

            # Base task is not trained on stairs, so these stair-jump suppressors stay off.
            no_gait = 5.0
            both_feet_air = 0.0
            upward_vel_spike = 0.0
            contact_upward_bounce = 0.0

            # Generic support-geometry regularization.
            body_pos_to_feet_x = 1.0
            body_feet_distance_x = -10.0
            body_feet_distance_y = -50.0
            body_symmetry_y = 0.2
            body_symmetry_z = 0.3

            heading = 5.0
            upward = 1.0
            head_los_distance = -20.0

    class domain_rand(LeggedRobotCfg.domain_rand):
        # Mild randomization for base. Strong disturbances are left for recovery expert.
        randomize_friction = True
        friction_range = [0.6, 1.8]

        randomize_restitution = False
        restitution_range = [0.0, 0.2]

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

    class depth(LeggedRobotCfg.depth):
        use_camera = False
        camera_num_envs = 192
        camera_terrain_num_rows = 10
        camera_terrain_num_cols = 20

        position = [0.27, 0, 0.03]
        angle = [-5, 5]

        update_interval = 1

        original = (106, 60)
        resized = (87, 58)
        horizontal_fov = 87
        buffer_len = 2

        near_clip = 0
        far_clip = 2
        dis_noise = 0.0

        scale = 1
        invert = True

    class costs:
        class scales:
            pos_limit = 0.3
            torque_limit = 0.3
            dof_vel_limits = 0.3
            acc_smoothness = 0.1
            feet_contact_forces = 0.8
            stumble = 0.1

        class d_values:
            pos_limit = 0.0
            torque_limit = 0.0
            dof_vel_limits = 0.0
            acc_smoothness = 0.0
            feet_contact_forces = 0.0
            stumble = 0.0

    class cost:
        num_costs = 6

    class terrain(LeggedRobotCfg.terrain):
        static_friction = 1.0
        dynamic_friction = 1.0
        mesh_type = 'trimesh'
        measure_heights = True
        include_act_obs_pair_buf = False

        # Terrain types in utils/terrain.py:
        # [smooth slope, rough slope, stairs down, stairs up, discrete, stepping stones, gap, pit, ...]
        # Base should not learn stair climbing. Keep stairs/discrete/gap/pit at 0.
        terrain_proportions = [0.8, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        slope = [0.0, 0.05]
        step_height = [0.02, 0.04]  # unused for base because stair proportions are zero
        step_width_range = [0.55, 0.65]
        discrete_obstacles_height = [0.02, 0.06]
        pit_depth = [0.0, 0.2]

        slope_treshold = 0.75
        max_init_terrain_level = 4


class D1hBaseCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        learning_rate = 1.e-3
        max_grad_norm = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        cost_value_loss_coef = 0.1
        cost_viol_loss_coef = 0.1

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        continue_from_last_std = True

        # Smaller base network for fair residual-MoE comparison.
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
        imi_flag = True

    class runner(LeggedRobotCfgPPO.runner):
        run_name = 'd1h_base'
        experiment_name = 'd1h_base'
        policy_class_name = 'ActorCriticBarlowTwins'
        runner_class_name = 'OnConstraintPolicyRunner'
        algorithm_class_name = 'NP3O'
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


class D1hBase(Y1v0hEvt1Command):
    def _reward_tracking_lin_vel_x(self):
        # Track smoothed x velocity command.
        lin_vel_x_error = torch.square(self.commands_given[:, 0] - self.base_lin_vel[:, 0])
        reward = torch.exp(-lin_vel_x_error / self.cfg.rewards.tracking_sigma)
        return reward

    def _reward_tracking_lin_vel_y(self):
        # Track smoothed y velocity command.
        lin_vel_y_error = torch.square(self.commands_given[:, 1] - self.base_lin_vel[:, 1])
        return torch.exp(-lin_vel_y_error / self.cfg.rewards.tracking_sigma)

    def _cost_dof_vel_limits(self):
        # Check only main leg joints, excluding foot/wheel joints 3 and 7.
        leg_joint_indices = [0, 1, 2, 4, 5, 6]
        return 1.0 * (
            torch.sum(
                1.0 * (
                    torch.abs(self.dof_vel[:, leg_joint_indices])
                    > self.dof_vel_limits[leg_joint_indices] * self.cfg.rewards.soft_dof_vel_limit
                ),
                dim=1,
            )
            > 0.0
        )
