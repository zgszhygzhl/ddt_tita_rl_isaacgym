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

class Y1v0hEvt1ClimbCfg( LeggedRobotCfg ):
    class env(LeggedRobotCfg.env):
        num_envs = 4096

        n_scan = 187
        n_priv_latent =  4 + 1 + 8 + 8 + 8 + 6 + 1 + 2 + 1 - 3
        n_proprio = 33
        history_len = 10
        num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent

    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.5] # x,y,z [m]
        rot = [0, 0.0, 0.0, 1]  # x, y, z, w [quat]
        lin_vel = [0.0, 0.0, 0.0]  # x, y, z [m/s]
        ang_vel = [0.0, 0.0, 0.0]  # x, y, z [rad/s]  
        random_ori_probability = 0.0
        random_dof_pos_probability = 0.0
        default_joint_angles = {
                'FL_hip_joint': 0,
                'FR_hip_joint': 0,

                'FL_thigh_joint': 0.8,
                'FR_thigh_joint': 0.8,

                'FL_calf_joint': -1.5,
                'FR_calf_joint': -1.5,

                'FL_foot_joint': 0,
                'FR_foot_joint': 0,
        }
        desired_feet_distance = 0.38
        # 双足期望间距随机化，增强爬楼站姿鲁棒性。
        feet_distance_range = [0.32, 0.50]  # [min, max] feet distance range for randomization


    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 40}  # [N*m/rad]
        damping = {'joint': 1.0}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        hip_scale_reduction = 0.5

        use_filter = True

    class commands( LeggedRobotCfg.control ):
        # 命令类型比例：前后、横移、混合、转向和站立。
        # 下面三行保留历史比例，方便回退调参。
        # commands_proportion = [0.3, 0.2, 0.2, 0.15, 0.05, 0.025, 0.025, 0.05]
        # commands_proportion = [0.2, 0.2, 0.1, 0.2, 0.025, 0.025, 0.15, 0.1]
        # commands_proportion = [0.1, 0.1, 0.1, 0.1, 0.175, 0.175, 0.15, 0.1]
        commands_proportion = [0.45,0.1,0.1,0.1,0.05,0.05,0.05,0.05,0.05]
        curriculum = True
        max_curriculum = 1.0
        
        max_curriculum_x = 1.0
        max_curriculum_y = 1.0
        min_curriculum_x = -1.0
        min_curriculum_y = -1.0
        max_curriculum_z = 1.0

        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10  # time before command are changed[s]
        # resample_probability = 0.1
        heading_command = False  # if true: compute ang vel command from heading error
        global_reference = False
        
        # 同向命令按概率取反，增加命令覆盖范围。
        flip_same_sign_probability = 0.2
        
        # 限制命令变化率，避免目标速度突变。
        max_lin_vel_x_change_rate = 0.5  # x方向线速度最大变化率 [m/s^2]
        max_lin_vel_y_change_rate = 0.3  # y方向线速度最大变化率 [m/s^2]
        max_ang_vel_change_rate = 0.5    # yaw角速度最大变化率 [rad/s^2]
        
        # 命令缓冲和平滑，让跟踪命令逐步逼近采样命令。
        enable_command_buffer = True
        buffer_smoothing_factor = 0.1    # 越小越平滑，响应也越慢。
        
        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]  # min max [rad/s]
            heading = [-3.14, 3.14]

    class asset( LeggedRobotCfg.asset ):

        file = '{ROOT_DIR}/resources/d1h/urdf/robot.urdf'
        foot_name = "foot"
        root_name = "base"
        name = "d1h"
        penalize_contacts_on = ["calf","thigh"]
        # terminate_after_contacts_on = [] # for obstacle cross
        terminate_after_contacts_on = ["base"]
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
    
    # True for new structure by lkx, False for old structure
    class sim2sim:
        s2s_struct = True 
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.5
        # acc_smoothness_sigma = 0.5
        class scales( LeggedRobotCfg.rewards.scales ):

            torques = 0.0
            powers = -2e-5
            termination = -100.0
            tracking_lin_vel = 0.0
            tracking_lin_vel_x = 15.0
            tracking_lin_vel_y = 10.0
            tracking_ang_vel = 5.0
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            dof_vel = 0.0
            dof_acc = -2.5e-7
            base_height = -20.0
            feet_air_time = 0.0
            collision = -10.0
            stumble = 0.0
            action_rate = -0.1
            action_smoothness= 0
            stand_still = -1
            foot_clearance= -0.0
            orientation=-10.0
            no_gait = 5.0
            
            # 爬楼约束：鼓励身体保持在双足之间。
            body_pos_to_feet_x = 1.0
            # body_vel_to_feet_x = 10.0
            body_feet_distance_x = -50.0
            body_feet_distance_y = -100.0
            body_symmetry_y = 0.1
            body_symmetry_z = 0.3
            # body_symmetry = 10
            body_symmetry_y = 0.3
            body_symmetry_z = 0.9
            heading = 10.0
            upward = 1.0

    class domain_rand( LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.2, 2.75]
        randomize_restitution = True
        restitution_range = [0.0,1.0]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        randomize_base_com = True
        added_com_range = [-0.1, 0.1]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1

        randomize_motor = True
        motor_strength_range = [0.8, 1.2]

        randomize_kpkd = True
        kp_range = [0.8,1.2]
        kd_range = [0.8,1.2]

        randomize_lag_timesteps = True
        lag_timesteps = 3

        disturbance = False
        disturbance_range = [-30.0, 30.0]
        disturbance_interval = 8
    
    class depth( LeggedRobotCfg.depth):
        use_camera = False
        camera_num_envs = 192
        camera_terrain_num_rows = 10
        camera_terrain_num_cols = 20

        position = [0.27, 0, 0.03]  # front camera
        angle = [-5, 5]  # positive pitch down

        update_interval = 1  # 5 works without retraining, 8 worse

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
            # vel_smoothness = 0.1
            acc_smoothness = 0.1
            #collision = 0.1
            feet_contact_forces = 0.8
            stumble = 0.1
        class d_values:
            pos_limit = 0.0
            torque_limit = 0.0
            dof_vel_limits = 0.0
            # vel_smoothness = 0.0
            acc_smoothness = 0.0
            #collision = 0.0
            feet_contact_forces = 0.0
            stumble = 0.0
    
    class cost:
        num_costs = 6
    
    class terrain(LeggedRobotCfg.terrain):
        # mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        static_friction = 1.0
        dynamic_friction = 1.0
        mesh_type = 'trimesh'
        measure_heights = True
        include_act_obs_pair_buf = False
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete, stepping stones, gap, obstacles crossing, high platform]
        # terrain_proportions = [0.0, 0.0, 0.7, 0.3, 0.0, 0.0, 0.0]
        terrain_proportions = [0.1, 0.0, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
        stairs_max_height = 0.15

class Y1v0hEvt1ClimbCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        learning_rate = 1.e-3
        max_grad_norm = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        cost_value_loss_coef = 0.1
        cost_viol_loss_coef = 0.1

    class policy( LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        continue_from_last_std = True
        scan_encoder_dims = [128, 64, 32]
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        #priv_encoder_dims = [64, 20]
        priv_encoder_dims = []
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 512
        rnn_num_layers = 1

        tanh_encoder_output = False
        num_costs = 6

        teacher_act = True
        imi_flag = True
      
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = 'd1h_evt1_climb'
        experiment_name = 'd1h_evt1_climb'
        policy_class_name = 'ActorCriticBarlowTwins'
        runner_class_name = 'OnConstraintPolicyRunner'
        algorithm_class_name = 'NP3O'
        max_iterations = 40000
        num_steps_per_env = 24
        resume = False
        resume_path = ''
 
class Y1v0hEvt1Climb(Y1v0hEvt1Command):
    def _reward_tracking_lin_vel_x(self):
        # 跟踪平滑后的x方向线速度命令。
        lin_vel_x_error = torch.square(self.commands_given[:, 0] - self.base_lin_vel[:, 0])
        reward = torch.exp(-lin_vel_x_error/self.cfg.rewards.tracking_sigma)
        return reward

    def _reward_tracking_lin_vel_y(self):
        # 跟踪平滑后的y方向线速度命令。
        lin_vel_y_error = torch.square(self.commands_given[:, 1] - self.base_lin_vel[:, 1])
        return torch.exp(-lin_vel_y_error/self.cfg.rewards.tracking_sigma)

    def _cost_dof_vel_limits(self):
        # 只检查腿部主要关节速度限制，排除足端关节3和7。
        leg_joint_indices = [0, 1, 2, 4, 5, 6]
        return 1.*(torch.sum(1.*(torch.abs(self.dof_vel[:, leg_joint_indices]) > self.dof_vel_limits[leg_joint_indices]*self.cfg.rewards.soft_dof_vel_limit),dim=1) > 0.0)


