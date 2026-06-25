import time
import os
import re
from collections import deque
import statistics
import warnings

import numpy as np
from isaacgym import gymapi
from PIL import Image, ImageDraw, ImageFont

from torch.utils.tensorboard import SummaryWriter
import torch
from global_config import ROOT_DIR

from modules import ActorCriticRMA,ActorCriticBarlowTwins
from algorithm import NP3O
from envs.vec_env import VecEnv
from modules.depth_backbone import DepthOnlyFCBackbone58x87, RecurrentDepthBackbone
from utils.helpers import hard_phase_schedualer, partial_checkpoint_load, get_load_path
from utils.video_recorder import FfmpegVideoWriter
from copy import copy, deepcopy


class OnConstraintPolicyRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu'):

        self.cfg = train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.depth_encoder_cfg = train_cfg["depth_encoder"]
        self.device = device
        self.env = env

        # self.phase1_end = self.cfg["phase1_end"] 
 
        actor_critic_class = eval(self.cfg["policy_class_name"])  # ActorCritic
        actor_critic: ActorCriticRMA = actor_critic_class(self.env.cfg.env.n_proprio,
                                                      self.env.cfg.env.n_scan,
                                                      self.env.num_obs,
                                                      self.env.cfg.env.n_priv_latent,
                                                      self.env.cfg.env.history_len,
                                                      self.env.num_actions,
                                                      **self.policy_cfg)
        # Full checkpoint loading is handled after the algorithm/optimizer is created.
        # Loading here would only restore actor_critic weights and would miss optimizer
        # state plus learning iteration.
        actor_critic.to(self.device)
        

        # Depth encoder
        self.if_depth = self.depth_encoder_cfg["if_depth"]
        if self.if_depth:
            depth_backbone = DepthOnlyFCBackbone58x87(env.cfg.env.n_proprio, 
                                                    self.policy_cfg["scan_encoder_dims"][-1], 
                                                    self.depth_encoder_cfg["hidden_dims"],
                                                    )
            depth_encoder = RecurrentDepthBackbone(depth_backbone, env.cfg).to(self.device)
            depth_actor = deepcopy(actor_critic.actor)
        else:
            depth_encoder = None
            depth_actor = None

        # Create algorithm
        self.alg_cfg['k_value'] = self.env.cost_k_values
        alg_class = eval(self.cfg["algorithm_class_name"]) # PPO
        self.alg = alg_class(actor_critic, 
                                  depth_encoder, self.depth_encoder_cfg, depth_actor,
                                  device=self.device,
                                  **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        self.dagger_update_freq = self.alg_cfg["dagger_update_freq"]

        self.alg.init_storage(
            self.env.num_envs, 
            self.num_steps_per_env, 
            [self.env.num_obs], 
            [self.env.num_privileged_obs], 
            [self.env.num_actions],
            [self.env.cfg.cost.num_costs],
            self.env.cost_d_values_tensor
        )
        # Log
        self.log_dir = log_dir
        self.checkpoint_dir = None
        if self.log_dir is not None:
            self.checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        if self.cfg.get('resume', False):
            self._load_training_checkpoint()

        # Training video recording. A single mp4 contains a mosaic of several
        # randomly selected environments. Cameras are created once to avoid
        # accumulating camera sensors during long training runs.
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


    def _checkpoint_path(self, iteration):
        save_dir = self.checkpoint_dir if self.checkpoint_dir is not None else self.log_dir
        if save_dir is None:
            raise RuntimeError("Cannot save checkpoint because log_dir is None")
        os.makedirs(save_dir, exist_ok=True)
        return os.path.join(save_dir, f"model_{int(iteration)}.pt")

    def _parse_iteration_from_checkpoint_path(self, path):
        match = re.search(r"model_(\d+)\.pt$", os.path.basename(str(path)))
        return int(match.group(1)) if match is not None else 0

    def _resolve_resume_path(self):
        resume_path = self.cfg.get("resume_path", "")
        if resume_path not in [None, "", -1, "-1"]:
            resume_path = str(resume_path)
            if os.path.isabs(resume_path):
                return resume_path
            return os.path.join(ROOT_DIR, resume_path)

        experiment_name = self.cfg.get("experiment_name", None)
        if experiment_name is None:
            raise ValueError("resume=True but neither resume_path nor experiment_name is set")

        log_root = os.path.join(ROOT_DIR, "logs", experiment_name)
        load_run = self.cfg.get("load_run", -1)
        checkpoint = self.cfg.get("checkpoint", -1)
        return get_load_path(log_root, load_run=load_run, checkpoint=checkpoint)

    def _load_training_checkpoint(self):
        resume_path = self._resolve_resume_path()
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")

        print(f"[resume] loading checkpoint: {resume_path}")
        loaded_dict = torch.load(resume_path, map_location=self.device)

        if "model_state_dict" not in loaded_dict:
            raise KeyError(f"Checkpoint has no model_state_dict: {resume_path}")
        self.alg.actor_critic.load_state_dict(loaded_dict["model_state_dict"])

        if "optimizer_state_dict" in loaded_dict:
            try:
                self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
                print("[resume] optimizer_state_dict loaded")
            except Exception as error:
                warnings.warn(f"Failed to load optimizer_state_dict: {error}")
        else:
            warnings.warn("Checkpoint has no optimizer_state_dict; optimizer will restart")

        checkpoint_iter = loaded_dict.get("iter", None)
        if checkpoint_iter is None or int(checkpoint_iter) <= 0:
            checkpoint_iter = self._parse_iteration_from_checkpoint_path(resume_path)
        self.current_learning_iteration = int(checkpoint_iter)
        print(f"[resume] current_learning_iteration set to {self.current_learning_iteration}")

    def _setup_train_video_camera(self):
        """Create fixed world cameras for random training environments."""
        self.video_env_ids = []
        self.video_cam_handles = []

        max_cameras = min(
            self.env.num_envs,
            self.video_num_envs,
            self.video_tile_rows * self.video_tile_cols,
        )
        if max_cameras <= 0:
            return

        camera_props = gymapi.CameraProperties()
        camera_props.width = self.video_tile_width
        camera_props.height = self.video_tile_height

        # Randomly choose a diverse subset once at runner creation. We do not
        # recreate cameras every video_interval, because Isaac Gym does not need
        # camera churn during long training runs.
        if self.env.num_envs <= max_cameras:
            selected_env_ids = list(range(self.env.num_envs))
        else:
            selected_env_ids = torch.randperm(
                self.env.num_envs,
                device=self.env.device,
            )[:max_cameras].detach().cpu().tolist()
        selected_env_ids = [int(env_id) for env_id in selected_env_ids]

        self.video_env_ids = selected_env_ids
        print(f"[video] selected training envs: {self.video_env_ids}")

        for env_id in self.video_env_ids:
            env_handle = self.env.envs[env_id]
            cam_handle = self.env.gym.create_camera_sensor(env_handle, camera_props)
            self.video_cam_handles.append(cam_handle)

        self.video_record_every = max(1, int(round(1.0 / (self.video_fps * self.env.dt))))
        self._update_train_video_camera_locations()

    def _update_train_video_camera_locations(self):
        """Keep cameras looking at each selected environment origin."""
        for env_id, cam_handle in zip(self.video_env_ids, self.video_cam_handles):
            origin = self.env.env_origins[env_id].detach().cpu().numpy()
            env_handle = self.env.envs[env_id]

            cam_pos = gymapi.Vec3(
                float(origin[0] + 2.4),
                float(origin[1] - 3.3),
                float(origin[2] + 1.55),
            )
            cam_target = gymapi.Vec3(
                float(origin[0] + 0.35),
                float(origin[1] + 0.0),
                float(origin[2] + 0.65),
            )
            self.env.gym.set_camera_location(cam_handle, env_handle, cam_pos, cam_target)

    def _start_train_video(self, iteration):
        if not self.record_video or not self.video_cam_handles:
            return

        self._close_train_video()
        self.video_current_iteration = int(iteration)
        video_dir = os.path.join(self.log_dir, "videos")
        video_path = os.path.join(video_dir, f"train_iter_{iteration:06d}.mp4")
        self.video_writer = FfmpegVideoWriter(
            video_path,
            self.video_width,
            self.video_height,
            self.video_fps,
        )
        self.video_steps_left = max(1, int(np.ceil(self.video_duration / self.env.dt)))
        self.video_step_count = 0
        print(f"[video] recording training video: {video_path}")

    def _get_train_video_terrain_text(self, env_id):
        """Build compact terrain information text for one recorded training env."""
        if not hasattr(self.env, "terrain_levels"):
            return "ter=plane"

        try:
            level = int(self.env.terrain_levels[env_id].item())
        except Exception:
            level = -1

        num_rows = int(getattr(self.env.cfg.terrain, "num_rows", 1))
        num_cols = int(getattr(self.env.cfg.terrain, "num_cols", 1))
        max_level = max(num_rows - 1, 0)

        # utils/terrain.py uses: difficulty = row / num_rows
        difficulty = float(level) / max(float(num_rows), 1.0) if level >= 0 else 0.0

        terrain_type = -1
        if hasattr(self.env, "terrain_types"):
            try:
                terrain_type = int(self.env.terrain_types[env_id].item())
            except Exception:
                terrain_type = -1

        terrain_kind = "unknown"
        if hasattr(self.env.cfg.terrain, "terrain_proportions") and terrain_type >= 0:
            proportions = np.cumsum(
                np.asarray(self.env.cfg.terrain.terrain_proportions, dtype=np.float32)
            )
            choice = terrain_type / max(float(num_cols), 1.0) + 0.001

            if len(proportions) > 0 and choice < proportions[0]:
                terrain_kind = "slope"
            elif len(proportions) > 1 and choice < proportions[1]:
                terrain_kind = "rough_slope"
            elif len(proportions) > 3 and choice < proportions[3]:
                # In utils/terrain.py: if choice < proportions[2], step_height *= -1.
                if len(proportions) > 2 and choice < proportions[2]:
                    terrain_kind = "stairs_down"
                else:
                    terrain_kind = "stairs_up"
            elif len(proportions) > 4 and choice < proportions[4]:
                terrain_kind = "obstacles"
            elif len(proportions) > 5 and choice < proportions[5]:
                terrain_kind = "stones"
            elif len(proportions) > 6 and choice < proportions[6]:
                terrain_kind = "gap"
            else:
                terrain_kind = "pit"

        height_text = ""
        if hasattr(self.env.cfg.terrain, "step_height") and "stairs" in terrain_kind:
            step_min = float(self.env.cfg.terrain.step_height[0])
            step_max = float(self.env.cfg.terrain.step_height[1])
            step_h = step_min + (step_max - step_min) * difficulty
            if terrain_kind == "stairs_down":
                step_h *= -1.0
            height_text = f" H={100.0 * step_h:+.1f}cm"

        return (
            f"ter={terrain_kind} L={level}/{max_level} "
            f"D={difficulty:.2f}{height_text} T={terrain_type}"
        )

    def _draw_train_video_overlay(self, frame, lines):
        """Draw a compact semi-transparent text panel on one RGB uint8 tile."""
        image = Image.fromarray(frame)
        draw = ImageDraw.Draw(image, "RGBA")

        # Tile is usually only 320x180, so keep the font small.
        font_size = int(self.cfg.get("video_overlay_font_size", 10))
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                font_size,
            )
        except Exception:
            font = ImageFont.load_default()

        padding = 4
        line_gap = 1

        bboxes = [draw.textbbox((0, 0), line, font=font) for line in lines]
        text_width = max((bbox[2] - bbox[0]) for bbox in bboxes)
        text_height = sum((bbox[3] - bbox[1]) for bbox in bboxes) + line_gap * (len(lines) - 1)

        box_w = min(text_width + 2 * padding, frame.shape[1])
        box_h = min(text_height + 2 * padding, frame.shape[0])

        draw.rectangle([0, 0, box_w, box_h], fill=(0, 0, 0, 170))

        y = padding
        for line, bbox in zip(lines, bboxes):
            draw.text((padding, y), line, font=font, fill=(255, 255, 255, 255))
            y += (bbox[3] - bbox[1]) + line_gap

        return np.asarray(image, dtype=np.uint8)

    def _annotate_train_video_tile(self, frame, env_id):
        """Overlay command, actual velocity, and terrain information on one tile."""
        cmd = self.env.commands[env_id, :3].detach().cpu().numpy()

        if hasattr(self.env, "commands_given"):
            cmd_given = self.env.commands_given[env_id, :3].detach().cpu().numpy()
        else:
            cmd_given = cmd

        vel_x = float(self.env.base_lin_vel[env_id, 0].item())
        vel_y = float(self.env.base_lin_vel[env_id, 1].item())
        yaw_vel = float(self.env.base_ang_vel[env_id, 2].item())

        terrain_text = self._get_train_video_terrain_text(env_id)

        lines = [
            f"it={self.video_current_iteration} env={env_id}",
            f"cmd   {cmd[0]:+4.2f} {cmd[1]:+4.2f} {cmd[2]:+4.2f}",
            f"given {cmd_given[0]:+4.2f} {cmd_given[1]:+4.2f} {cmd_given[2]:+4.2f}",
            f"vel   {vel_x:+4.2f} {vel_y:+4.2f} {yaw_vel:+4.2f}",
            terrain_text,
        ]

        return self._draw_train_video_overlay(frame, lines)


    def _capture_train_video_frame(self):
        if self.video_writer is None or not self.video_cam_handles:
            return

        self._update_train_video_camera_locations()
        self.env.gym.step_graphics(self.env.sim)
        self.env.gym.render_all_camera_sensors(self.env.sim)

        if self.video_step_count % self.video_record_every == 0:
            tiles = []
            total_tiles = self.video_tile_rows * self.video_tile_cols

            for env_id, cam_handle in zip(self.video_env_ids, self.video_cam_handles):
                image = self.env.gym.get_camera_image(
                    self.env.sim,
                    self.env.envs[env_id],
                    cam_handle,
                    gymapi.IMAGE_COLOR,
                )
                frame = np.asarray(image, dtype=np.uint8).reshape(
                    (self.video_tile_height, self.video_tile_width, 4)
                )[:, :, :3].copy()
                frame = self._annotate_train_video_tile(frame, env_id)
                tiles.append(frame)

            while len(tiles) < total_tiles:
                tiles.append(self.video_black_tile)

            rows = []
            for row_index in range(self.video_tile_rows):
                row_start = row_index * self.video_tile_cols
                row_tiles = tiles[row_start:row_start + self.video_tile_cols]
                rows.append(np.concatenate(row_tiles, axis=1))

            mosaic_frame = np.concatenate(rows, axis=0)
            self.video_writer.write(mosaic_frame)

        self.video_step_count += 1
        self.video_steps_left -= 1
        if self.video_steps_left <= 0:
            self._close_train_video()

    def _close_train_video(self):
        if self.video_writer is None:
            return
        try:
            self.video_writer.close()
        except Exception as error:
            warnings.warn(f"Failed to finalize training video: {error}")
        finally:
            self.video_writer = None
            self.video_steps_left = 0
            self.video_step_count = 0

    def _checkpoint_path(self, iteration):
        save_dir = self.checkpoint_dir if self.checkpoint_dir is not None else self.log_dir
        if save_dir is None:
            raise RuntimeError("Cannot save checkpoint because log_dir is None")
        os.makedirs(save_dir, exist_ok=True)
        return os.path.join(save_dir, f"model_{iteration}.pt")

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        # initialize writer
        if self.log_dir is not None and self.writer is None:
            self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf,
                                                             high=int(self.env.max_episode_length))

        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        infos = {}
        infos["depth"] = self.env.depth_buffer.clone().to(self.device) if self.if_depth else None
        self.alg.actor_critic.train() # switch to train mode (for dropout for example)

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        tot_iter = self.current_learning_iteration + num_learning_iterations
        # self.act_shed,self.imi_shed,self.lag_shed = hard_phase_schedualer(max_iters=tot_iter,
        #             phase1_end=self.phase1_end)

        #imitation_mode
        if self.alg.actor_critic.imi_flag and self.cfg['resume']: 
            self.alg.actor_critic.imitation_mode()
            
        for it in range(self.current_learning_iteration, tot_iter):
            if hasattr(self.alg.actor_critic, "set_learning_iteration"):
                self.alg.actor_critic.set_learning_iteration(it)

            if self.record_video and it % self.video_interval == 0:
                self._start_train_video(it)
            # act_teacher_flag = self.act_shed[it]
            # imi_flag = self.imi_shed[it]
            # lag_flag = self.lag_shed[it]

            # self.alg.set_imi_flag(imi_flag)
            # self.alg.actor_critic.set_teacher_act(act_teacher_flag)
            # # self.env.randomize_lag_timesteps = lag_flag
            # # if self.env.randomize_lag_timesteps:
            # #     print("lag is on")
            # # else:
            # #     print("lag is off")
            if self.alg.actor_critic.imi_flag and self.cfg['resume']: 
                step_size = 1/int(tot_iter/2)
                imi_weight = max(0,1 - it * step_size)
                self.alg.set_imi_weight(imi_weight)
            
            start = time.time()
            # Rollout
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                   
                    actions = self.alg.act(obs, critic_obs, infos)
                    obs, privileged_obs, rewards,costs,dones, infos = self.env.step(actions)  # obs has changed to next_obs !! if done obs has been reset
                    self._capture_train_video_frame()
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs,rewards,costs,dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device),costs.to(self.device),dones.to(self.device)
                    self.alg.process_env_step(rewards,costs,dones, infos)

                    if self.log_dir is not None:
                        # Book keeping
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start

                # Learning step
                start = stop
                self.alg.compute_returns(critic_obs)
                self.alg.compute_cost_returns(critic_obs)

            #update k value for better expolration
            k_value = self.alg.update_k_value(it)
            
            mean_value_loss,mean_cost_value_loss,mean_viol_loss,mean_surrogate_loss, mean_imitation_loss = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(self._checkpoint_path(it), iteration=it)
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(self._checkpoint_path(self.current_learning_iteration), iteration=self.current_learning_iteration)
        self._close_train_video()

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        #mean_std = self.alg.actor_critic.std.mean()
        mean_std = self.alg.actor_critic.get_std().mean()
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))
        #mean_kl_loss,mean_recons_loss,mean_vel_recons_loss
        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/cost_value_function', locs['mean_cost_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Loss/mean_viol_loss', locs['mean_viol_loss'], locs['it'])
        self.writer.add_scalar('Loss/mean_imitation_loss', locs['mean_imitation_loss'], locs['it'])
        self.writer.add_scalar('Loss/learning_rate', self.alg.learning_rate, locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        if hasattr(self.alg.actor_critic, "last_current_alpha"):
            self.writer.add_scalar('Policy/residual_alpha', self.alg.actor_critic.last_current_alpha.item(), locs['it'])
        if hasattr(self.alg.actor_critic, "last_delta_norm"):
            self.writer.add_scalar('Policy/residual_delta_norm', self.alg.actor_critic.last_delta_norm.item(), locs['it'])
        if hasattr(self.alg.actor_critic, "last_saturation_ratio"):
            self.writer.add_scalar('Policy/action_saturation_ratio', self.alg.actor_critic.last_saturation_ratio.item(), locs['it'])
        self.writer.add_scalar('Perf/total_fps', fps, locs['it'])
        self.writer.add_scalar('Perf/collection time', locs['collection_time'], locs['it'])
        self.writer.add_scalar('Perf/learning_time', locs['learn_time'], locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'cost value function loss:':>{pad}} {locs['mean_cost_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'viol loss:':>{pad}} {locs['mean_viol_loss']:.4f}\n"""

                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'cost value function loss:':>{pad}} {locs['mean_cost_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'viol loss:':>{pad}} {locs['mean_viol_loss']:.4f}\n"""

                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n""")
            #   f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
            #   f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def save(self, path, infos=None, iteration=None):
        if iteration is None:
            iteration = self.current_learning_iteration
        state_dict = {
            'model_state_dict': self.alg.actor_critic.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': int(iteration),
            'infos': infos,
            }
        if self.if_depth:
            state_dict['depth_encoder_state_dict'] = self.alg.depth_encoder.state_dict()
            state_dict['depth_actor_state_dict'] = self.alg.depth_actor.state_dict()
        torch.save(state_dict, path)

    def load(self, path, load_optimizer=True):
        print("*" * 80)
        print("Loading model from {}...".format(path))
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        self.alg.estimator.load_state_dict(loaded_dict['estimator_state_dict'])
        if self.if_depth:
            if 'depth_encoder_state_dict' not in loaded_dict:
                warnings.warn("'depth_encoder_state_dict' key does not exist, not loading depth encoder...")
            else:
                print("Saved depth encoder detected, loading...")
                self.alg.depth_encoder.load_state_dict(loaded_dict['depth_encoder_state_dict'])
            if 'depth_actor_state_dict' in loaded_dict:
                print("Saved depth actor detected, loading...")
                self.alg.depth_actor.load_state_dict(loaded_dict['depth_actor_state_dict'])
            else:
                print("No saved depth actor, Copying actor critic actor to depth actor...")
                self.alg.depth_actor.load_state_dict(self.alg.actor_critic.actor.state_dict())
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        # self.current_learning_iteration = loaded_dict['iter']
        print("*" * 80)
        return loaded_dict['infos']

    def get_inference_policy(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic.act_inference
    
    def get_actor_critic(self, device=None):
        self.alg.actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.actor_critic.to(device)
        return self.alg.actor_critic
    
