import isaacgym
import numpy as np
import os
from datetime import datetime
from configs.tita_constraint_config import TitaConstraintRoughCfg, TitaConstraintRoughCfgPPO
from configs.d1h_constraint_config import D1HConstraintRoughCfg, D1HConstraintRoughCfgPPO
from configs.y1v0h_evt1_climb_config import Y1v0hEvt1Climb, Y1v0hEvt1ClimbCfg, Y1v0hEvt1ClimbCfgPPO
from configs.d1h_base_config import D1hBase, D1hBaseCfg, D1hBaseCfgPPO
from envs.no_constrains_legged_robot import Tita

from global_config import ROOT_DIR, ENVS_DIR

from utils.helpers import get_args
from envs import LeggedRobot
from utils.task_registry import task_registry

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)

    # Define the log path and task configuration folder path
    logs_path = os.path.join(ROOT_DIR, "logs")
    task_config_folder = os.path.join(logs_path, f"{args.task}")

    # Check if the task configuration folder exists and save configurations
    if os.path.exists(task_config_folder) and os.path.isdir(task_config_folder):
        print(f"Task configuration folder exists: {task_config_folder}, saving configuration files.")
        task_registry.save_cfgs(name=args.task, env_cfg=env_cfg, train_cfg=train_cfg)
    else:
        print(f"Task configuration folder does not exist: {task_config_folder}, skipping configuration saving.")

    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':

    task_registry.register("tita_constraint", LeggedRobot, TitaConstraintRoughCfg(), TitaConstraintRoughCfgPPO())
    task_registry.register("d1h_constraint", LeggedRobot, D1HConstraintRoughCfg(), D1HConstraintRoughCfgPPO())
    task_registry.register("d1h_evt1_climb", Y1v0hEvt1Climb, Y1v0hEvt1ClimbCfg(), Y1v0hEvt1ClimbCfgPPO())
    task_registry.register("d1h_base", D1hBase, D1hBaseCfg(), D1hBaseCfgPPO())
    
    args = get_args()
    train(args)
