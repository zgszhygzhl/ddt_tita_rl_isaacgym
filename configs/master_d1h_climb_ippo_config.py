from configs.master_d1h_climb_config import (
    MasterD1HClimb,
    MasterD1HClimbCfg,
    MasterD1HClimbCfgPPO,
)

class MasterD1HClimbIPPOCfg(MasterD1HClimbCfg):
    """Same environment, rewards, terrain, commands and costs as master_d1h_climb."""
    pass

class MasterD1HClimbIPPOCfgPPO(MasterD1HClimbCfgPPO):
    """IPPO variant of master_d1h_climb.

    Only algorithm-level IPPO options are changed.
    Environment and reward definitions remain unchanged.
    """

    class algorithm(MasterD1HClimbCfgPPO.algorithm):
        use_ippo = True

        ippo_state_source = "obs_action"
        ippo_score_mode = "sm_inverse"
        ippo_select_mode = "topk"
        ippo_retain_ratio = 0.5

        ippo_z_dim_limit = 64
        ippo_z_norm = True
        ippo_ridge = 1e-3
        ippo_reset_T_each_rollout = True

        ippo_threshold = 0.5
        ippo_min_retain_ratio = 0.1
        ippo_max_retain_ratio = 1.0

        ippo_weight_mode = "normalized_score"
        ippo_weight_clip = 3.0
        ippo_weight_eps = 1e-6

        ippo_apply_to_actor = True
        ippo_apply_to_critic = False
        ippo_apply_to_entropy = False

        ippo_log_analysis = True
        ippo_analysis_interval = 10
        ippo_gradient_probe_interval = 50
        ippo_gradient_probe_num_batches = 8

    class runner(MasterD1HClimbCfgPPO.runner):
        run_name = "master_d1h_ippo_topk50"
        experiment_name = "master_d1h_climb_ippo"

