from configs.tita_constraint_config import TitaConstraintRoughCfg, TitaConstraintRoughCfgPPO


class D1HConstraintRoughCfg(TitaConstraintRoughCfg):
    class init_state(TitaConstraintRoughCfg.init_state):
        pos = [0.0, 0.0, 0.5]
        rot = [0, 0.0, 0.0, 1]

        default_joint_angles = {
            "FL_hip_joint": 0,
            "FR_hip_joint": 0,

            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,

            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,

            "FL_foot_joint": 0,
            "FR_foot_joint": 0,
        }

        desired_feet_distance = 0.4

    class control(TitaConstraintRoughCfg.control):
        control_type = "P"

        stiffness = {
            "hip": 40.,
            "thigh": 40.,
            "calf": 40.,
            "foot": 10.,
        }

        damping = {
            "hip": 1.0,
            "thigh": 1.0,
            "calf": 1.0,
            "foot": 0.5,
        }

        action_scale = 0.5
        decimation = 4
        hip_scale_reduction = 0.5
        use_filter = True

    class asset(TitaConstraintRoughCfg.asset):
        file = "{ROOT_DIR}/resources/d1h/urdf/robot.urdf"
        name = "d1h"

        foot_name = "foot"

        penalize_contacts_on = ["calf"]
        terminate_after_contacts_on = ["base"]

        self_collisions = 0
        flip_visual_attachments = False

    class rewards(TitaConstraintRoughCfg.rewards):
        base_height_target = 0.5


class D1HConstraintRoughCfgPPO(TitaConstraintRoughCfgPPO):
    class runner(TitaConstraintRoughCfgPPO.runner):
        experiment_name = "d1h_constraint"
        run_name = "d1h_from_tita_framework"
        resume = False
        resume_path = ""
        max_iterations = 10000
