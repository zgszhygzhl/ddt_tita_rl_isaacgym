# Central task registration for train.py and play scripts.
# Add new tasks here once, then every script can call register_all_tasks().

from utils.task_registry import task_registry
from envs import LeggedRobot

from configs.tita_constraint_config import TitaConstraintRoughCfg, TitaConstraintRoughCfgPPO
from configs.d1h_constraint_config import D1HConstraintRoughCfg, D1HConstraintRoughCfgPPO
from configs.y1v0h_evt1_climb_config import (
    Y1v0hEvt1Climb,
    Y1v0hEvt1ClimbCfg,
    Y1v0hEvt1ClimbCfgPPO,
)

# d1h_base is optional so this file still works before d1h_base_config.py is copied in.
try:
    from configs.d1h_base_config import D1hBase, D1hBaseCfg, D1hBaseCfgPPO
except ModuleNotFoundError:
    D1hBase = None
    D1hBaseCfg = None
    D1hBaseCfgPPO = None

try:
    from configs.d1h_disc_residual_config import (
        D1hDiscResidual,
        D1hDiscResidualCfg,
        D1hDiscResidualCfgPPO,
    )
except ModuleNotFoundError:
    D1hDiscResidual = None
    D1hDiscResidualCfg = None
    D1hDiscResidualCfgPPO = None

def register_all_tasks():
    """Register every available task exactly once for the current process."""

    task_registry.register(
        "tita_constraint",
        LeggedRobot,
        TitaConstraintRoughCfg(),
        TitaConstraintRoughCfgPPO(),
    )

    task_registry.register(
        "d1h_constraint",
        LeggedRobot,
        D1HConstraintRoughCfg(),
        D1HConstraintRoughCfgPPO(),
    )

    task_registry.register(
        "d1h_evt1_climb",
        Y1v0hEvt1Climb,
        Y1v0hEvt1ClimbCfg(),
        Y1v0hEvt1ClimbCfgPPO(),
    )

    if D1hBase is not None:
        task_registry.register(
            "d1h_base",
            D1hBase,
            D1hBaseCfg(),
            D1hBaseCfgPPO(),
        )

    if D1hDiscResidual is not None:
        task_registry.register(
            "d1h_disc_residual",
            D1hDiscResidual,
            D1hDiscResidualCfg(),
            D1hDiscResidualCfgPPO(),
        )
