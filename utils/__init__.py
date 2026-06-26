from .helpers import (
    class_to_dict,
    get_load_path,
    get_args,
    export_policy_as_jit,
    set_seed,
    update_class_from_dict,
    hard_phase_schedualer,
)

from .logger import Logger
from .math import quat_apply_yaw, wrap_to_pi, torch_rand_sqrt_float, get_scale_shift
from .terrain import Terrain

from .utils import (
    split_and_pad_trajectories,
    unpad_trajectories,
    quaternion_slerp,
    Normalize,
    Normalizer,
    RunningMeanStd,
)


def __getattr__(name):
    """
    Lazy import task_registry to avoid circular imports.

    不要在 utils/__init__.py 顶层 import task_registry。
    否则任何 from utils import xxx 都会触发：
        utils -> task_registry -> runner -> algorithm -> rollout_storage -> utils
    从而产生 partially initialized module 的循环导入错误。
    """
    if name == "task_registry":
        from .task_registry import task_registry
        return task_registry
    raise AttributeError(f"module 'utils' has no attribute {name!r}")