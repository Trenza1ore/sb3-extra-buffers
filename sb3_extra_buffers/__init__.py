from typing import Union

import numpy as np
from stable_baselines3.common.buffers import BaseBuffer

from sb3_extra_buffers.current_version import package_version
from sb3_extra_buffers.recording.base import BaseRecordBuffer

ReplayLike = Union[BaseBuffer, BaseRecordBuffer]
NumberType = Union[int, float, np.integer, np.floating]

__version__ = package_version
