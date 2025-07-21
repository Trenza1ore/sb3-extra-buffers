from typing import Union

from stable_baselines3.common.buffers import BaseBuffer
from sb3_extra_buffers.current_version import package_version
from sb3_extra_buffers.recording.base import BaseRecordBuffer


BufferType = Union[BaseBuffer, BaseRecordBuffer]

__version__ = package_version
