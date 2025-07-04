from typing import Literal, Generator, Optional, Union

import warnings
from functools import partial
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import BaseBuffer, ReplayBuffer
from sb3_extra_buffers.compressed.compression_methods import _compression_method_mapping
from sb3_extra_buffers.compressed.utils import find_optimal_shape


class CompressedReplayBuffer(ReplayBuffer):
    raise NotImplementedError("CompressedReplayBuffer not yet implemented")
