from typing import Union

import numpy as np


class DummyArray:
    """A dummy array with NumPy-like interfaces"""
    def __setitem__(self, *args, **kwargs) -> None:
        return None

    def transpose(self, *args, **kwargs) -> None:
        return None

    def fill(self, *args, **kwargs) -> None:
        return None


class BaseRecordBuffer:
    """For type-checking"""
    frames: Union[np.ndarray, DummyArray]
    features: Union[np.ndarray, DummyArray]
    rewards: Union[np.ndarray, DummyArray]
    actions: Union[np.ndarray, DummyArray]
