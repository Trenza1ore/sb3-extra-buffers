"""Base types for record buffers and no-op array stubs."""

from typing import Union

import numpy as np


class DummyArray:
    """A dummy array with NumPy-like interfaces."""

    def __setitem__(self, *args, **kwargs) -> None:
        """Ignore item assignment."""
        return None

    def transpose(self, *args, **kwargs) -> None:
        """Ignore transpose calls."""
        return None

    def fill(self, *args, **kwargs) -> None:
        """Ignore fill calls."""
        return None


class BaseRecordBuffer:
    """For type-checking."""

    frames: Union[np.ndarray, DummyArray]
    features: Union[np.ndarray, DummyArray]
    rewards: Union[np.ndarray, DummyArray]
    actions: Union[np.ndarray, DummyArray]
