"""Base classes and helpers for GPU observation storage."""

# pylint: disable=too-few-public-methods

import re
from functools import partial
from typing import Any, Optional, Union

import numpy as np
import torch as th

from sb3_extra_buffers import __version__
from sb3_extra_buffers.gpu_buffers.utils import find_smallest_dtype


def find_gpu_buffer_dtypes(
    obs_shape: Union[int, tuple],
    elem_dtype: th.dtype = th.uint8,
    compression_method: str = "rle",
) -> dict[str, Any]:
    """Find Torch dtypes for GPU buffer compression."""
    del compression_method
    if isinstance(obs_shape, tuple):
        obs_shape = int(np.prod(obs_shape))
    runs_type = find_smallest_dtype(int(obs_shape))
    if runs_type in (th.uint8, th.int8):
        runs_type = th.uint16
    return {"elem_type": elem_dtype, "runs_type": runs_type}


class BaseGpuBuffer:
    """Base GPU buffer class wiring compression callables."""

    def __init__(
        self,
        compression_method: Optional[str] = None,
        compression_kwargs: Optional[dict] = None,
        decompression_kwargs: Optional[dict] = None,
        flatten_config: Optional[dict] = None,
    ):
        """Configure compression and decompression callables."""
        self.version = __version__
        if compression_method is None:
            return
        from sb3_extra_buffers.gpu_buffers.compression_methods import COMPRESSION_METHOD_MAP

        compression_kwargs = compression_kwargs or {}
        decompression_kwargs = decompression_kwargs or {}
        if compression_method[-1].isdigit():
            re_match = re.search(r"^((?:[A-Za-z]+)|(?:[\w\-]+/))(\-?[0-9]+)$", compression_method)
            assert re_match, f"Invalid compression shorthand: {compression_method}"
            compression_method = re_match.group(1).removesuffix("/")
            compression_kwargs["compresslevel"] = int(re_match.group(2))
        assert compression_method in COMPRESSION_METHOD_MAP, f"Unknown compression method {compression_method}"
        self.compression_method = compression_method
        self._compress = partial(COMPRESSION_METHOD_MAP[compression_method].compress, **compression_kwargs)
        self._decompress = partial(
            COMPRESSION_METHOD_MAP[compression_method].decompress,
            arr_configs=flatten_config,
            **decompression_kwargs,
        )
