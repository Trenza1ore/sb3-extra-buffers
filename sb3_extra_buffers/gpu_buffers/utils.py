"""Torch dtype helpers for GPU buffer compression."""

from functools import lru_cache
from typing import Union

import numpy as np
import torch as th

_unsigned_int_types = [th.uint8, th.uint16, th.uint32, th.uint64]
_signed_int_types = [th.int8, th.int16, th.int32, th.int64]
_float_types = [th.float32, th.float64]
_max_val_lookup = {dtype: th.iinfo(dtype).max for dtype in (_unsigned_int_types + _signed_int_types)}
_max_val_lookup.update({dtype: th.finfo(dtype).max for dtype in _float_types})


@lru_cache(typed=True)
def find_smallest_dtype(max_val: int, signed: bool = False, fallback: th.dtype = th.float32) -> th.dtype:
    """Find smallest dtype for runs_type."""
    dtypes = _signed_int_types if signed else _unsigned_int_types
    for d in dtypes + _float_types:
        if max_val <= _max_val_lookup[d]:
            return d
    return fallback


@lru_cache(typed=True)
def torch_dtype_element_size(dtype: th.dtype) -> int:
    """Return the element size in bytes for a Torch dtype."""
    return th.tensor([], dtype=dtype).element_size()


@lru_cache
def estimate_max_slot_bytes(
    flat_len: int,
    elem_type: th.dtype,
    runs_type: th.dtype,
    compression_method: str,
) -> int:
    """Estimate per-slot byte capacity for dense or compressed storage."""
    elem_bytes = flat_len * torch_dtype_element_size(elem_type)
    if compression_method == "none":
        return elem_bytes
    if compression_method == "zstd":
        return elem_bytes // 2 + elem_bytes // 255 + 64
    return elem_bytes + flat_len * torch_dtype_element_size(runs_type)


@lru_cache(typed=True)
def numpy_dtype_to_torch(dtype: Union[np.dtype, type, str]) -> th.dtype:
    """Convert a NumPy dtype to the matching Torch dtype."""
    return th.from_numpy(np.empty(0, dtype=dtype)).dtype


@lru_cache
def torch_dtype_to_numpy(dtype: th.dtype) -> np.dtype:
    """Convert a Torch dtype to the matching NumPy dtype."""
    return th.zeros(1, dtype=dtype).cpu().numpy().dtype
