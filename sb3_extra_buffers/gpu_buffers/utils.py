"""Torch dtype helpers for GPU buffer compression."""

from functools import lru_cache
from typing import Optional, Union

import numpy as np
import torch as th

from sb3_extra_buffers.gpu_buffers.size_estimation import (
    estimate_save_ratio,
    is_benchmark_family,
    parse_compression_method,
)

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


def _legacy_max_slot_bytes(flat_len: int, elem_type: th.dtype, runs_type: th.dtype, compression_method: str) -> int:
    elem_bytes = flat_len * torch_dtype_element_size(elem_type)
    if compression_method == "none":
        return elem_bytes
    if compression_method.startswith("zstd"):
        return elem_bytes // 4 + elem_bytes // 255 + 64
    return elem_bytes + flat_len * torch_dtype_element_size(runs_type)


@lru_cache(typed=True)
def estimate_max_slot_bytes(
    flat_len: int,
    elem_type: th.dtype,
    runs_type: th.dtype,
    compression_method: str,
    overalloc_factor: float = 1.5,
    compresslevel: Optional[int] = None,
) -> int:
    """Estimate per-cell byte capacity for packed heap storage.

    Uses the larger of the MsPacman benchmark ratio and the legacy analytic
    bound so incompressible or noisy frames do not exceed the reserved span.

    Args:
        flat_len: Flattened observation length.
        elem_type: Observation element dtype on the buffer device.
        runs_type: Run-length dtype used by the legacy analytic bound.
        compression_method: Codec name or shorthand (e.g. ``zstd3``, ``zstd-5``).
        overalloc_factor: Headroom multiplier applied to the benchmark ratio.
        compresslevel: If set, overrides any level parsed from ``compression_method``.
    """
    family, _ = parse_compression_method(compression_method)
    legacy = _legacy_max_slot_bytes(flat_len, elem_type, runs_type, compression_method)
    if not is_benchmark_family(family):
        return legacy
    elem_bytes = flat_len * torch_dtype_element_size(elem_type)
    ratio = estimate_save_ratio(compression_method, compresslevel=compresslevel)
    benchmark = max(64, int(elem_bytes * ratio * overalloc_factor))
    return max(benchmark, legacy)


@lru_cache(typed=True)
def estimate_total_heap_bytes(
    n_cells: int,
    flat_len: int,
    elem_type: th.dtype,
    runs_type: th.dtype,
    compression_method: str,
    overalloc_factor: float = 1.5,
    compresslevel: Optional[int] = None,
) -> int:
    """Estimate total heap capacity for ``n_cells`` packed observations."""
    per_cell = estimate_max_slot_bytes(
        flat_len,
        elem_type,
        runs_type,
        compression_method,
        overalloc_factor=overalloc_factor,
        compresslevel=compresslevel,
    )
    return n_cells * per_cell


@lru_cache(typed=True)
def numpy_dtype_to_torch(dtype: Union[np.dtype, type, str]) -> th.dtype:
    """Convert a NumPy dtype to the matching Torch dtype."""
    return th.from_numpy(np.empty(0, dtype=dtype)).dtype


@lru_cache
def torch_dtype_to_numpy(dtype: th.dtype) -> np.dtype:
    """Convert a Torch dtype to the matching NumPy dtype."""
    return th.zeros(1, dtype=dtype).cpu().numpy().dtype
