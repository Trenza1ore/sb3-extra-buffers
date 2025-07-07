import numpy as np

_unsigned_int_types = [np.uint8, np.uint16, np.uint32, np.uint64]
_signed_int_types = [np.int8, np.int16, np.int32, np.int64]
_float_types = [np.float32, np.float64]
_max_val_lookup = {dtype: np.iinfo(dtype).max for dtype in (_unsigned_int_types + _signed_int_types)}
_max_val_lookup.update({dtype: np.finfo(dtype).max for dtype in _float_types})


def find_optimal_shape(arr_len: int, dtype: np.dtype = np.uint8) -> tuple[int, int, int]:
    """Find a way to slice longer 1D arrays"""
    max_col = _max_val_lookup[dtype]
    max_row = arr_len // max_col
    remainder = 0

    # Try to pack in equal-length slices
    if not arr_len % max_col:
        return max_row, max_col, remainder
    if not arr_len % max_row:
        max_col = arr_len // max_row
        return max_row, max_col, remainder

    # Fine, guess last row is a bit shorter...
    remainder = arr_len - (max_row * max_col)
    return max_row, max_col, remainder


def find_smallest_dtype(max_val: int, signed: bool = False, fallback: np.dtype = np.float32) -> np.dtype:
    """Find smallest dtype for runs_type"""
    dtypes = _signed_int_types if signed else _unsigned_int_types
    for d in (dtypes + _float_types):
        if max_val <= _max_val_lookup[d]:
            return d
    return fallback
