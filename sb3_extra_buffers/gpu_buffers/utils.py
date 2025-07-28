import torch as th

_unsigned_int_types = [th.uint8, th.uint16, th.uint32, th.uint64]
_signed_int_types = [th.int8, th.int16, th.int32, th.int64]
_float_types = [th.float32, th.float64]
_max_val_lookup = {dtype: th.iinfo(dtype).max for dtype in (_unsigned_int_types + _signed_int_types)}
_max_val_lookup.update({dtype: th.finfo(dtype).max for dtype in _float_types})


def find_smallest_dtype(max_val: int, signed: bool = False, fallback: th.dtype = th.float32) -> th.dtype:
    """Find smallest dtype for runs_type"""
    dtypes = _signed_int_types if signed else _unsigned_int_types
    for d in (dtypes + _float_types):
        if max_val <= _max_val_lookup[d]:
            return d
    return fallback
