import numpy as np
from numba import njit


@njit
def _rle_numba_decompress(elements: np.ndarray, runs: np.ndarray, out: np.ndarray) -> np.ndarray:
    """RLE Decompression with Numba JIT"""
    idx = 0
    for i in range(len(runs)):
        run_len = int(runs[i])
        out[idx:idx + run_len] = elements[i]
        idx += run_len
    return out


def rle_numba_decompress(data: bytes, *args, elem_type: np.dtype, runs_type: np.dtype,
                         arr_configs: dict, **kwargs) -> np.ndarray:
    """RLE Decompression with Numba JIT (wrapped)"""
    data_len = len(data)
    out = np.zeros(**arr_configs)
    runs_itemsize = int(np.dtype(runs_type).itemsize)
    elem_itemsize = int(np.dtype(elem_type).itemsize)
    run_count = data_len // (runs_itemsize + elem_itemsize)

    runs_totalsize = run_count * runs_itemsize
    runs = np.frombuffer(data[:runs_totalsize], dtype=runs_type)
    elements = np.frombuffer(data[runs_totalsize:runs_totalsize + run_count * elem_itemsize], dtype=elem_type)

    return _rle_numba_decompress(elements, runs, out)


def init_jit(*, elem_type: np.dtype = np.uint8, runs_type: np.dtype = np.uint16, **kwargs):
    """Initialize Numba JIT"""
    dummy_len = np.array([1], dtype=runs_type)
    dummy_val = np.array([1], dtype=elem_type)
    dummy_out = np.zeros(shape=1, dtype=np.float32)
    _rle_numba_decompress(dummy_len, dummy_val, dummy_out)
