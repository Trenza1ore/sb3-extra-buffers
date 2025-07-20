from collections import namedtuple
import gzip
import numpy as np
from sb3_extra_buffers.compressed.utils import find_smallest_dtype

HAS_IGZIP: bool = False
HAS_NUMBA: bool = False

try:
    import isal.igzip as igzip
    HAS_IGZIP: bool = True
except ImportError:
    igzip = gzip

CompressionMethods = namedtuple("CompressionMethod", ["compress", "decompress"])


def rle_compress(arr: np.ndarray, elem_type: np.dtype = np.uint8, runs_type: np.dtype = np.uint16) -> bytes:
    """RLE Compression, credits:
    https://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array-run-length-encodi/32681075#32681075
    """
    n = len(arr)
    if n == 0:
        return None
    else:
        y = arr[1:] != arr[:-1]
    idx_arr = np.append(np.where(y), n - 1)
    runs = np.diff(np.append(-1, idx_arr))
    return runs.astype(runs_type, copy=False).tobytes() + arr[idx_arr].astype(elem_type, copy=False).tobytes()


def rle_numpy_decompress(data: bytes, elem_type: np.dtype, runs_type: np.dtype, arr_configs: dict) -> np.ndarray:
    """RLE Decompression, NumPy vectorized"""
    # Find how to split bytes
    data_len = len(data)
    runs_itemsize = int(np.dtype(runs_type).itemsize)
    elem_itemsize = int(np.dtype(elem_type).itemsize)
    run_count = data_len // (runs_itemsize + elem_itemsize)
    runs_totalsize = run_count * runs_itemsize

    # Find array length and suitable dtypes for intermediate calculations (we don't want floats!)
    arr_length = arr_configs["shape"]
    intermediate_dtype = find_smallest_dtype(arr_length, signed=False, fallback=np.int64)
    padding = np.array([0], dtype=intermediate_dtype)

    # Get elements, runs back from bytes, calculate start_pos for each run
    runs = np.frombuffer(data[:runs_totalsize], dtype=runs_type)
    elements = np.frombuffer(data[runs_totalsize:runs_totalsize + run_count * elem_itemsize], dtype=elem_type)
    start_pos = np.cumsum(np.append(padding, runs), dtype=intermediate_dtype)[:-1]

    # Indexing magics
    run_indices = np.repeat(np.arange(run_count), runs)
    cumulative_starts = np.concatenate([padding, np.cumsum(runs, axis=0, dtype=intermediate_dtype)[:-1]])
    offsets = np.arange(arr_length, dtype=intermediate_dtype) - cumulative_starts[run_indices]
    del cumulative_starts, run_indices
    indices = np.repeat(start_pos, runs) + offsets

    out = np.empty(**arr_configs)
    out[indices] = np.repeat(elements, runs)
    return out


def rle_numpy_decompress_old(data: bytes, elem_type: np.dtype, runs_type: np.dtype, arr_configs: dict) -> np.ndarray:
    """RLE Decompression, old version, less vectorized"""
    data_len = len(data)
    runs_itemsize = int(np.dtype(runs_type).itemsize)
    elem_itemsize = int(np.dtype(elem_type).itemsize)
    run_count = data_len // (runs_itemsize + elem_itemsize)

    runs_totalsize = run_count * runs_itemsize
    runs = np.frombuffer(data[:runs_totalsize], dtype=runs_type)
    elements = np.frombuffer(data[runs_totalsize:runs_totalsize + run_count * elem_itemsize], dtype=elem_type)

    out = np.zeros(**arr_configs)
    idx = 0
    for run, elem in zip(runs, elements):
        run_len = int(run)
        out[idx:idx + run_len] = elem
        idx += run_len
    return out


def gzip_compress(arr: np.ndarray, *args, compresslevel: int = 9, **kwargs) -> bytes:
    """gzip Compression"""
    return gzip.compress(arr, compresslevel)


def gzip_decompress(data: bytes, *args, elem_type: np.dtype = np.uint8, **kwargs) -> np.ndarray:
    """gzip Decompression"""
    return np.frombuffer(gzip.decompress(data), dtype=elem_type)


def igzip_compress(arr: np.ndarray, *args, compresslevel: int = 9, **kwargs) -> bytes:
    """igzip Compression"""
    return igzip.compress(arr, min(compresslevel, 3))


def igzip_decompress(data: bytes, *args, elem_type: np.dtype = np.uint8, **kwargs) -> np.ndarray:
    """igzip Decompression"""
    return np.frombuffer(igzip.decompress(data), dtype=elem_type)


def no_compress(arr: np.ndarray, *args, elem_type: np.dtype = np.uint8, **kwargs) -> bytes:
    """Skip Compression"""
    return arr.astype(elem_type).tobytes()


def no_decompress(data: bytes, *args, elem_type: np.dtype, **kwargs) -> np.ndarray:
    """Skip Decompression"""
    return np.frombuffer(data, dtype=elem_type)


def has_numba() -> bool:
    return HAS_NUMBA


def has_igzip() -> bool:
    return HAS_IGZIP


try:
    from sb3_extra_buffers.compressed.compression_methods.compression_methods_numba import rle_numba_decompress
    HAS_NUMBA = True
except ImportError:
    rle_numba_decompress = rle_numpy_decompress

COMPRESSION_METHOD_MAP: dict[str, CompressionMethods] = {
    "rle": CompressionMethods(compress=rle_compress, decompress=rle_numpy_decompress),
    "rle-jit": CompressionMethods(compress=rle_compress, decompress=rle_numba_decompress),
    "rle-old": CompressionMethods(compress=rle_compress, decompress=rle_numpy_decompress_old),
    "gzip": CompressionMethods(compress=gzip_compress, decompress=gzip_decompress),
    "igzip": CompressionMethods(compress=igzip_compress, decompress=igzip_decompress),
    "none": CompressionMethods(compress=no_compress, decompress=no_decompress),
}
