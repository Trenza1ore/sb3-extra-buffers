import gzip
try:
    import isal.igzip as igzip
except:
    igzip = gzip
from collections import namedtuple
import numpy as np

CompressionMethods = namedtuple("CompressionMethod", ["compress", "decompress", "rle_like"])


def rle_compress(arr: np.ndarray, len_type: np.dtype = np.uint16, pos_type: np.dtype = np.uint16,
                 elem_type: np.dtype = np.uint8) -> tuple[bytes, bytes, bytes]:
    """RLE Compression, credits:
    https://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array-run-length-encodi/32681075#32681075
    """
    n = len(arr)
    if n == 0:
        return (None, None, None)
    else:
        y = arr[1:] != arr[:-1]
    idx_arr = np.append(np.where(y), n - 1)
    len_arr = np.diff(np.append(-1, idx_arr))
    pos_arr = np.cumsum(np.append(0, len_arr))[:-1]
    return (len_arr.astype(len_type, copy=False).tobytes(),
            pos_arr.astype(pos_type, copy=False).tobytes(),
            arr[idx_arr].astype(elem_type, copy=False).tobytes())


def rle_decompress(len_arr: np.ndarray, pos_arr: np.ndarray, elements: np.ndarray, arr_configs: dict) -> np.ndarray:
    """RLE Decompression"""
    sum_lengths = len_arr.sum()
    run_indices = np.repeat(np.arange(len(len_arr)), len_arr)

    # Compute indices using vectorized operations
    cumulative_starts = np.concatenate([
        np.array([0], dtype=np.uint8),
        np.cumsum(len_arr, axis=0)[:-1]
    ])
    offsets = np.arange(sum_lengths) - cumulative_starts[run_indices]
    indices = np.repeat(pos_arr, len_arr) + offsets

    # Create values and assign to output tensor
    values = np.repeat(elements, len_arr)
    arr_reconstructed = np.empty(**arr_configs)
    arr_reconstructed[indices] = values
    return arr_reconstructed


def gzip_compress(arr: np.ndarray, len_type: None = None, pos_type: None = None, elem_type: np.dtype = np.uint8,
                  compresslevel: int = 9, **kwargs) -> tuple[None, None, bytes]:
    """gzip Compression"""
    return None, None, gzip.compress(arr, compresslevel=compresslevel, **kwargs)


def gzip_decompress(len_arr: None, pos_arr: None, elements: bytes, arr_configs: None,
                    elem_type: np.dtype = np.uint8) -> np.ndarray:
    """gzip Decompression"""
    return np.frombuffer(gzip.decompress(elements), dtype=elem_type)


def igzip_compress(arr: np.ndarray, len_type: None = None, pos_type: None = None, elem_type: np.dtype = np.uint8,
                  compresslevel: int = 3, **kwargs) -> tuple[None, None, bytes]:
    """igzip Compression"""
    return None, None, igzip.compress(arr, compresslevel=max(compresslevel, 3), **kwargs)


def igzip_decompress(len_arr: None, pos_arr: None, elements: bytes, arr_configs: None,
                    elem_type: np.dtype = np.uint8) -> np.ndarray:
    """igzip Decompression"""
    return np.frombuffer(igzip.decompress(elements), dtype=elem_type)


def no_compress(arr: np.ndarray, len_type: None = None, pos_type: None = None, elem_type: np.dtype = np.uint8,
                **kwargs) -> tuple[None, None, bytes]:
    """Skip Compression"""
    return None, None, arr.astype(elem_type).tobytes()


def no_decompress(len_arr: None, pos_arr: None, elements: bytes, arr_configs: None,
                  elem_type: np.dtype = np.uint8) -> np.ndarray:
    """Skip Decompression"""
    return np.frombuffer(elements, dtype=elem_type)


_compression_method_mapping: dict[str, CompressionMethods] = {
    "rle": CompressionMethods(compress=rle_compress, decompress=rle_decompress, rle_like=True),
    "gzip": CompressionMethods(compress=gzip_compress, decompress=gzip_decompress, rle_like=False),
    "none": CompressionMethods(compress=no_compress, decompress=no_decompress, rle_like=False),
}
