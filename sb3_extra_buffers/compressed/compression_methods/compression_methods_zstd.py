import zstd
import numpy as np


def zstd_compress(arr: np.ndarray, *args, compresslevel: int = 0, threads: int = 0, **kwargs) -> bytes:
    """zstd Compression"""
    return zstd.compress(arr, compresslevel, threads)


def zstd_decompress(data: bytes, *args, elem_type: np.dtype = np.uint8, **kwargs) -> np.ndarray:
    """zstd Decompression"""
    return np.frombuffer(zstd.decompress(data), dtype=elem_type)
