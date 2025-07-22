import lz4.frame
import lz4.block
import numpy as np


def lz4_frame_compress(arr: np.ndarray, *args, compresslevel: int = 0, block_size: int = 0, **kwargs) -> bytes:
    """lz4 frame Compression"""
    return lz4.frame.compress(arr, compression_level=compresslevel, block_size=block_size)


def lz4_frame_decompress(data: bytes, *args, elem_type: np.dtype = np.uint8, **kwargs) -> np.ndarray:
    """lz4 frame Decompression"""
    return np.frombuffer(lz4.frame.decompress(data), dtype=elem_type)


def lz4_block_compress(arr: np.ndarray, *args, compresslevel: int = 9, **kwargs) -> bytes:
    """lz4 block Compression"""
    if compresslevel < 0:
        return lz4.block.compress(arr, mode="fast", acceleration=-compresslevel)
    elif compresslevel == 0:
        return lz4.block.compress(arr)
    return lz4.block.compress(arr, mode="high_compression", compression=compresslevel)


def lz4_block_decompress(data: bytes, *args, elem_type: np.dtype = np.uint8, **kwargs) -> np.ndarray:
    """lz4 block Decompression"""
    return np.frombuffer(lz4.block.decompress(data), dtype=elem_type)
