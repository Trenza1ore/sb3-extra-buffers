from collections import namedtuple
import numpy as np
import torch as th
from sb3_extra_buffers import logger
from sb3_extra_buffers.compressed.utils import find_smallest_dtype
from sb3_extra_buffers.gpu_buffers.raw_buffer import RawBuffer

CompressionMethods = namedtuple("CompressionMethod", ["compress", "decompress"])


def rle_compress(arr: th.tensor, buffer: RawBuffer, elem_type: th.dtype = th.uint8, runs_type: th.dtype = th.uint16):
    """RLE Compression, credits:
    https://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array-run-length-encodi/32681075#32681075
    """
    n = len(arr)
    idx_arr = th.cat([th.where(arr[1:] != arr[:-1])[0], th.tensor([n - 1], dtype=arr.dtype, device=arr.device)])
    runs = th.diff(th.cat([th.tensor([-1], dtype=arr.dtype, device=arr.device), idx_arr]))

    run_length = len(runs)
    len_runs = run_length * runs_type.itemsize
    len_elem = run_length * elem_type.itemsize
    malloc = buffer.malloc(len_runs + len_elem)
    pos_runs = malloc[0]
    pos_elem = pos_runs + len_runs

    # buffer.write_bytes((pos_runs, run_length), runs.to(dtype=runs_type))
    # buffer.write_bytes((pos_elem, run_length), arr[idx_arr].to(dtype=elem_type))
    return pos_runs, pos_elem, run_length, runs.to(dtype=runs_type), arr[idx_arr].to(dtype=elem_type)


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


COMPRESSION_METHOD_MAP: dict[str, CompressionMethods] = {
    "rle": CompressionMethods(compress=rle_compress, decompress=rle_numpy_decompress),
}

logger.info(f"Loaded GPU compression methods:\n{", ".join(COMPRESSION_METHOD_MAP)}")
