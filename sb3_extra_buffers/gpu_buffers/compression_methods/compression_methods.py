from collections import namedtuple

import numpy as np
import torch as th

from sb3_extra_buffers import logger
from sb3_extra_buffers.gpu_buffers.raw_buffer import RawBuffer
from sb3_extra_buffers.gpu_buffers.utils import find_smallest_dtype

CompressionMethods = namedtuple("CompressionMethod", ["compress", "decompress"])


def rle_compress(
    arr: th.Tensor,
    buffer: RawBuffer,
    elem_type: th.dtype = th.uint8,
    runs_type: th.dtype = th.uint16,
) -> tuple[int, int, int]:
    """RLE Compression, credits:
    https://stackoverflow.com/questions/1066758/find-length-of-sequences-of-identical-values-in-a-numpy-array-run-length-encodi/32681075#32681075
    """
    n = arr.size(0)

    # Safe index finding
    change_idxs = th.where(arr[1:] != arr[:-1])[0]
    idx_arr = th.cat(
        [change_idxs, th.tensor([n - 1], dtype=th.long, device=arr.device)]
    )

    # Compute run lengths safely
    prev_idx = th.cat([th.tensor([-1], dtype=th.long, device=arr.device), idx_arr[:-1]])
    runs = (idx_arr - prev_idx).to(dtype=runs_type)

    values = arr[idx_arr].to(dtype=elem_type)
    run_length = runs.size(0)
    len_runs = run_length * runs_type.itemsize
    len_elem = run_length * elem_type.itemsize

    malloc = buffer.malloc(len_runs + len_elem)
    pos_runs = malloc[0]
    pos_elem = pos_runs + len_runs

    buffer.write_bytes((pos_runs, run_length), runs)
    buffer.write_bytes((pos_elem, run_length), values)

    return pos_runs, pos_elem, run_length


def rle_decompress(
    buffer: RawBuffer,
    pos_runs: int,
    pos_elem: int,
    run_length: int,
    elem_type: th.dtype,
    runs_type: th.dtype,
    arr_configs: dict,
) -> th.Tensor:
    """RLE Decompression, PyTorch version"""
    # Find array length and suitable dtypes for intermediate calculations (we don't want floats!)
    arr_length = arr_configs["size"]
    intermediate_dtype = find_smallest_dtype(arr_length, signed=True, fallback=th.int64)
    padding = th.tensor([0], dtype=intermediate_dtype)

    # Get elements, runs back from bytes, calculate start_pos for each run
    runs = buffer.read_bytes((pos_runs, run_length), dtype=runs_type)
    elements = buffer.read_bytes((pos_elem, run_length), dtype=elem_type)
    start_pos = th.cumsum(th.concat([padding, runs]), dim=0, dtype=intermediate_dtype)[
        :-1
    ]

    # Indexing magics
    run_indices = th.repeat_interleave(th.arange(run_length), runs)
    cumulative_starts = th.concat(
        [padding, th.cumsum(runs, axis=0, dtype=intermediate_dtype)[:-1]]
    )
    offsets = (
        th.arange(arr_length, dtype=intermediate_dtype) - cumulative_starts[run_indices]
    )
    del cumulative_starts, run_indices
    indices = th.repeat_interleave(start_pos, runs) + offsets

    out = th.empty(arr_length, dtype=th.int)
    out[indices.to(dtype=th.int)] = th.repeat_interleave(elements, runs)
    return out


COMPRESSION_METHOD_MAP: dict[str, CompressionMethods] = {
    "rle": CompressionMethods(compress=rle_compress, decompress=rle_decompress),
}

logger.info(f"Loaded GPU compression methods:\n{', '.join(COMPRESSION_METHOD_MAP)}")
