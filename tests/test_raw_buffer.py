import numpy as np
import pytest
import torch as th

from sb3_extra_buffers.compressed.compression_methods.compression_methods import \
    rle_numpy_decompress_old
from sb3_extra_buffers.gpu_buffers.compression_methods.compression_methods import (
    rle_compress, rle_decompress)
from sb3_extra_buffers.gpu_buffers.raw_buffer import RawBuffer


@pytest.mark.parametrize(
    "input_arr",
    [
        th.tensor([1, 1, 2, 2, 2, 3, 3, 1], dtype=th.uint8),
        th.tensor([5] * 50 + [3] * 20 + [1] * 5, dtype=th.uint8),
        th.randint(0, 3, (100,), dtype=th.uint8),
    ],
)
def test_rle_compression_roundtrip(input_arr):
    buffer = RawBuffer(1000)
    elem_type = th.int32
    runs_type = th.int32

    # Compress
    pos_runs, pos_elem, run_length = rle_compress(
        input_arr, buffer, elem_type, runs_type
    )
    th_decomp = rle_decompress(
        buffer,
        pos_runs,
        pos_elem,
        run_length,
        elem_type,
        runs_type,
        dict(size=(input_arr.size(0)), dtype=elem_type),
    )
    runs = (
        buffer.read_bytes((pos_runs, run_length), runs_type)
        .cpu()
        .numpy()
        .copy()
        .tobytes()
    )
    elem = (
        buffer.read_bytes((pos_elem, run_length), elem_type)
        .cpu()
        .numpy()
        .copy()
        .tobytes()
    )

    # Decompress
    out = rle_numpy_decompress_old(
        data=runs + elem,
        elem_type=np.uint8,
        runs_type=np.uint16,
        arr_configs={"shape": len(input_arr), "dtype": np.uint8},
    )

    # Check correctness
    th.testing.assert_close(th_decomp, input_arr.to(dtype=th.int))
