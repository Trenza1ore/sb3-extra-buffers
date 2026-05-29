import numpy as np
import pytest
import torch as th

from sb3_extra_buffers.compressed.compression_methods.compression_methods import (
    rle_numpy_decompress_old,
)
from sb3_extra_buffers.gpu_buffers.compression_methods.compression_methods import (
    rle_compress,
    rle_decompress,
)
from sb3_extra_buffers.gpu_buffers.raw_buffer import SlotArena


@pytest.mark.parametrize(
    "input_arr",
    [
        th.tensor([1, 1, 2, 2, 2, 3, 3, 1], dtype=th.uint8),
        th.tensor([5] * 50 + [3] * 20 + [1] * 5, dtype=th.uint8),
        th.randint(0, 3, (100,), dtype=th.uint8),
    ],
)
def test_rle_compression_roundtrip(input_arr):
    elem_type = th.uint8
    runs_type = th.uint16
    slot_bytes = input_arr.numel() * elem_type.itemsize + input_arr.numel() * runs_type.itemsize
    arena = SlotArena(n_slots=1, max_slot_bytes=slot_bytes)

    meta = rle_compress(input_arr, arena, slot_id=0, elem_type=elem_type, runs_type=runs_type)
    th_decomp = rle_decompress(
        arena,
        meta,
        elem_type=elem_type,
        runs_type=runs_type,
        arr_configs={"size": input_arr.size(0), "dtype": elem_type},
    )

    runs = arena.read_at(0, meta.pos_runs, meta.run_length, runs_type).cpu().numpy().copy().tobytes()
    elem = arena.read_at(0, meta.pos_elem, meta.run_length, elem_type).cpu().numpy().copy().tobytes()

    rle_numpy_decompress_old(
        data=runs + elem,
        elem_type=np.uint8,
        runs_type=np.uint16,
        arr_configs={"shape": len(input_arr), "dtype": np.uint8},
    )

    th.testing.assert_close(th_decomp, input_arr)
