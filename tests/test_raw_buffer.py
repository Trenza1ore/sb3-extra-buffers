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
from sb3_extra_buffers.gpu_buffers.metadata import SlotMetadata
from sb3_extra_buffers.gpu_buffers.raw_buffer import RawBuffer, SharedRawHeap, read_at


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
    heap_bytes = input_arr.numel() * elem_type.itemsize + input_arr.numel() * runs_type.itemsize
    buffer = RawBuffer(heap_bytes, device="cpu")

    meta = rle_compress(input_arr, buffer, byte_start=0, elem_type=elem_type, runs_type=runs_type)
    th_decomp = rle_decompress(
        buffer,
        meta,
        elem_type=elem_type,
        runs_type=runs_type,
        arr_configs={"size": input_arr.size(0), "dtype": elem_type},
    )

    runs = read_at(buffer, meta.byte_start, meta.pos_runs, meta.run_length, runs_type).cpu().numpy().copy().tobytes()
    elem = read_at(buffer, meta.byte_start, meta.pos_elem, meta.run_length, elem_type).cpu().numpy().copy().tobytes()

    rle_numpy_decompress_old(
        data=runs + elem,
        elem_type=np.uint8,
        runs_type=np.uint16,
        arr_configs={"shape": len(input_arr), "dtype": np.uint8},
    )

    th.testing.assert_close(th_decomp, input_arr)


def test_shared_raw_heap_compact_after_growth():
    elem_type = th.uint8
    runs_type = th.uint16
    flat_len = 16
    heap_bytes = flat_len * 8
    heap = SharedRawHeap(n_cells=2, heap_bytes=heap_bytes, device="cpu")
    arr_configs = {"size": flat_len, "dtype": elem_type}

    small = th.zeros(flat_len, dtype=elem_type)
    large = th.ones(flat_len, dtype=elem_type)

    meta0 = rle_compress(small, heap.buffer, byte_start=0, elem_type=elem_type, runs_type=runs_type)
    heap.start_idx[0] = 0
    heap.lengths[0] = meta0.payload_bytes
    heap.data_end = meta0.payload_bytes

    meta1 = rle_compress(large, heap.buffer, byte_start=heap.data_end, elem_type=elem_type, runs_type=runs_type)
    heap.start_idx[1] = heap.data_end
    heap.lengths[1] = meta1.payload_bytes
    heap.data_end += meta1.payload_bytes

    heap.start_idx[0] = heap.data_end
    meta0_grown = rle_compress(large, heap.buffer, byte_start=heap.data_end, elem_type=elem_type, runs_type=runs_type)
    heap.lengths[0] = meta0_grown.payload_bytes
    heap.data_end += meta0_grown.payload_bytes

    fragmented_end = heap.data_end
    heap.compact()
    assert heap.data_end < fragmented_end
    assert heap.data_end == int(heap.lengths.sum().item())

    meta0_after = SlotMetadata(
        byte_start=int(heap.start_idx[0].item()),
        pos_runs=meta0_grown.pos_runs,
        pos_elem=meta0_grown.pos_elem,
        run_length=meta0_grown.run_length,
        payload_bytes=int(heap.lengths[0].item()),
    )
    meta1_after = SlotMetadata(
        byte_start=int(heap.start_idx[1].item()),
        pos_runs=meta1.pos_runs,
        pos_elem=meta1.pos_elem,
        run_length=meta1.run_length,
        payload_bytes=int(heap.lengths[1].item()),
    )

    th.testing.assert_close(
        rle_decompress(heap.buffer, meta0_after, elem_type=elem_type, runs_type=runs_type, arr_configs=arr_configs),
        large,
    )
    th.testing.assert_close(
        rle_decompress(heap.buffer, meta1_after, elem_type=elem_type, runs_type=runs_type, arr_configs=arr_configs),
        large,
    )
