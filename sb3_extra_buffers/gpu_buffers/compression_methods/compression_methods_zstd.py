"""GPU slot storage with CPU Zstandard compression."""

# pylint: disable=too-many-positional-arguments, unused-argument, c-extension-no-member

import numpy as np
import torch as th
import zstd

from sb3_extra_buffers.gpu_buffers.metadata import SlotMetadata, arr_config_length
from sb3_extra_buffers.gpu_buffers.raw_buffer import SlotArena
from sb3_extra_buffers.gpu_buffers.utils import torch_dtype_to_numpy


def zstd_compress(
    arr: th.Tensor,
    arena: SlotArena,
    slot_id: int,
    elem_type: th.dtype = th.uint8,
    runs_type: th.dtype = th.uint16,
    compresslevel: int = 0,
    threads: int = 0,
    **kwargs,
) -> SlotMetadata:
    """Compress a flat tensor with Zstandard and store bytes in a slot."""
    del runs_type, kwargs
    flat = arr.to(dtype=elem_type).contiguous().cpu().numpy()
    compressed = zstd.compress(flat, compresslevel, threads)
    compressed_bytes = len(compressed)
    if compressed_bytes > arena.max_slot_bytes:
        raise ValueError(f"Zstd output ({compressed_bytes} bytes) exceeds slot capacity ({arena.max_slot_bytes} bytes)")
    payload = th.from_numpy(np.frombuffer(compressed, dtype=np.uint8).copy()).to(arena.device)
    arena.write_at(slot_id, 0, payload)
    return SlotMetadata(slot_id=slot_id, pos_runs=0, pos_elem=0, run_length=compressed_bytes)


def zstd_decompress(
    arena: SlotArena,
    meta: SlotMetadata,
    elem_type: th.dtype,
    runs_type: th.dtype,
    arr_configs: dict,
    **kwargs,
) -> th.Tensor:
    """Decompress Zstandard bytes from a slot back to a flat tensor."""
    output_dtype = arr_configs.get("dtype", elem_type)
    np_dtype = torch_dtype_to_numpy(output_dtype)
    compressed = arena.read_at(meta.slot_id, 0, meta.run_length, th.uint8).cpu().numpy().tobytes()
    flat = np.frombuffer(zstd.decompress(compressed), dtype=np_dtype).copy()
    arr_length = arr_config_length(arr_configs)
    if flat.size != arr_length:
        raise ValueError(f"Decompressed length {flat.size} != expected {arr_length}")
    return th.from_numpy(flat).to(device=arena.device, dtype=output_dtype)
