"""Torch-based compression into GPU slot arenas."""

# pylint: disable=too-many-positional-arguments

from collections import namedtuple

import torch as th

from sb3_extra_buffers.gpu_buffers.metadata import SlotMetadata, arr_config_length
from sb3_extra_buffers.gpu_buffers.raw_buffer import SlotArena
from sb3_extra_buffers.logging import logger

HAS_ZSTD = False

GpuCompressionMethods = namedtuple("GpuCompressionMethod", ["compress", "decompress"])


def rle_compress(
    arr: th.Tensor,
    arena: SlotArena,
    slot_id: int,
    elem_type: th.dtype = th.uint8,
    runs_type: th.dtype = th.uint16,
    **kwargs,
) -> SlotMetadata:
    """Run-length encode a 1D tensor into a slot in ``arena``."""
    n = arr.size(0)

    change_idxs = th.where(arr[1:] != arr[:-1])[0]
    idx_arr = th.cat([change_idxs, th.tensor([n - 1], dtype=th.long, device=arr.device)])

    prev_idx = th.cat([th.tensor([-1], dtype=th.long, device=arr.device), idx_arr[:-1]])
    runs = (idx_arr - prev_idx).to(dtype=runs_type)

    values = arr[idx_arr].to(dtype=elem_type)
    run_length = runs.size(0)
    len_runs = run_length * runs_type.itemsize

    arena.write_at(slot_id, 0, runs)
    arena.write_at(slot_id, len_runs, values)

    return SlotMetadata(slot_id=slot_id, pos_runs=0, pos_elem=len_runs, run_length=run_length)


def rle_decompress(
    arena: SlotArena,
    meta: SlotMetadata,
    elem_type: th.dtype,
    runs_type: th.dtype,
    arr_configs: dict,
    **kwargs,
) -> th.Tensor:
    """RLE decompression, PyTorch version."""
    arr_length = arr_config_length(arr_configs)
    output_dtype = arr_configs.get("dtype", elem_type)
    intermediate_dtype = th.int64
    padding = th.zeros(1, dtype=intermediate_dtype, device=arena.device)

    slot_id = meta.slot_id
    run_length = meta.run_length
    runs = arena.read_at(slot_id, meta.pos_runs, run_length, runs_type).to(intermediate_dtype)
    elements = arena.read_at(slot_id, meta.pos_elem, run_length, elem_type)
    start_pos = th.cumsum(th.concat([padding, runs]), dim=0, dtype=intermediate_dtype)[:-1]

    run_indices = th.repeat_interleave(
        th.arange(run_length, device=arena.device, dtype=intermediate_dtype),
        runs,
    )
    cumulative_starts = th.concat([padding, th.cumsum(runs, axis=0, dtype=intermediate_dtype)[:-1]])
    offsets = th.arange(arr_length, dtype=intermediate_dtype, device=arena.device) - cumulative_starts[run_indices]
    del cumulative_starts, run_indices
    indices = th.repeat_interleave(start_pos, runs) + offsets

    out = th.empty(arr_length, dtype=output_dtype, device=arena.device)
    out[indices] = th.repeat_interleave(elements.to(intermediate_dtype), runs).to(output_dtype)
    return out


def none_compress(
    arr: th.Tensor,
    arena: SlotArena,
    slot_id: int,
    elem_type: th.dtype = th.uint8,
) -> SlotMetadata:
    """Store a flat tensor as raw bytes inside a slot."""
    flat = arr.to(dtype=elem_type).contiguous()
    arena.write_at(slot_id, 0, flat)
    return SlotMetadata(slot_id=slot_id, pos_runs=0, pos_elem=0, run_length=flat.numel())


def none_decompress(
    arena: SlotArena,
    meta: SlotMetadata,
    elem_type: th.dtype,
    arr_configs: dict,
) -> th.Tensor:
    """Read a flat tensor from raw slot bytes."""
    arr_length = arr_config_length(arr_configs)
    output_dtype = arr_configs.get("dtype", elem_type)
    return arena.read_at(meta.slot_id, 0, arr_length, output_dtype)


COMPRESSION_METHOD_MAP: dict[str, GpuCompressionMethods] = {
    "none": GpuCompressionMethods(compress=none_compress, decompress=none_decompress),
    "rle": GpuCompressionMethods(compress=rle_compress, decompress=rle_decompress),
}

try:
    from sb3_extra_buffers.gpu_buffers.compression_methods.compression_methods_zstd import (
        zstd_compress,
        zstd_decompress,
    )

    HAS_ZSTD = True
    COMPRESSION_METHOD_MAP["zstd"] = GpuCompressionMethods(compress=zstd_compress, decompress=zstd_decompress)
except ImportError:
    logger.warning("GPU compression extension not installed: zstd")


def has_zstd() -> bool:
    """Return whether the Zstandard backend is available."""
    return HAS_ZSTD


logger.info("Loaded GPU compression methods:\n%s", ", ".join(COMPRESSION_METHOD_MAP))
