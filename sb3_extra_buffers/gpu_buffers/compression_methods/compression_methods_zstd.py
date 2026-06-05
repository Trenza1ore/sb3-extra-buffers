"""GPU heap storage with CPU Zstandard compression."""

# pylint: disable=too-many-positional-arguments, unused-argument, c-extension-no-member

import numpy as np
import torch as th
import zstd

from sb3_extra_buffers.gpu_buffers.metadata import SlotMetadata, arr_config_length
from sb3_extra_buffers.gpu_buffers.raw_buffer import RawBuffer, read_at, write_at
from sb3_extra_buffers.gpu_buffers.utils import torch_dtype_to_numpy


def zstd_compress(
    arr: th.Tensor,
    buffer: RawBuffer,
    byte_start: int,
    elem_type: th.dtype = th.uint8,
    runs_type: th.dtype = th.uint16,
    compresslevel: int = 0,
    threads: int = 0,
    **kwargs,
) -> SlotMetadata:
    """Compress a flat tensor with Zstandard and store bytes in the heap."""
    del runs_type, kwargs
    flat = arr.to(dtype=elem_type).contiguous().cpu().numpy()
    compressed = zstd.compress(flat, compresslevel, threads)
    compressed_bytes = len(compressed)
    end = byte_start + compressed_bytes
    if end > buffer.size:
        raise ValueError(f"Zstd output ({compressed_bytes} bytes) exceeds heap capacity ({buffer.size} bytes)")
    payload = th.from_numpy(np.frombuffer(compressed, dtype=np.uint8).copy()).to(buffer.device)
    write_at(buffer, byte_start, 0, payload)
    return SlotMetadata(
        byte_start=byte_start,
        pos_runs=0,
        pos_elem=0,
        run_length=compressed_bytes,
        payload_bytes=compressed_bytes,
    )


def zstd_decompress(
    buffer: RawBuffer,
    meta: SlotMetadata,
    elem_type: th.dtype,
    runs_type: th.dtype,
    arr_configs: dict,
    **kwargs,
) -> th.Tensor:
    """Decompress Zstandard bytes from the heap back to a flat tensor."""
    del runs_type, kwargs
    output_dtype = arr_configs.get("dtype", elem_type)
    np_dtype = torch_dtype_to_numpy(output_dtype)
    compressed = read_at(buffer, meta.byte_start, 0, meta.run_length, th.uint8).cpu().numpy().tobytes()
    flat = np.frombuffer(zstd.decompress(compressed), dtype=np_dtype).copy()
    arr_length = arr_config_length(arr_configs)
    if flat.size != arr_length:
        raise ValueError(f"Decompressed length {flat.size} != expected {arr_length}")
    return th.from_numpy(flat).to(device=buffer.device, dtype=output_dtype)
