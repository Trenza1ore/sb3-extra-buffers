"""Device-resident observation storage backends."""

# pylint: disable=too-few-public-methods

from typing import Callable, Optional, Union

import torch as th

from sb3_extra_buffers.gpu_buffers.metadata import SlotMetadata
from sb3_extra_buffers.gpu_buffers.raw_buffer import SharedRawHeap
from sb3_extra_buffers.gpu_buffers.utils import estimate_max_slot_bytes


def _flat_index(buffer_size: int, pos: int, env_idx: int) -> int:
    """Convert rollout coordinates to SB3 ``swap_and_flatten`` index."""
    return env_idx * buffer_size + pos


class DenseObservationStore:
    """Store flattened observations in a dense Torch tensor."""

    def __init__(
        self,
        buffer_size: int,
        n_envs: int,
        flat_len: int,
        elem_type: th.dtype,
        device: Union[str, th.device],
    ):
        """Allocate dense observation storage on ``device``."""
        self.buffer_size = buffer_size
        self.n_envs = n_envs
        self.flat_len = flat_len
        self.elem_type = elem_type
        self.device = th.device(device)
        self.observations = th.zeros(
            (buffer_size, n_envs, flat_len),
            dtype=elem_type,
            device=self.device,
        )
        self._flattened = False

    def write(self, pos: int, env_idx: int, flat_tensor: th.Tensor):
        """Write a flattened observation."""
        self.observations[pos, env_idx] = flat_tensor.to(dtype=self.elem_type, device=self.device)

    def read(self, pos: int, env_idx: int) -> th.Tensor:
        """Read a flattened observation."""
        if self._flattened:
            flat_idx = _flat_index(self.buffer_size, pos, env_idx)
            return self.observations[flat_idx]
        return self.observations[pos, env_idx]

    def read_flat(self, flat_idx: int) -> th.Tensor:
        """Read by flattened index after :meth:`flatten`."""
        return self.observations[flat_idx]

    def flatten(self):
        """Flatten using the same ordering as SB3 ``swap_and_flatten``."""
        if not self._flattened:
            self.observations = self.observations.swapaxes(0, 1).reshape(
                self.buffer_size * self.n_envs,
                self.flat_len,
            )
            self._flattened = True


class RawObservationStore:
    """Store compressed observations in a packed raw byte heap."""

    def __init__(
        self,
        buffer_size: int,
        n_envs: int,
        flat_len: int,
        elem_type: th.dtype,
        device: Union[str, th.device],
        field_offset: int,
        compress: Callable[..., SlotMetadata],
        decompress: Callable[..., th.Tensor],
        compression_method: str = "rle",
        runs_type: th.dtype = th.uint16,
        shared_heap: Optional[SharedRawHeap] = None,
        n_fields: int = 1,
    ):
        """Create heap-backed observation storage."""
        self.buffer_size = buffer_size
        self.n_envs = n_envs
        self.flat_len = flat_len
        self.elem_type = elem_type
        self.device = th.device(device)
        self.field_offset = field_offset
        self._compress = compress
        self._decompress = decompress
        self._flattened = False
        self._max_cell_bytes = estimate_max_slot_bytes(
            flat_len,
            elem_type,
            runs_type,
            compression_method,
        )

        if shared_heap is None:
            raise ValueError("Compressed stores require a shared SharedRawHeap.")
        self.heap = shared_heap

        self.metadata = th.zeros((buffer_size, n_envs, 3), dtype=th.int64, device=self.device)

    def cell_id(self, pos: int, env_idx: int) -> int:
        """Map buffer coordinates to a global cell id."""
        return self.field_offset * (self.buffer_size * self.n_envs) + pos * self.n_envs + env_idx

    def write(self, pos: int, env_idx: int, flat_tensor: th.Tensor):
        """Compress and store a flattened observation."""
        cell = self.cell_id(pos, env_idx)
        old_len = int(self.heap.lengths[cell].item())
        old_start = int(self.heap.start_idx[cell].item()) if old_len > 0 else None

        scratch = self.heap._scratch_buffer(self._max_cell_bytes)
        meta = self._compress(flat_tensor, scratch, 0)
        payload_bytes = meta.payload_bytes
        if payload_bytes > self._max_cell_bytes:
            raise ValueError(
                f"Compressed payload ({payload_bytes} bytes) exceeds per-cell cap "
                f"({self._max_cell_bytes} bytes); increase heap_bytes or max_slot_bytes"
            )

        if old_len > 0 and payload_bytes <= old_len:
            final_start = old_start
            self.heap.lengths[cell] = payload_bytes
        else:
            self.heap.ensure_space(payload_bytes)
            final_start = self.heap.data_end
            self.heap.start_idx[cell] = final_start
            self.heap.lengths[cell] = payload_bytes
            self.heap.data_end = final_start + payload_bytes

        payload = scratch.read_bytes((0, payload_bytes), th.uint8)
        self.heap.buffer.write_bytes((final_start, payload_bytes), payload)

        self.metadata[pos, env_idx, 0] = meta.pos_runs
        self.metadata[pos, env_idx, 1] = meta.pos_elem
        self.metadata[pos, env_idx, 2] = meta.run_length

    def read(self, pos: int, env_idx: int) -> th.Tensor:
        """Decompress a flattened observation."""
        if self._flattened:
            flat_idx = _flat_index(self.buffer_size, pos, env_idx)
            return self.read_flat(flat_idx)
        return self._read_cell(self.cell_id(pos, env_idx), self.metadata[pos, env_idx])

    def read_flat(self, flat_idx: int) -> th.Tensor:
        """Read by flattened index after :meth:`flatten`."""
        pos = flat_idx % self.buffer_size
        env_idx = flat_idx // self.buffer_size
        cell = self.cell_id(pos, env_idx)
        return self._read_cell(cell, self.metadata.view(-1, 3)[flat_idx])

    def _read_cell(self, cell: int, meta_row: th.Tensor) -> th.Tensor:
        byte_start = int(self.heap.start_idx[cell].item())
        meta = SlotMetadata(
            byte_start=byte_start,
            pos_runs=int(meta_row[0].item()),
            pos_elem=int(meta_row[1].item()),
            run_length=int(meta_row[2].item()),
            payload_bytes=int(self.heap.lengths[cell].item()),
        )
        return self._decompress(self.heap.buffer, meta)

    def flatten(self):
        """Flatten metadata to match rollout ``swap_and_flatten`` ordering."""
        if not self._flattened:
            self.metadata = self.metadata.swapaxes(0, 1).reshape(self.buffer_size * self.n_envs, 3)
            self._flattened = True

    def compact(self) -> None:
        """Pack the shared heap (no-op when this store does not own the heap)."""
        self.heap.compact()


def create_observation_store(
    compression_method: str,
    buffer_size: int,
    n_envs: int,
    flat_len: int,
    elem_type: th.dtype,
    device: Union[str, th.device],
    field_offset: int = 0,
    compress: Optional[Callable[..., SlotMetadata]] = None,
    decompress: Optional[Callable[..., th.Tensor]] = None,
    runs_type: th.dtype = th.uint16,
    shared_heap: Optional[SharedRawHeap] = None,
    n_fields: int = 1,
):
    """Create the observation store backend for ``compression_method``."""
    if compression_method == "none":
        return DenseObservationStore(buffer_size, n_envs, flat_len, elem_type, device)
    if compress is None or decompress is None:
        raise ValueError("Compressed stores require compress and decompress callables.")
    return RawObservationStore(
        buffer_size,
        n_envs,
        flat_len,
        elem_type,
        device,
        field_offset,
        compress,
        decompress,
        compression_method=compression_method,
        runs_type=runs_type,
        shared_heap=shared_heap,
        n_fields=n_fields,
    )
