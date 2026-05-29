"""Device-resident observation storage backends."""

# pylint: disable=too-few-public-methods

from typing import Callable, Optional, Union

import torch as th

from sb3_extra_buffers.gpu_buffers.metadata import SlotMetadata
from sb3_extra_buffers.gpu_buffers.raw_buffer import SlotArena
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


class SlotArenaObservationStore:
    """Store compressed observations in a fixed slot arena."""

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
        max_slot_bytes: Optional[int] = None,
        compression_method: str = "rle",
        runs_type: th.dtype = th.uint16,
        arena: Optional[SlotArena] = None,
    ):
        """Create slot-backed observation storage."""
        self.buffer_size = buffer_size
        self.n_envs = n_envs
        self.flat_len = flat_len
        self.elem_type = elem_type
        self.device = th.device(device)
        self.field_offset = field_offset
        self._compress = compress
        self._decompress = decompress
        if max_slot_bytes is None:
            max_slot_bytes = estimate_max_slot_bytes(flat_len, elem_type, runs_type, compression_method)
        self.max_slot_bytes = max_slot_bytes
        self._flattened = False

        if arena is None:
            n_fields = 1 if field_offset == 0 else 2
            n_slots = buffer_size * n_envs * n_fields
            self.arena = SlotArena(n_slots, self.max_slot_bytes, device=self.device)
        else:
            self.arena = arena

        self.metadata = th.zeros((buffer_size, n_envs, 3), dtype=th.int64, device=self.device)

    def slot_id(self, pos: int, env_idx: int) -> int:
        """Map buffer coordinates to a global slot id."""
        return self.field_offset * (self.buffer_size * self.n_envs) + pos * self.n_envs + env_idx

    def write(self, pos: int, env_idx: int, flat_tensor: th.Tensor):
        """Compress and store a flattened observation."""
        slot_id = self.slot_id(pos, env_idx)
        meta = self._compress(flat_tensor, self.arena, slot_id)
        self.metadata[pos, env_idx, 0] = meta.pos_runs
        self.metadata[pos, env_idx, 1] = meta.pos_elem
        self.metadata[pos, env_idx, 2] = meta.run_length

    def read(self, pos: int, env_idx: int) -> th.Tensor:
        """Decompress a flattened observation."""
        if self._flattened:
            flat_idx = _flat_index(self.buffer_size, pos, env_idx)
            return self.read_flat(flat_idx)
        return self._read_meta(self.metadata[pos, env_idx], self.slot_id(pos, env_idx))

    def read_flat(self, flat_idx: int) -> th.Tensor:
        """Read by flattened index after :meth:`flatten`."""
        pos = flat_idx % self.buffer_size
        env_idx = flat_idx // self.buffer_size
        slot_id = self.slot_id(pos, env_idx)
        return self._read_meta(self.metadata.view(-1, 3)[flat_idx], slot_id)

    def _read_meta(self, meta_row: th.Tensor, slot_id: int) -> th.Tensor:
        meta = SlotMetadata(
            slot_id=slot_id,
            pos_runs=int(meta_row[0].item()),
            pos_elem=int(meta_row[1].item()),
            run_length=int(meta_row[2].item()),
        )
        return self._decompress(self.arena, meta)

    def flatten(self):
        """Flatten metadata to match rollout ``swap_and_flatten`` ordering."""
        if not self._flattened:
            self.metadata = self.metadata.swapaxes(0, 1).reshape(self.buffer_size * self.n_envs, 3)
            self._flattened = True


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
    max_slot_bytes: Optional[int] = None,
    runs_type: th.dtype = th.uint16,
    shared_arena: Optional[SlotArena] = None,
):
    """Create the observation store backend for ``compression_method``."""
    if compression_method == "none":
        return DenseObservationStore(buffer_size, n_envs, flat_len, elem_type, device)
    if compress is None or decompress is None:
        raise ValueError("Compressed stores require compress and decompress callables.")
    return SlotArenaObservationStore(
        buffer_size,
        n_envs,
        flat_len,
        elem_type,
        device,
        field_offset,
        compress,
        decompress,
        max_slot_bytes=max_slot_bytes,
        compression_method=compression_method,
        runs_type=runs_type,
        arena=shared_arena,
    )
