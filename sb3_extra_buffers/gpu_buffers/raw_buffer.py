"""Untyped GPU/CPU byte storage with slot-oriented helpers."""

from typing import Union

import torch as th


class RawBuffer:
    """Linear byte storage backed by ``torch.UntypedStorage``."""

    def __init__(self, size: int, device: Union[str, th.device] = "cpu"):
        """Allocate ``size`` bytes on ``device``.

        Args:
            size: Total storage size in bytes.
            device: Torch device for the underlying storage.
        """
        self.buffer = th.UntypedStorage(size, device=device)
        self.device: th.device = th.device(device)
        self.size: int = size

    def write_bytes(self, malloc: tuple[int, int], tensor: th.Tensor):
        """Copy tensor bytes into the region described by ``malloc``."""
        dstart, dlength = malloc
        th.tensor([], dtype=tensor.dtype, device=self.device).set_(self.buffer, dstart, (dlength,))[:] = tensor

    def read_bytes(self, malloc: tuple[int, int], dtype: th.dtype):
        """View ``malloc`` bytes as a 1D tensor with ``dtype``."""
        dstart, dlength = malloc
        return th.tensor([], dtype=dtype, device=self.device).set_(self.buffer, dstart, (dlength,))

    def read_into(self, malloc: tuple[int, int], tensor: th.Tensor):
        """Copy bytes from ``malloc`` into the start of ``tensor``."""
        dstart, dlength = malloc
        tensor[:dlength] = th.tensor([], dtype=tensor.dtype, device=self.device).set_(self.buffer, dstart, (dlength,))


class SlotArena:
    """Fixed-size slot arena backed by a dense ``uint8`` tensor."""

    def __init__(
        self,
        n_slots: int,
        max_slot_bytes: int,
        device: Union[str, th.device] = "cpu",
    ):
        """Create ``n_slots`` contiguous regions of ``max_slot_bytes`` each."""
        self.n_slots = n_slots
        self.max_slot_bytes = max_slot_bytes
        self.device = th.device(device)
        self.slots = th.zeros((n_slots, max_slot_bytes), dtype=th.uint8, device=self.device)

    def slot_base(self, slot_id: int) -> int:
        """Return the byte offset for ``slot_id`` (always ``slot_id * max_slot_bytes``)."""
        return slot_id * self.max_slot_bytes

    def write_at(self, slot_id: int, rel_byte_offset: int, tensor: th.Tensor):
        """Write ``tensor`` into a slot at a byte offset."""
        data = tensor.contiguous().to(self.device)
        as_bytes = data.view(th.uint8).reshape(-1)
        end = rel_byte_offset + as_bytes.numel()
        if end > self.max_slot_bytes:
            raise ValueError(f"Slot {slot_id} write exceeds max_slot_bytes ({end} > {self.max_slot_bytes})")
        self.slots[slot_id, rel_byte_offset:end] = as_bytes

    def read_at(self, slot_id: int, rel_byte_offset: int, elem_count: int, dtype: th.dtype) -> th.Tensor:
        """Read ``elem_count`` elements from a slot at a byte offset."""
        byte_len = elem_count * th.tensor([], dtype=dtype).element_size()
        end = rel_byte_offset + byte_len
        as_bytes = self.slots[slot_id, rel_byte_offset:end]
        return as_bytes.view(dtype).reshape(elem_count).clone()
