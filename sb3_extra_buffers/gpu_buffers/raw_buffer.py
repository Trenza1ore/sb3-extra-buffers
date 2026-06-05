"""Untyped GPU/CPU byte storage for packed observation heaps."""

from typing import Optional, Union

import torch as th


def write_at(buffer: "RawBuffer", byte_start: int, rel_byte_offset: int, tensor: th.Tensor) -> None:
    """Write ``tensor`` bytes at ``byte_start + rel_byte_offset``."""
    data = tensor.contiguous().to(buffer.device)
    blob = data.view(th.uint8).reshape(-1)
    offset = byte_start + rel_byte_offset
    end = offset + blob.numel()
    if end > buffer.size:
        raise ValueError(f"Write exceeds heap capacity ({end} > {buffer.size})")
    buffer.write_bytes((offset, blob.numel()), blob)


def read_at(
    buffer: "RawBuffer",
    byte_start: int,
    rel_byte_offset: int,
    elem_count: int,
    dtype: th.dtype,
) -> th.Tensor:
    """Read ``elem_count`` elements at ``byte_start + rel_byte_offset``."""
    byte_len = elem_count * th.tensor([], dtype=dtype).element_size()
    offset = byte_start + rel_byte_offset
    raw = buffer.read_bytes((offset, byte_len), th.uint8)
    return raw.clone().view(dtype).reshape(elem_count)


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

    def copy_region(self, src_start: int, length: int, dst_start: int) -> None:
        """Copy ``length`` bytes from ``src_start`` to ``dst_start``."""
        if length == 0:
            return
        chunk = self.read_bytes((src_start, length), th.uint8)
        self.write_bytes((dst_start, length), chunk)


class SharedRawHeap:
    """Shared byte heap with per-cell ``start_idx`` and ``lengths``."""

    def __init__(self, n_cells: int, heap_bytes: int, device: Union[str, th.device] = "cpu"):
        """Allocate heap storage and per-cell index arrays."""
        self.n_cells = n_cells
        self.device = th.device(device)
        self.buffer = RawBuffer(heap_bytes, device=self.device)
        self.start_idx = th.zeros(n_cells, dtype=th.int64, device=self.device)
        self.lengths = th.zeros(n_cells, dtype=th.int64, device=self.device)
        self.data_end = 0
        self._scratch: Optional[RawBuffer] = None

    def _scratch_buffer(self, size: int) -> RawBuffer:
        if self._scratch is None or self._scratch.size < size:
            self._scratch = RawBuffer(size, device=self.device)
        return self._scratch

    def compact(self) -> None:
        """Pack all cell payloads into a contiguous prefix of the heap."""
        lengths = self.lengths.cpu().tolist()
        old_starts = self.start_idx.cpu().tolist()
        new_starts = []
        cursor = 0
        for length in lengths:
            new_starts.append(cursor)
            cursor += length
        packed_end = cursor
        if packed_end == 0:
            self.data_end = 0
            return

        scratch = self._scratch_buffer(packed_end)
        for old_start, length, dst in zip(old_starts, lengths, new_starts):
            if length == 0:
                continue
            chunk = self.buffer.read_bytes((old_start, length), th.uint8)
            scratch.write_bytes((dst, length), chunk)

        for dst, length in zip(new_starts, lengths):
            if length == 0:
                continue
            chunk = scratch.read_bytes((dst, length), th.uint8)
            self.buffer.write_bytes((dst, length), chunk)

        self.start_idx = th.tensor(new_starts, dtype=th.int64, device=self.device)
        self.data_end = packed_end

    def ensure_space(self, needed: int) -> None:
        """Compact until at least ``needed`` bytes are free at ``data_end``."""
        if self.data_end + needed <= self.buffer.size:
            return
        self.compact()
        if self.data_end + needed > self.buffer.size:
            raise ValueError(f"Heap needs {needed} bytes at offset {self.data_end} but capacity is {self.buffer.size}")
