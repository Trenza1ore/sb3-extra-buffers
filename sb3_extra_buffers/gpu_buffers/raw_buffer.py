from typing import Union
from collections import deque
import torch as th

class RawBuffer:
    def __init__(self, size: int, device: Union[str, th.device] = "cpu"):
        self.buffer = th.UntypedStorage(size, device=device)
        self.allocations: deque[tuple[int, int]] = deque()
        self.device: th.device = th.device(device)
        self.size: int = size
        self.ptr: int = 0

    def malloc(self, length: int) -> tuple[int, int]:
        if self.ptr + length > self.size:
            end = 0
            while end < length:
                _, end = self.allocations.popleft()
            self.ptr = end
            run = (0, length)
            self.allocations.append(run)
            return run
        ptr = self.ptr
        self.ptr += length
        run = (ptr, length)
        self.allocations.append(run)
        return run

    def write_bytes(self, malloc: tuple[int, int], tensor: th.Tensor):
        dstart, dlength = malloc
        th.tensor([], dtype=tensor.dtype, device=self.device).set_(self.buffer, dstart, (dlength,))[:] = tensor

    def read_bytes(self, malloc: tuple[int, int], dtype: th.dtype):
        dstart, dlength = malloc
        return th.tensor([], dtype=dtype, device=self.device).set_(self.buffer, dstart, (dlength,))

    def read_into(self, malloc: tuple[int, int], tensor: th.Tensor):
        dstart, dlength = malloc
        tensor[:dlength] = th.tensor([], dtype=tensor.dtype, device=self.device).set_(self.buffer, dstart, (dlength,))
