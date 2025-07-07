from sb3_extra_buffers.recording.base import BaseRecordBuffer, DummyArray
from sb3_extra_buffers.recording.record_buffer import RecordBuffer


class DummyRecordBuffer(BaseRecordBuffer):
    """A dummy RecordBuffer that does not record anything"""
    def __init__(self) -> None:
        self._ptr = -1
        self.frames = self.features = self.rewards = self.actions = DummyArray()

    def add(self, *args, **kwargs) -> None:
        self._ptr += 1


class FramelessRecordBuffer(BaseRecordBuffer):
    """A dummy RecordBuffer that does not record actual game frames"""
    def __init__(self, *args, **kwargs) -> None:
        for fixed_value in ["res", "ch_num"]:
            if fixed_value in kwargs:
                del kwargs[fixed_value]
        mem = RecordBuffer(res=(1, 1), ch_num=1, **kwargs)
        mem.frames = DummyArray()
        self.dummy_frame = DummyArray()
        self._memory = mem
        self._ptr = self._memory._ptr
        self.features, self.rewards, self.actions = mem.features, mem.rewards, mem.actions

    def add(self, _, *args, **kwargs) -> None:
        self._memory.add(self.dummy_frame, *args, **kwargs)
        self._ptr = self._memory._ptr
