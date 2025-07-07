from sb3_extra_buffers.recording.base import BaseRecordBuffer


def force_alloc_mem(mem: BaseRecordBuffer, val: object = 0) -> None:
    """Force allocate memory, avoids the illusion of sufficient memory"""
    mem.frames.fill(val)
    mem.features.fill(val)
    mem.rewards.fill(val)
    mem.actions.fill(val)
