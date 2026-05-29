"""GPU compression method registry."""

from sb3_extra_buffers.gpu_buffers.compression_methods.compression_methods import (
    COMPRESSION_METHOD_MAP,
    GpuCompressionMethods,
    has_zstd,
)

__all__ = ["COMPRESSION_METHOD_MAP", "GpuCompressionMethods", "has_zstd"]
