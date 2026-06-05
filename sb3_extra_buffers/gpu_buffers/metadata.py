"""Shared types for GPU buffer compression."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SlotMetadata:
    """Compression metadata for one observation cell in the raw heap."""

    byte_start: int
    pos_runs: int
    pos_elem: int
    run_length: int
    payload_bytes: int


def arr_config_length(arr_configs: dict) -> int:
    """Return flattened observation length from ``size`` or ``shape`` keys."""
    if "size" in arr_configs:
        return int(arr_configs["size"])
    return int(arr_configs["shape"])
