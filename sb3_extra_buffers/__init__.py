"""Extra buffer classes for Stable-Baselines3 with compression."""

from functools import lru_cache
from typing import Tuple

from stable_baselines3 import __version__ as sb3_ver

from sb3_extra_buffers.current_version import package_version
from sb3_extra_buffers.types import NumberType, ReplayLike


@lru_cache(maxsize=1)
def sb3_version() -> Tuple[int, int, int]:
    """Parse SB3 version and return (major, minor, patch) as integers.

    Parses version strings like "2.7.1", "2.7.1a0", "2.7.1rc1" into (major, minor, patch).
    For pre-release versions, only the numeric part of the patch is extracted.
    """
    parts = sb3_ver.split(".")
    # Take first three components
    version_parts = parts[:3]
    # For the patch version, split by non-numeric characters and take first part
    if len(version_parts) >= 3:
        patch_parts = "".join(c if c.isdigit() else " " for c in version_parts[2]).split()
        version_parts[2] = patch_parts[0] if patch_parts else "0"
    # Pad with zeros if needed
    while len(version_parts) < 3:
        version_parts.append("0")
    # Convert to integers, defaulting to 0 for non-numeric parts
    result = []
    for p in version_parts:
        try:
            result.append(int(p))
        except ValueError:
            result.append(0)
    return tuple(result)


__version__ = package_version
__all__ = ["__version__", "NumberType", "ReplayLike"]
