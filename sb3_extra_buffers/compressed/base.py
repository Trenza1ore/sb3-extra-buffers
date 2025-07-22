from typing import Union, Optional, Any

import re
import warnings
from functools import partial
import numpy as np
from sb3_extra_buffers import __version__
from sb3_extra_buffers.compressed.compression_methods import COMPRESSION_METHOD_MAP, has_numba, has_igzip
from sb3_extra_buffers.compressed.utils import find_smallest_dtype

if has_numba():
    from sb3_extra_buffers.compressed.compression_methods.compression_methods_numba import init_jit
else:
    def init_jit(*args, **kwargs):
        raise ModuleNotFoundError("Numba library doesn't seem to be installed, try installing via:\n"
                                  "pip install \"sb3-extra-buffers[numba]\"")


def find_buffer_dtypes(obs_shape: Union[int, tuple], elem_dtype: Union[np.integer, np.floating] = np.uint8,
                       compression_method: str = "rle") -> dict[str, Any]:
    """Find the best data types to use for CompressedBuffer based on obs shape and compression method"""
    if isinstance(obs_shape, tuple):
        obs_shape = np.prod(obs_shape)
    buffer_dtypes = dict(elem_type=elem_dtype, runs_type=find_smallest_dtype(obs_shape))
    if compression_method.endswith("-jit"):
        init_jit(**buffer_dtypes)
    return buffer_dtypes


class BaseCompressedBuffer:
    """Base Compressed Buffer Class"""
    def __init__(self, compression_method: Optional[str] = None, compression_kwargs: Optional[dict] = None,
                 decompression_kwargs: Optional[dict] = None, flatten_config: Optional[dict] = None):
        self.version = __version__
        if compression_method is None:
            return
        if compression_method[-1].isdigit():
            re_match = re.search(r"^((?:[A-Za-z]+)|(?:[\w\-]+/))(\-?[0-9]+)$", compression_method)
            assert re_match, f"Invalid compression shorthand: {compression_method}"
            compression_method = re_match.group(1).removesuffix("/")
            compression_kwargs["compresslevel"] = int(re_match.group(2))
        # Warn user about optional dependncies missing
        if compression_method == "igzip" and not has_igzip():
            warnings.warn("Failed to initialize igzip, falls back to gzip backend. "
                          "If you want to use igzip, consider installing the python-isal library via:\n"
                          "pip install \"sb3-extra-buffers[isal]\"", category=ImportWarning)
            compression_method = "gzip"
        if compression_method.endswith("-jit") and not has_numba():
            warnings.warn("Failed to initialize Numba for jit compiler, falls back to NumPy backend. "
                          "If you want to use jit version, consider installing the Numba library via:\n"
                          "pip install \"sb3-extra-buffers[numba]\"", category=ImportWarning)
            compression_method = compression_method.removesuffix("-jit")
        # Get the actual compression methods
        assert compression_method in COMPRESSION_METHOD_MAP, f"Unknown compression method {compression_method}"
        self._compress = partial(COMPRESSION_METHOD_MAP[compression_method].compress, **compression_kwargs)
        self._decompress = partial(COMPRESSION_METHOD_MAP[compression_method].decompress,
                                   arr_configs=flatten_config, **decompression_kwargs)


class DummyCls:
    def __init__(*args, **kwargs):
        pass
