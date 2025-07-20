from typing import Optional, Union, Any, Literal, Callable

import re
from functools import partial
import numpy as np
from sb3_extra_buffers.compressed.compression_methods import COMPRESSION_METHOD_MAP
from sb3_extra_buffers.compressed.utils import find_smallest_dtype


class CompressedArray(np.ndarray):
    def __new__(
        cls,
        shape: Union[int, tuple, Any],
        dtype: Union[np.integer, np.floating],
        obs_shape: Union[int, tuple, Any],
        buffer: Optional[Any] = None,
        offset: Any = 0,
        strides: Optional[Any] = None,
        order: Literal[None, "K", "A", "C", "F"] = None,
        dtypes: Optional[dict] = None,
        compression_method: str = "rle",
        compression_kwargs: Optional[dict] = None,
        decompression_kwargs: Optional[dict] = None,
        **kwargs
    ):
        self = super().__new__(cls, shape=shape, dtype=object, buffer=buffer,
                               offset=offset, strides=strides, order=order)
        self.obs_shape = obs_shape
        flatten_len = np.prod(obs_shape)
        flatten_config = dict(shape=flatten_len, dtype=np.float32)

        # Handle dtypes
        self.dtypes = dtypes or dict(elem_type=dtype, runs_type=find_smallest_dtype(flatten_len))
        self._dtype = dtype

        # Compress and decompress
        compression_kwargs = compression_kwargs or self.dtypes
        decompression_kwargs = decompression_kwargs or self.dtypes
        if compression_method[-1].isdigit():
            re_match = re.search(r"(\w+?)([0-9]+)", compression_method)
            assert re_match, f"Invalid compression shorthand: {compression_method}"
            compression_method = re_match.group(1)
            compression_kwargs["compresslevel"] = int(re_match.group(2))
        self._compress = partial(COMPRESSION_METHOD_MAP[compression_method].compress, **compression_kwargs)
        self._decompress = partial(COMPRESSION_METHOD_MAP[compression_method].decompress,
                                   arr_configs=flatten_config, **decompression_kwargs)
        self._suppress_get_item = False
        return self

    def __array_finalize__(self, obj):
        if obj is None:
            return
        super().__array_finalize__(obj)
        self._suppress_get_item = False
        for attr in ["obs_shape", "dtypes", "_dtype", "_compress", "_decompress"]:
            setattr(self, attr, getattr(obj, attr))

    def reconstruct_obs(self, data: bytes):
        obs = self._decompress(data).reshape(self.obs_shape)
        return obs.astype(self._dtype, copy=False)

    def __setitem__(self, index, value):
        self._suppress_get_item = True
        arr = np.ravel(np.asarray(value))
        super().__setitem__(index, self._compress(arr))
        self._suppress_get_item = False

    def __getitem__(self, index):
        retrieved = super().__getitem__(index)
        if self._suppress_get_item or retrieved is None:
            return retrieved
        if isinstance(retrieved, np.ndarray):
            return [self.reconstruct_obs(x) for x in retrieved]
        else:
            return self.reconstruct_obs(retrieved)
