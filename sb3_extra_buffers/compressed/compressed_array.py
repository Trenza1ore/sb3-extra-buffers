from typing import Optional, Union, Any, Literal, Callable

import re
from functools import partial
import numpy as np
from sb3_extra_buffers.compressed.compression_methods import COMPRESSION_METHOD_MAP
from sb3_extra_buffers.compressed.utils import find_smallest_dtype


class CompressedArray(np.ndarray):
    def __init__(
        self,
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
        super().__init__(shape=shape, dtype=object, buffer=buffer, offset=offset, strides=strides, order=order)
        self.flatten_len = np.prod(obs_shape)
        self.flatten_config = dict(shape=self.flatten_len, dtype=np.float32)
        self.__dtype = dtype

        # Handle dtypes
        self.dtypes = dtypes or dict(elem_type=dtype, runs_type=find_smallest_dtype(self.flatten_len))

        # Compress and decompress
        self.compression_kwargs = compression_kwargs or self.dtypes
        self.decompression_kwargs = decompression_kwargs or self.dtypes
        if compression_method[-1].isdigit():
            re_match = re.search(r"(\w+?)([0-9]+)", compression_method)
            assert re_match, f"Invalid compression shorthand: {compression_method}"
            compression_method = re_match.group(1)
            compression_kwargs["compresslevel"] = int(re_match.group(2))
        self._compress = partial(COMPRESSION_METHOD_MAP[compression_method].compress, **self.compression_kwargs)
        self._decompress = partial(COMPRESSION_METHOD_MAP[compression_method].decompress,
                                   arr_configs=self.flatten_config, **self.decompression_kwargs)

    def reconstruct_obs(self, data: bytes):
        obs = self._decompress(data).reshape(self.obs_shape)
        return obs.astype(self.__dtype, copy=False)

    def __setitem__(self, index, value):
        arr = np.asarray(value)
        super().__setitem__(index, self._compress(arr))

    def __getitem__(self, index):
        retrieved = super().__getitem__(index)
        if isinstance(retrieved, np.ndarray):
            return [self.reconstruct_obs(x) for x in retrieved]
        else:
            return self.reconstruct_obs(retrieved)
