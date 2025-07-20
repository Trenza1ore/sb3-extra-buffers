from typing import Optional, Union, Any, Literal

from threading import Lock
import numpy as np
from sb3_extra_buffers.compressed.base import BaseCompressedBuffer
from sb3_extra_buffers.compressed.utils import find_smallest_dtype


class CompressedArray(np.ndarray, BaseCompressedBuffer):
    """Experimental Compressed Array Class"""
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
        self.obs_shape = obs_shape
        flatten_len = np.prod(obs_shape)
        self.flatten_config = dict(shape=flatten_len, dtype=dtype)

        # Handle dtypes
        self.dtypes = dtypes or dict(elem_type=dtype, runs_type=find_smallest_dtype(flatten_len))
        self._dtype = dtype

        # Compress and decompress
        self.compression_kwargs = compression_kwargs or self.dtypes
        self.decompression_kwargs = decompression_kwargs or self.dtypes
        BaseCompressedBuffer.__init__(self, compression_method=compression_method,
                                      compression_kwargs=self.compression_kwargs,
                                      decompression_kwargs=self.decompression_kwargs,
                                      flatten_config=self.flatten_config)
        self._suppress_get_item = False
        self._thread_lock = Lock()

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
        return self

    def __array_finalize__(self, obj):
        if obj is None:
            return
        super().__array_finalize__(obj)
        self._suppress_get_item = False
        self._thread_lock = Lock()
        for attr in ["flatten_config", "compression_kwargs", "decompression_kwargs", "version",
                     "obs_shape", "dtypes", "_dtype", "_compress", "_decompress"]:
            setattr(self, attr, getattr(obj, attr))

    def __setitem__(self, index, value):
        with self._thread_lock:
            self._suppress_get_item = True
            arr = np.ravel(np.asarray(value))
            super().__setitem__(index, self._compress(arr))
            self._suppress_get_item = False

    def __getitem__(self, index):
        # np.ndarray.__setitem__ may invoke np.ndarray.__getitem__ through internal operations
        retrieved = super().__getitem__(index)
        if self._suppress_get_item or retrieved is None:
            return retrieved
        if isinstance(retrieved, np.ndarray):
            return [self._reconstruct_obs(x) for x in retrieved]
        else:
            return self._reconstruct_obs(retrieved)

    def _reconstruct_obs(self, data: bytes):
        obs = self._decompress(data).reshape(self.obs_shape)
        return obs.astype(self._dtype, copy=False)
