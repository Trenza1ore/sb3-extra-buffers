from typing import Union, Iterable, Optional, Any

import warnings
import concurrent.futures
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import BaseBuffer, RolloutBuffer
from sb3_extra_buffers import logger


class DummyVecRolloutBuffer(RolloutBuffer):
    def __init__(self, buffer_size: int, observation_space: spaces.Space, action_space: spaces.Space,
                 device: Union[th.device, str] = "auto", n_envs: int = 1, use_threads: bool = False,
                 buffers: Optional[Iterable[BaseBuffer]] = None, **kwargs):
        warnings.warn("DummyVecRolloutBuffer is fully experimental, use it at caution.", category=ImportWarning)
        super().__init__(buffer_size, observation_space, action_space, device, n_envs, **kwargs)
        del self.observations
        self._buffers = buffers or []
        if use_threads:
            self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(buffers))
        else:
            self._executor = None
        logger.debug(f"Initializing DummyVecBuf{self.__hash__()}")
        self.__patch_methods()
        logger.debug(f"Initialized DummyVecBuf{self.__hash__()} successfully")

    def __del__(self):
        try:
            self._executor.shutdown(cancel_futures=True)
        except Exception:
            del self._executor

    @property
    def full(self):
        return all(buffer.full for buffer in self._buffers)

    @full.setter
    def full(self, val: bool):
        return val

    def _get_samples(self):
        pass

    def __patch_methods(self):
        private_header = "_" + self.__class__.__name__ + "__"
        for attr in dir(self):
            not_private = not (attr.startswith("__") or attr.startswith(private_header))
            if not_private and callable(getattr(self, attr)):
                class Wrapper:
                    def __init__(self, name: str, buffers: list[BaseBuffer], executor: Any = None):
                        self._name = name
                        self._buffers = buffers
                        self._executor = executor

                    def __call__(self, *args, **kwargs):
                        if self._executor is None:
                            return [getattr(buffer, self._name)(*args, **kwargs)for buffer in self._buffers]
                        tasks = [self._executor.submit(getattr(buffer, self._name), *args, **kwargs)
                                 for buffer in self._buffers]
                        concurrent.futures.wait(tasks)
                        return [t.result() for t in tasks]

                setattr(self, attr, Wrapper(attr, self._buffers, self._executor))
                logger.debug(f"Patched DummyVecBuf{self.__hash__()}.{attr}")
