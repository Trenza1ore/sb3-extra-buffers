"""Experimental rollout buffer that delegates to multiple underlying buffers."""

# pylint: disable=too-few-public-methods, too-many-positional-arguments

import concurrent.futures
import warnings
from typing import Any, Iterable, Optional, Union

import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import BaseBuffer, RolloutBuffer

from sb3_extra_buffers.logging import logger


class DummyVecRolloutBuffer(RolloutBuffer):
    """Forward rollout API calls to a list of underlying buffers."""

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        use_threads: bool = False,
        buffers: Optional[Iterable[BaseBuffer]] = None,
        **kwargs,
    ):
        """Create a facade over multiple rollout buffers.

        Args:
            buffer_size: Steps per environment for each underlying buffer.
            observation_space: Shared observation space.
            action_space: Shared action space.
            device: Torch device passed to the base rollout buffer.
            n_envs: Number of parallel environments for the facade.
            use_threads: Run delegated calls in a thread pool when ``True``.
            buffers: Underlying buffers to invoke; defaults to an empty list.
            **kwargs: Additional arguments forwarded to :class:`RolloutBuffer`.
        """
        warnings.warn(
            "DummyVecRolloutBuffer is fully experimental, use it at caution.",
            category=ImportWarning,
        )
        super().__init__(buffer_size, observation_space, action_space, device, n_envs, **kwargs)
        del self.observations
        self._buffers = buffers or []
        if use_threads:
            self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(buffers))
        else:
            self._executor = None
        logger.debug("Initializing DummyVecBuf%s", hash(self))
        self.__patch_methods()
        logger.debug("Initialized DummyVecBuf%s successfully", hash(self))

    def __del__(self):
        """Shut down the optional thread pool executor."""
        try:
            self._executor.shutdown(cancel_futures=True)
        except Exception:  # pylint: disable=broad-exception-caught
            del self._executor

    @property
    def full(self):
        """Return whether every underlying buffer is full."""
        return all(buffer.full for buffer in self._buffers)

    @full.setter
    def full(self, val: bool):
        return val

    def _get_samples(self, *args):
        pass

    def __patch_methods(self):
        private_header = "_" + self.__class__.__name__ + "__"
        for attr in dir(self):
            not_private = not (attr.startswith("__") or attr.startswith(private_header))
            if not_private and callable(getattr(self, attr)):

                class Wrapper:
                    """Wrapper class to perform some meta-programming magic."""

                    def __init__(self, name: str, buffers: list[BaseBuffer], executor: Any = None):
                        self._name = name
                        self._buffers = buffers
                        self._executor = executor

                    def __call__(self, *args, **kwargs):
                        if self._executor is None:
                            return [getattr(buffer, self._name)(*args, **kwargs) for buffer in self._buffers]
                        tasks = [
                            self._executor.submit(getattr(buffer, self._name), *args, **kwargs)
                            for buffer in self._buffers
                        ]
                        concurrent.futures.wait(tasks)
                        return [t.result() for t in tasks]

                setattr(self, attr, Wrapper(attr, self._buffers, self._executor))
                logger.debug("Patched DummyVecBuf%s.%s", hash(self), attr)
