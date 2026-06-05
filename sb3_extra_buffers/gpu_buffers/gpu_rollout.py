"""Rollout buffers with device-resident observations."""

# pylint: disable=too-many-instance-attributes, too-many-positional-arguments

from typing import Generator, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import BaseBuffer, RolloutBuffer, RolloutBufferSamples, VecNormalize

from sb3_extra_buffers import sb3_version
from sb3_extra_buffers.gpu_buffers.base import BaseGpuBuffer
from sb3_extra_buffers.gpu_buffers.gpu_replay import _normalize_obs_tensor, _prepare_obs_batch
from sb3_extra_buffers.gpu_buffers.observation_store import create_observation_store
from sb3_extra_buffers.gpu_buffers.raw_buffer import SharedRawHeap
from sb3_extra_buffers.gpu_buffers.utils import estimate_total_heap_bytes, numpy_dtype_to_torch
from sb3_extra_buffers.logging import logger

LEGACY_BEHAVIOR = sb3_version() < (2, 7, 1)


class GpuRolloutBuffer(RolloutBuffer, BaseGpuBuffer):
    """Rollout buffer with observations stored on a Torch device."""

    actions: np.ndarray
    rewards: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    episode_starts: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray

    def __init__(  # pylint: disable=super-init-not-called
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
        dtypes: Optional[dict] = None,
        normalize_images: bool = False,
        compression_method: str = "none",
        compression_kwargs: Optional[dict] = None,
        decompression_kwargs: Optional[dict] = None,
        buffer_device: Optional[Union[th.device, str]] = None,
        heap_bytes: Optional[int] = None,
        max_slot_bytes: Optional[int] = None,
    ):
        """Create a rollout buffer with device-resident observations."""
        BaseBuffer.__init__(  # pylint: disable=non-parent-init-called
            self, buffer_size, observation_space, action_space, device, n_envs=n_envs
        )
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False
        self.normalize_images = normalize_images
        self.flatten_len = int(np.prod(self.obs_shape))
        if LEGACY_BEHAVIOR:
            elem_torch_dtype = th.float32
        else:
            elem_torch_dtype = numpy_dtype_to_torch(observation_space.dtype)
        self.flatten_config = {"size": self.flatten_len, "dtype": elem_torch_dtype}
        self.buffer_device = th.device(buffer_device) if buffer_device is not None else self.device

        self.dtypes = dtypes or {"elem_type": th.uint8, "runs_type": th.uint16}
        if not isinstance(self.dtypes, dict):
            elem_type = self.dtypes
            self.dtypes = {"elem_type": elem_type, "runs_type": elem_type}

        self.compression_kwargs = compression_kwargs or self.dtypes
        self.decompression_kwargs = decompression_kwargs or self.dtypes
        BaseGpuBuffer.__init__(
            self,
            compression_method=compression_method,
            compression_kwargs=self.compression_kwargs,
            decompression_kwargs=self.decompression_kwargs,
            flatten_config=self.flatten_config,
        )
        self.compression_method = compression_method
        self.heap_bytes = heap_bytes
        self.max_slot_bytes = max_slot_bytes
        self.reset()

    def reset(self) -> None:
        """Clear rollout storage and reset the write position."""
        elem_type = self.dtypes["elem_type"]
        runs_type = self.dtypes["runs_type"]
        n_cells = self.buffer_size * self.n_envs
        if self.heap_bytes is not None:
            total_heap_bytes = self.heap_bytes
        elif self.max_slot_bytes is not None:
            total_heap_bytes = n_cells * self.max_slot_bytes
        else:
            total_heap_bytes = estimate_total_heap_bytes(
                n_cells,
                self.flatten_len,
                elem_type,
                runs_type,
                self.compression_method,
            )

        self.shared_heap = None
        if self.compression_method != "none":
            self.shared_heap = SharedRawHeap(n_cells, total_heap_bytes, device=self.buffer_device)

        self.obs_store = create_observation_store(
            self.compression_method,
            self.buffer_size,
            self.n_envs,
            self.flatten_len,
            elem_type,
            self.buffer_device,
            field_offset=0,
            compress=self._compress,
            decompress=self._decompress,
            shared_heap=self.shared_heap,
            runs_type=runs_type,
        )

        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim),
            dtype=self.action_space.dtype,
        )
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.generator_ready = False
        BaseBuffer.reset(self)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob: th.Tensor,
    ) -> None:
        """Add a rollout step with a device-resident observation."""
        if len(log_prob.shape) == 0:
            log_prob = log_prob.reshape(-1, 1)

        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        action = action.reshape((self.n_envs, self.action_dim))
        elem_type = self.dtypes["elem_type"]
        obs_tensor = _prepare_obs_batch(obs, elem_type, self.buffer_device)

        for env_idx in range(self.n_envs):
            self.obs_store.write(self.pos, env_idx, obs_tensor[env_idx].reshape(-1))

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            if self.shared_heap is not None:
                self.shared_heap.compact()

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        """Yield shuffled rollout minibatches after the buffer is full."""
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        if not self.generator_ready:
            self.obs_store.flatten()
            _tensor_names = ["actions", "values", "log_probs", "advantages", "returns"]
            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx : start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> RolloutBufferSamples:
        obs = th.stack([self.reconstruct_obs(int(i)).reshape(self.obs_shape) for i in batch_inds])
        obs = _normalize_obs_tensor(obs, env)
        if self.normalize_images:
            obs = obs / 255.0
        obs = obs.to(device=self.device)
        data = (
            self.actions[batch_inds].astype(np.float32, copy=False),
            self.values[batch_inds].ravel(),
            self.log_probs[batch_inds].ravel(),
            self.advantages[batch_inds].ravel(),
            self.returns[batch_inds].ravel(),
        )
        return RolloutBufferSamples(obs, *tuple(map(self.to_torch, data)))

    def reconstruct_obs(self, idx: int) -> th.Tensor:
        """Return the flattened observation at flattened index ``idx``."""
        return self.obs_store.read_flat(idx)


if LEGACY_BEHAVIOR:

    def legacy_reconstruct_obs(self, idx: int) -> th.Tensor:
        """Decompress the flattened observation at ``idx`` as float32."""
        return self.obs_store.read_flat(idx).to(device=self.device, dtype=th.float32)

    GpuRolloutBuffer.reconstruct_obs = legacy_reconstruct_obs
    logger.warning(
        "Legacy Stable Baselines3 version %s detected, GpuRolloutBuffer will always use data type float32 "
        "for returned observations.",
        sb3_version(),
    )
