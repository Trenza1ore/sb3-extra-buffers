"""Replay buffers with device-resident observations."""

# pylint: disable=too-many-instance-attributes, too-many-positional-arguments

from functools import lru_cache
from typing import Any, Literal, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import BaseBuffer, ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize

from sb3_extra_buffers.gpu_buffers.base import BaseGpuBuffer
from sb3_extra_buffers.gpu_buffers.observation_store import create_observation_store
from sb3_extra_buffers.gpu_buffers.raw_buffer import SlotArena
from sb3_extra_buffers.gpu_buffers.utils import estimate_max_slot_bytes, numpy_dtype_to_torch


def _elem_bounds(elem_type: th.dtype) -> tuple[Union[int, float], Union[int, float]]:
    if th.is_floating_point(th.zeros(1, dtype=elem_type)):
        info = th.finfo(elem_type)
    else:
        info = th.iinfo(elem_type)
    return info.min, info.max


def _prepare_obs_batch(obs: Union[np.ndarray, th.Tensor], elem_type: th.dtype, buffer_device: th.device) -> th.Tensor:
    """Convert an observation batch to a clipped tensor on ``buffer_device``."""
    obs_tensor = th.as_tensor(obs, device=buffer_device)
    elem_min, elem_max = _elem_bounds(elem_type)
    return obs_tensor.clamp(elem_min, elem_max).to(elem_type)


def _normalize_obs_tensor(
    obs: th.Tensor,
    env: Optional[VecNormalize],
) -> th.Tensor:
    """Apply VecNormalize to a batched observation tensor."""
    if env is None:
        return obs
    obs_numpy = obs.detach().cpu().numpy()
    normalized = env.normalize_obs(obs_numpy)
    return th.as_tensor(normalized, device=obs.device)


class GpuReplayBuffer(ReplayBuffer, BaseGpuBuffer):
    """Replay buffer with observations stored on a Torch device."""

    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    timeouts: np.ndarray

    def __init__(  # pylint: disable=super-init-not-called, too-many-arguments
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        dtypes: Optional[dict] = None,
        normalize_images: bool = False,
        compression_method: str = "none",
        compression_kwargs: Optional[dict] = None,
        decompression_kwargs: Optional[dict] = None,
        output_dtype: Literal["raw", "float"] = "raw",
        buffer_device: Optional[Union[th.device, str]] = None,
        max_slot_bytes: Optional[int] = None,
    ):
        """Create a replay buffer with device-resident observations."""
        BaseBuffer.__init__(  # pylint: disable=non-parent-init-called
            self, buffer_size, observation_space, action_space, device, n_envs=n_envs
        )
        self.normalize_images = normalize_images
        self.flatten_len = int(np.prod(self.obs_shape))
        elem_torch_dtype = numpy_dtype_to_torch(observation_space.dtype)
        self.flatten_config = {"size": self.flatten_len, "dtype": elem_torch_dtype}
        self.output_dtype = th.float32 if output_dtype == "float" else None
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

        self.buffer_size = max(buffer_size // n_envs, 1)

        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "GpuReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = optimize_memory_usage
        self.handle_timeout_termination = handle_timeout_termination

        n_fields = 1 if optimize_memory_usage else 2
        elem_type = self.dtypes["elem_type"]
        runs_type = self.dtypes["runs_type"]
        slot_bytes = max_slot_bytes or estimate_max_slot_bytes(
            self.flatten_len,
            elem_type,
            runs_type,
            compression_method,
        )
        shared_arena = None
        if compression_method != "none":
            shared_arena = SlotArena(
                self.buffer_size * self.n_envs * n_fields,
                slot_bytes,
                device=self.buffer_device,
            )

        self.obs_store = create_observation_store(
            compression_method,
            self.buffer_size,
            self.n_envs,
            self.flatten_len,
            elem_type,
            self.buffer_device,
            field_offset=0,
            compress=self._compress,
            decompress=self._decompress,
            max_slot_bytes=slot_bytes,
            shared_arena=shared_arena,
            runs_type=runs_type,
        )
        self.next_obs_store = None
        if not optimize_memory_usage:
            self.next_obs_store = create_observation_store(
                compression_method,
                self.buffer_size,
                self.n_envs,
                self.flatten_len,
                elem_type,
                self.buffer_device,
                field_offset=1,
                compress=self._compress,
                decompress=self._decompress,
                max_slot_bytes=slot_bytes,
                shared_arena=shared_arena,
                runs_type=runs_type,
            )

        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim),
            dtype=self._maybe_cast_dtype(action_space.dtype),
        )
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        """Add a transition with device-resident observations."""
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        action = action.reshape((self.n_envs, self.action_dim))
        elem_type = self.dtypes["elem_type"]
        obs_tensor = _prepare_obs_batch(obs, elem_type, self.buffer_device)
        next_obs_tensor = _prepare_obs_batch(next_obs, elem_type, self.buffer_device)

        for env_idx in range(self.n_envs):
            self.obs_store.write(self.pos, env_idx, obs_tensor[env_idx].reshape(-1))

        if self.optimize_memory_usage:
            next_pos = (self.pos + 1) % self.buffer_size
            for env_idx in range(self.n_envs):
                self.obs_store.write(next_pos, env_idx, next_obs_tensor[env_idx].reshape(-1))
        else:
            for env_idx in range(self.n_envs):
                self.next_obs_store.write(self.pos, env_idx, next_obs_tensor[env_idx].reshape(-1))

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
        self.reconstruct_obs.cache_clear()
        obs = th.stack(
            [
                self.reconstruct_obs(int(idx), int(env_idx)).reshape(self.obs_shape)
                for idx, env_idx in zip(batch_inds, env_indices)
            ]
        )
        if self.optimize_memory_usage:
            batch_inds_offset = (batch_inds + 1) % self.buffer_size
            n_obs = th.stack(
                [
                    self.reconstruct_obs(int(idx), int(env_idx)).reshape(self.obs_shape)
                    for idx, env_idx in zip(batch_inds_offset, env_indices)
                ]
            )
        else:
            self.reconstruct_nextobs.cache_clear()
            n_obs = th.stack(
                [
                    self.reconstruct_nextobs(int(idx), int(env_idx)).reshape(self.obs_shape)
                    for idx, env_idx in zip(batch_inds, env_indices)
                ]
            )

        obs = _normalize_obs_tensor(obs, env)
        n_obs = _normalize_obs_tensor(n_obs, env)
        if self.output_dtype is not None:
            obs = obs.to(dtype=self.output_dtype)
            n_obs = n_obs.to(dtype=self.output_dtype)
        else:
            obs = obs.to(device=self.device)
            n_obs = n_obs.to(device=self.device)

        if self.normalize_images:
            obs = obs / 255.0
            n_obs = n_obs / 255.0

        actions = self.actions[batch_inds, env_indices, :]
        dones = (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1)
        rewards = self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env)

        return ReplayBufferSamples(
            obs,
            self.to_torch(actions),
            n_obs,
            self.to_torch(dones),
            self.to_torch(rewards),
        )

    @lru_cache(maxsize=1024)
    def reconstruct_obs(self, idx: int, env_idx: int) -> th.Tensor:
        """Return the flattened observation at ``(idx, env_idx)``."""
        return self.obs_store.read(idx, env_idx)

    @lru_cache(maxsize=1024)
    def reconstruct_nextobs(self, idx: int, env_idx: int) -> th.Tensor:
        """Return the flattened next observation at ``(idx, env_idx)``."""
        if self.optimize_memory_usage:
            next_idx = (idx + 1) % self.buffer_size
            return self.obs_store.read(next_idx, env_idx)
        return self.next_obs_store.read(idx, env_idx)
