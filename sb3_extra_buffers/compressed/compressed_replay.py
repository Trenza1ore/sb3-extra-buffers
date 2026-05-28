"""Replay buffers that store compressed observations."""

# pylint: disable=too-many-instance-attributes, too-many-positional-arguments

import warnings
from functools import lru_cache
from typing import Any, Literal, Optional, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import BaseBuffer, ReplayBuffer, psutil
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    ReplayBufferSamples,
)
from stable_baselines3.common.vec_env import VecNormalize

from sb3_extra_buffers.compressed.base import BaseCompressedBuffer


class CompressedReplayBuffer(ReplayBuffer, BaseCompressedBuffer):
    """ReplayBuffer, but compressed!"""

    observations: np.ndarray[object]
    next_observations: Optional[np.ndarray[object]] = None
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    timeouts: np.ndarray

    def __init__(  # pylint: disable=super-init-not-called
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
        compression_method: str = "rle",
        compression_kwargs: Optional[dict] = None,
        decompression_kwargs: Optional[dict] = None,
        output_dtype: Literal["raw", "float"] = "raw",
    ):
        """Create a compressed replay buffer for vector or image observations.

        Args:
            buffer_size: Maximum number of transitions per environment.
            observation_space: Gymnasium observation space.
            action_space: Gymnasium action space.
            device: Torch device used when sampling batches.
            n_envs: Number of parallel environments.
            optimize_memory_usage: Reuse observation slots for next observations.
            handle_timeout_termination: Store timeout flags from ``TimeLimit.truncated``.
            dtypes: Element and run-length dtypes for compression.
            normalize_images: Divide image observations by 255 when sampling.
            compression_method: Registered compression method name.
            compression_kwargs: Keyword arguments for compression.
            decompression_kwargs: Keyword arguments for decompression.
            output_dtype: Sample dtype for observations (``"raw"`` keeps storage dtype).
        """
        # Avoid calling ReplayBuffer.__init__ which might be over-allocating memory for observations
        BaseBuffer.__init__(  # pylint: disable=non-parent-init-called
            self, buffer_size, observation_space, action_space, device, n_envs=n_envs
        )
        self.normalize_images = normalize_images
        self.flatten_len = np.prod(self.obs_shape)
        self.flatten_config = {"shape": self.flatten_len, "dtype": observation_space.dtype}
        self.output_dtype = th.float32 if output_dtype == "float" else None

        # Handle dtypes
        self.dtypes = dtypes or {"elem_type": np.uint8, "runs_type": np.uint16}
        if not isinstance(self.dtypes, dict):
            elem_type = self.dtypes
            self.dtypes = {"elem_type": elem_type, "runs_type": elem_type}

        # Compress and decompress
        self.compression_kwargs = compression_kwargs or self.dtypes
        self.decompression_kwargs = decompression_kwargs or self.dtypes
        BaseCompressedBuffer.__init__(
            self,
            compression_method=compression_method,
            compression_kwargs=self.compression_kwargs,
            decompression_kwargs=self.decompression_kwargs,
            flatten_config=self.flatten_config,
        )

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # there is a bug if both optimize_memory_usage and handle_timeout_termination are true
        # see https://github.com/DLR-RM/stable-baselines3/issues/934
        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "CompressedReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.empty((self.buffer_size, self.n_envs), dtype=object)

        if not optimize_memory_usage:
            # When optimizing memory, `observations` contains also the next observation
            self.next_observations = np.empty((self.buffer_size, self.n_envs), dtype=object)

        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim),
            dtype=self._maybe_cast_dtype(action_space.dtype),
        )

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available
            total_memory_usage: float = (
                self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            )

            if not optimize_memory_usage:
                total_memory_usage += self.next_observations.nbytes

            # Size for one obs after flattening
            elem_size = int(np.dtype(self.dtypes["elem_type"]).itemsize)
            flatten_obs_size = int(self.buffer_size * self.flatten_len * elem_size)

            memory_sufficiency = {}
            msg = f"Available memory: {mem_available / 1e9:.2f}"
            for c_rate in [5, 10, 15, 20, 30, 50, 70, 90, 100]:
                if not optimize_memory_usage:
                    c_rate *= 2
                bytes_usage = flatten_obs_size * (c_rate / 100)
                estimated_usage = total_memory_usage + bytes_usage
                is_sufficient = mem_available > estimated_usage
                memory_sufficiency[c_rate] = is_sufficient
                msg += f"\n[{'v' if is_sufficient else 'x'}]{c_rate:3d}% would take {estimated_usage / 1e9:.2f}GB"

            if not all(memory_sufficiency.values()):
                msg += "\nUsually a suitable observation input should take less than 10% of original size, "
                msg += "but be warned. Also consider using gzip or igzip instead of RLE for RGB-like input."
                warnings.warn(msg)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        """Add a transition, compressing observations before storage.

        Args:
            obs: Current observation batch.
            next_obs: Next observation batch.
            action: Action batch.
            reward: Reward batch.
            done: Episode termination flags.
            infos: Per-environment info dicts from the vectorized environment.
        """
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        elem_type = self.dtypes["elem_type"]
        elem_is_int = np.issubdtype(elem_type, np.integer)
        elem_type_info = np.iinfo(elem_type) if elem_is_int else np.finfo(elem_type)
        elem_min, elem_max = elem_type_info.min, elem_type_info.max

        if isinstance(obs, th.Tensor):
            obs = th.clamp(obs, elem_min, elem_max).cpu().numpy().astype(elem_type, casting="unsafe")
            next_obs = th.clamp(next_obs, elem_min, elem_max).cpu().numpy().astype(elem_type, casting="unsafe")
        else:
            obs = np.clip(obs, elem_min, elem_max, dtype=elem_type, casting="unsafe")
            next_obs = np.clip(next_obs, elem_min, elem_max, dtype=elem_type, casting="unsafe")

        # Compress everything
        self.observations[self.pos] = [self._compress(env_obs.ravel()) for env_obs in obs]

        if self.optimize_memory_usage:
            next_pos = (self.pos + 1) % self.buffer_size
            self.observations[next_pos] = [self._compress(env_obs.ravel()) for env_obs in next_obs]
        else:
            self.next_observations[self.pos] = [self._compress(env_obs.ravel()) for env_obs in next_obs]

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
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
        self.reconstruct_obs.cache_clear()
        obs = np.stack([self.reconstruct_obs(idx, env_idx) for idx, env_idx in zip(batch_inds, env_indices)])
        if self.optimize_memory_usage:
            batch_inds_offset = (batch_inds + 1) % self.buffer_size
            n_obs = np.stack(
                [self.reconstruct_obs(idx, env_idx) for idx, env_idx in zip(batch_inds_offset, env_indices)]
            )
        else:
            self.reconstruct_nextobs.cache_clear()
            n_obs = np.stack([self.reconstruct_nextobs(idx, env_idx) for idx, env_idx in zip(batch_inds, env_indices)])

        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore",
                message="The given NumPy array is not writable.*",
                category=UserWarning,
            )
            obs = th.from_numpy(self._normalize_obs(obs, env)).to(
                device=self.device, dtype=self.output_dtype, copy=True
            )
            n_obs = th.from_numpy(self._normalize_obs(n_obs, env)).to(
                device=self.device, dtype=self.output_dtype, copy=True
            )

        if self.normalize_images:
            obs /= 255.0
            n_obs /= 255.0

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
    def reconstruct_obs(self, idx: int, env_idx: int):
        """Decompress the observation stored at ``(idx, env_idx)``."""
        return self._decompress(self.observations[idx, env_idx]).reshape(self.obs_shape)

    @lru_cache(maxsize=1024)
    def reconstruct_nextobs(self, idx: int, env_idx: int):
        """Decompress the next observation stored at ``(idx, env_idx)``."""
        return self._decompress(self.next_observations[idx, env_idx]).reshape(self.obs_shape)


class CompressedDictReplayBuffer(CompressedReplayBuffer):
    """DictReplayBuffer, but compressed!"""

    observation_space: spaces.Dict
    obs_shape: dict[str, tuple[int, ...]]  # type: ignore[assignment]
    observations: dict[str, np.ndarray]  # type: ignore[assignment]
    next_observations: dict[str, np.ndarray]  # type: ignore[assignment]

    def __init__(  # pylint: disable=super-init-not-called
        self,
        buffer_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
        dtypes: Optional[dict] = None,
        normalize_images: bool = False,
        compression_method: str = "rle",
        compression_kwargs: Optional[dict] = None,
        decompression_kwargs: Optional[dict] = None,
        output_dtype: Literal["raw", "float"] = "raw",
    ):
        """Create a compressed replay buffer for dictionary observations.

        Args:
            buffer_size: Maximum number of transitions per environment.
            observation_space: Gymnasium ``Dict`` observation space.
            action_space: Gymnasium action space.
            device: Torch device used when sampling batches.
            n_envs: Number of parallel environments.
            optimize_memory_usage: Must be ``False`` for dict observations.
            handle_timeout_termination: Store timeout flags from ``TimeLimit.truncated``.
            dtypes: Element and run-length dtypes for compression.
            normalize_images: Divide image observations by 255 when sampling.
            compression_method: Registered compression method name.
            compression_kwargs: Keyword arguments for compression.
            decompression_kwargs: Keyword arguments for decompression.
            output_dtype: Sample dtype for observations (``"raw"`` keeps storage dtype).
        """
        # Avoid calling ReplayBuffer.__init__ which might be over-allocating memory for observations
        BaseBuffer.__init__(  # pylint: disable=non-parent-init-called
            self, buffer_size, observation_space, action_space, device, n_envs=n_envs
        )
        assert isinstance(self.obs_shape, dict), "CompressedDictReplayBuffer must be used with Dict obs space only"
        assert not optimize_memory_usage, "CompressedDictReplayBuffer does not support optimize_memory_usage"
        self.normalize_images = normalize_images
        self.flatten_configs = {
            key: {"shape": np.prod(obs_shape), "dtype": observation_space[key].dtype}
            for key, obs_shape in self.obs_shape.items()
        }
        print(self.flatten_configs)
        self.output_dtype = th.float32 if output_dtype == "float" else None

        # Handle dtypes
        self.dtypes = dtypes or {"elem_type": np.uint8, "runs_type": np.uint16}
        if not isinstance(self.dtypes, dict):
            elem_type = self.dtypes
            self.dtypes = {"elem_type": elem_type, "runs_type": elem_type}

        # Compress and decompress
        self.compression_kwargs = compression_kwargs or self.dtypes
        self.decompression_kwargs = decompression_kwargs or self.dtypes
        BaseCompressedBuffer.__init__(  # pylint: disable=non-parent-init-called
            self,
            compression_method=compression_method,
            compression_kwargs=self.compression_kwargs,
            decompression_kwargs=self.decompression_kwargs,
            flatten_config=next(iter(self.flatten_configs.values())),
        )

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = {key: np.empty((self.buffer_size, self.n_envs), dtype=object) for key in self.obs_shape}
        self.next_observations = {
            key: np.empty((self.buffer_size, self.n_envs), dtype=object) for key in self.obs_shape
        }

        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim),
            dtype=self._maybe_cast_dtype(action_space.dtype),
        )
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

    def add(
        self,
        obs: dict[str, np.ndarray],
        next_obs: dict[str, np.ndarray],
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        """Add a dict observation transition, compressing each key separately.

        Args:
            obs: Current observation dict.
            next_obs: Next observation dict.
            action: Action batch.
            reward: Reward batch.
            done: Episode termination flags.
            infos: Per-environment info dicts from the vectorized environment.
        """
        # Copy to avoid modification by reference
        elem_type = self.dtypes["elem_type"]
        elem_is_int = np.issubdtype(elem_type, np.integer)
        elem_type_info = np.iinfo(elem_type) if elem_is_int else np.finfo(elem_type)
        elem_min, elem_max = elem_type_info.min, elem_type_info.max

        # Compress everything
        for key in self.observations:
            obs_ = obs[key]
            next_obs_ = next_obs[key]
            # Reshape needed when using multiple envs with discrete observations
            # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
            if isinstance(self.observation_space.spaces[key], spaces.Discrete):
                obs_ = obs_.reshape((self.n_envs,) + self.obs_shape[key])
                next_obs_ = next_obs_.reshape((self.n_envs,) + self.obs_shape[key])
            if isinstance(obs_, th.Tensor):
                obs_ = th.clamp(obs_, elem_min, elem_max).cpu().numpy().astype(elem_type, casting="unsafe")
                next_obs_ = th.clamp(next_obs_, elem_min, elem_max).cpu().numpy().astype(elem_type, casting="unsafe")
            else:
                obs_ = np.clip(obs_, elem_min, elem_max, dtype=elem_type, casting="unsafe")
                next_obs_ = np.clip(next_obs_, elem_min, elem_max, dtype=elem_type, casting="unsafe")
            self.observations[key][self.pos] = [self._compress(env_obs.ravel()) for env_obs in obs_]
            self.next_observations[key][self.pos] = [self._compress(env_obs.ravel()) for env_obs in next_obs_]

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> DictReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))
        self._reconstruct_obs.cache_clear()
        self._reconstruct_nextobs.cache_clear()
        obs = {
            key: np.stack([self._reconstruct_obs(idx, env_idx, key) for idx, env_idx in zip(batch_inds, env_indices)])
            for key in self.observations.keys()
        }
        n_obs = {
            key: np.stack(
                [self._reconstruct_nextobs(idx, env_idx, key) for idx, env_idx in zip(batch_inds, env_indices)]
            )
            for key in self.next_observations.keys()
        }

        obs = self._normalize_obs(obs, env)
        n_obs = self._normalize_obs(n_obs, env)
        assert isinstance(obs, dict)
        assert isinstance(n_obs, dict)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                action="ignore",
                message="The given NumPy array is not writable.*",
                category=UserWarning,
            )
            observations = {
                key: th.from_numpy(obs_).to(device=self.device, dtype=self.output_dtype, copy=True)
                for key, obs_ in obs.items()
            }
            next_observations = {
                key: th.from_numpy(obs_).to(device=self.device, dtype=self.output_dtype, copy=True)
                for key, obs_ in n_obs.items()
            }

        if self.normalize_images:
            observations = {key: obs / 255.0 for key, obs in observations.items()}
            next_observations = {key: obs / 255.0 for key, obs in next_observations.items()}

        actions = self.actions[batch_inds, env_indices, :]
        dones = (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1)
        rewards = self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env)

        return DictReplayBufferSamples(
            observations,
            self.to_torch(actions),
            next_observations,
            self.to_torch(dones),
            self.to_torch(rewards),
        )

    @lru_cache(maxsize=1024)
    def _reconstruct_obs(self, idx: int, env_idx: int, key: str):
        return self._decompress(self.observations[key][idx, env_idx], arr_configs=self.flatten_configs[key]).reshape(
            self.obs_shape[key]
        )

    @lru_cache(maxsize=1024)
    def _reconstruct_nextobs(self, idx: int, env_idx: int, key: str):
        return self._decompress(
            self.next_observations[key][idx, env_idx],
            arr_configs=self.flatten_configs[key],
        ).reshape(self.obs_shape[key])
