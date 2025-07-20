from typing import Generator, Optional, Union

import warnings
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import BaseBuffer, RolloutBuffer, RolloutBufferSamples, VecNormalize
from sb3_extra_buffers.compressed.base import BaseCompressedBuffer


class CompressedRolloutBuffer(RolloutBuffer, BaseCompressedBuffer):
    """ReplayBuffer, but compressed!"""
    observations: np.ndarray[object]
    actions: np.ndarray
    rewards: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    episode_starts: np.ndarray
    log_probs: np.ndarray
    values: np.ndarray

    def __init__(
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
        compression_method: str = "rle",
        compression_kwargs: Optional[dict] = None,
        decompression_kwargs: Optional[dict] = None
    ):
        # Avoid calling RolloutBuffer.reset which might be over-allocating memory for observations
        BaseBuffer.__init__(self, buffer_size, observation_space, action_space, device, n_envs=n_envs)
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.generator_ready = False
        self.normalize_images = normalize_images
        self.flatten_len = np.prod(self.obs_shape)
        self.flatten_config = dict(shape=self.flatten_len, dtype=np.float32)

        # Handle dtypes
        self.dtypes = dtypes or dict(elem_type=np.uint8, runs_type=np.uint16)
        if not isinstance(self.dtypes, dict):
            elem_type = self.dtypes
            self.dtypes = dict(elem_type=elem_type, runs_type=elem_type)

        # Compress and decompress
        self.compression_kwargs = compression_kwargs or self.dtypes
        self.decompression_kwargs = decompression_kwargs or self.dtypes
        BaseCompressedBuffer.__init__(self, compression_method=compression_method,
                                      compression_kwargs=self.compression_kwargs,
                                      decompression_kwargs=self.decompression_kwargs,
                                      flatten_config=self.flatten_config)
        self.reset()

    def reset(self) -> None:
        self.observations = np.empty((self.buffer_size, self.n_envs), dtype=object)

        self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
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
        """
        :param obs: Observation
        :param action: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        """
        if len(log_prob.shape) == 0:
            # Reshape 0-d tensor to avoid error
            log_prob = log_prob.reshape(-1, 1)

        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        elem_type = self.dtypes["elem_type"]
        elem_is_int = np.issubdtype(elem_type, np.integer)
        elem_type_info = np.iinfo(elem_type) if elem_is_int else np.finfo(elem_type)
        elem_min, elem_max = elem_type_info.min, elem_type_info.max

        if isinstance(obs, th.Tensor):
            obs = th.clamp(obs, elem_min, elem_max).cpu().numpy().astype(elem_type, casting="unsafe")
        else:
            obs = np.clip(obs, elem_min, elem_max, dtype=elem_type, casting="unsafe")

        # Compress everything
        self.observations[self.pos] = [self._compress(env_obs.ravel()) for env_obs in obs]

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(self, batch_size: Optional[int] = None) -> Generator[RolloutBufferSamples, None, None]:
        assert self.full, ""
        indices = np.random.permutation(self.buffer_size * self.n_envs)
        # Prepare the data
        if not self.generator_ready:
            _tensor_names = [
                "observations",
                "actions",
                "values",
                "log_probs",
                "advantages",
                "returns",
            ]

            for tensor in _tensor_names:
                self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
            self.generator_ready = True

        # Return everything, don't create minibatches
        if batch_size is None:
            batch_size = self.buffer_size * self.n_envs

        start_idx = 0
        while start_idx < self.buffer_size * self.n_envs:
            yield self._get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> RolloutBufferSamples:
        with warnings.catch_warnings():
            warnings.filterwarnings(action="ignore", message="The given NumPy array is not writable.*",
                                    category=UserWarning)
            obs = th.stack([self.reconstruct_obs(i) for i in batch_inds])
        if self.normalize_images:
            obs /= 255.0
        data = (
            self.actions[batch_inds],
            self.values[batch_inds].ravel(),
            self.log_probs[batch_inds].ravel(),
            self.advantages[batch_inds].ravel(),
            self.returns[batch_inds].ravel(),
        )
        return RolloutBufferSamples(obs, *tuple(map(self.to_torch, data)))

    def reconstruct_obs(self, idx: int):
        obs = self._decompress(self.observations[idx, 0]).reshape(self.obs_shape)
        return th.from_numpy(obs).to(self.device, th.float32)
