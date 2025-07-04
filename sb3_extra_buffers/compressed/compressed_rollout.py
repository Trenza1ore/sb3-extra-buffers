from typing import Literal, Generator, Optional, Union

import warnings
from functools import partial
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.buffers import BaseBuffer, RolloutBuffer, RolloutBufferSamples, VecNormalize
from sb3_extra_buffers.compressed.compression_methods import _compression_method_mapping
from sb3_extra_buffers.compressed.utils import find_optimal_shape


class CompressedRolloutBuffer(RolloutBuffer):
    observations: np.ndarray[object]
    len_arr: Optional[np.ndarray[object]] = None
    pos_arr: Optional[np.ndarray[object]] = None
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
        compression_method: Literal["rle", "gzip", "none"] = "rle",
        compression_kwargs: Optional[dict] = None,
        decompression_kwargs: Optional[dict] = None,
        auto_slice: bool = False,
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
        self.dtypes = dtypes or dict(len_type=np.uint16, pos_type=np.uint16, elem_type=np.uint8)
        if not isinstance(self.dtypes, dict):
            elem_type = self.dtypes
            self.dtypes = dict(len_type=elem_type, pos_type=elem_type, elem_type=elem_type)

        # Compress and decompress
        self.rle_like = _compression_method_mapping[compression_method].rle_like
        self.compress = partial(_compression_method_mapping[compression_method].compress, **self.dtypes,
                                **compression_kwargs)
        self.decompress = _compression_method_mapping[compression_method].decompress
        compression_kwargs = compression_kwargs or {}
        self.decompression_kwargs = decompression_kwargs or {}
        if compression_method == "gzip":
            self.decompression_kwargs["elem_type"] = self.dtypes["elem_type"]

        # Handle auto slicing
        if self.rle_like and auto_slice:
            elem_type = self.dtypes["elem_type"]
            self.auto_slice = find_optimal_shape(arr_len=self.flatten_len, dtype=elem_type)
            self.dtypes = dict(len_type=elem_type, pos_type=elem_type, elem_type=elem_type)
        else:
            self.auto_slice = None
        self.reset()

    def reset(self) -> None:
        if self.auto_slice is None:
            buffer_shape = (self.buffer_size, self.n_envs)
        else:
            slice_num = self.auto_slice[0] + bool(self.auto_slice[1])
            buffer_shape = (self.buffer_size, self.n_envs, slice_num)
        self.observations = np.empty(buffer_shape, dtype=object)
        if self.rle_like:
            self.len_arr = np.empty(buffer_shape, dtype=object)
            self.pos_arr = np.empty(buffer_shape, dtype=object)
        else:
            self.len_arr = self.pos_arr = None

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
        for env in range(self.n_envs):
            arr = obs[env].ravel()
            if self.auto_slice:
                _, col, remainder = self.auto_slice
                to_iter = [arr[i:i+col] for i in range(0, self.flatten_len-remainder, col)]
                if remainder:
                    to_iter.append(arr[-remainder:])
                for i, slice in enumerate(to_iter):
                    l, p, e = self.compress(slice)
                    if self.rle_like:
                        self.len_arr[self.pos, env, i] = l
                        self.pos_arr[self.pos, env, i] = p
                    self.observations[self.pos, env, i] = e
            else:
                l, p, e = self.compress(arr)
                if self.rle_like:
                    self.len_arr[self.pos, env] = l
                    self.pos_arr[self.pos, env] = p
                self.observations[self.pos, env] = e

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
            if self.rle_like:
                _tensor_names += ["len_arr", "pos_arr"]

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
            self.values[batch_inds].flatten(),
            self.log_probs[batch_inds].flatten(),
            self.advantages[batch_inds].flatten(),
            self.returns[batch_inds].flatten(),
        )
        return RolloutBufferSamples(obs, *tuple(map(self.to_torch, data)))

    def reconstruct_obs(self, idx: int):
        if self.auto_slice:
            row, col, remainder = self.auto_slice
            slice_num = row + bool(remainder)
            slide_len = [col] * row
            if remainder:
                slide_len.append(remainder)
            if self.rle_like:
                len_arr = (np.frombuffer(self.len_arr[idx][i], self.dtypes["len_type"]) for i in range(slice_num))
                pos_arr = (np.frombuffer(self.pos_arr[idx][i], self.dtypes["pos_type"]) for i in range(slice_num))
                elem_arr = (np.frombuffer(self.observations[idx][i], self.dtypes["elem_type"]) for i in range(
                    slice_num))
            else:
                len_arr = pos_arr = [None] * slice_num
                elem_arr = (self.observations[idx][i] for i in range(slice_num))
            dtype = self.flatten_config["dtype"]

            buffer = []
            for l, p, e, length in zip(len_arr, pos_arr, elem_arr, slide_len):
                arr_configs = dict(shape=length, dtype=dtype)
                buffer.append(self.decompress(l, p, e, arr_configs=arr_configs, **self.decompression_kwargs))

            obs = np.concatenate(buffer, dtype=dtype).reshape(self.obs_shape)
            return th.from_numpy(obs).to(self.device, th.float32)
        if self.rle_like:
            len_arr = np.frombuffer(self.len_arr[idx][0], self.dtypes["len_type"])
            pos_arr = np.frombuffer(self.pos_arr[idx][0], self.dtypes["pos_type"])
        else:
            len_arr = pos_arr = None
        elem_arr = np.frombuffer(self.observations[idx][0], self.dtypes["elem_type"])
        obs = self.decompress(len_arr, pos_arr, elem_arr, arr_configs=self.flatten_config,
                              **self.decompression_kwargs).reshape(self.obs_shape)
        return th.from_numpy(obs).to(self.device, th.float32)
