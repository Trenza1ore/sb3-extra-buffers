"""Unit tests for compressed buffer classes."""

# pylint: disable=protected-access, attribute-defined-outside-init

import os
import re

import numpy as np
import pytest
import torch
from gymnasium import spaces
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.buffers import ReplayBuffer, RolloutBuffer
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper

from sb3_extra_buffers import sb3_version
from sb3_extra_buffers.compressed import (
    CompressedDictReplayBuffer,
    CompressedDictRolloutBuffer,
    CompressedReplayBuffer,
    CompressedRolloutBuffer,
    find_buffer_dtypes,
)
from sb3_extra_buffers.compressed.compression_methods import has_lz4, has_zstd
from sb3_extra_buffers.training_utils.atari import make_env

ENV_TO_TEST = ["MsPacmanNoFrameskip-v4", "PongNoFrameskip-v4"]

TEST_SEED = 2050808
NUM_OBS_SAMPLES = 10
FRAME_SKIP_INTERVAL = 4
TEST_TRAINING = re.compile(r"^(zstd)$")
DUMP_DEBUG_OBSERVATION = os.environ.get("DISABLE_TEST_OBSERVATIONS_SAVE", "false").lower() not in ("true", "1", "yes")
SB3_VERSION_TUPLE = sb3_version()


class RawObservationStore(VecEnvWrapper):
    """VecEnv wrapper that keeps a copy of the latest observation from the env."""

    def reset(self) -> np.ndarray:
        obs = self.venv.reset()
        self.last_obs = obs.copy()
        return obs

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        self.last_obs = obs.copy()
        return obs, rewards, dones, infos


def _dict_space():
    return spaces.Dict(
        {
            "image": spaces.Box(0, 255, shape=(2, 2), dtype=np.uint8),
            "vector": spaces.Box(0, 255, shape=(3,), dtype=np.uint8),
        }
    )


def _dict_obs(step: int, n_envs: int):
    image = np.full((n_envs, 2, 2), step, dtype=np.uint8)
    vector = np.full((n_envs, 3), step + 1, dtype=np.uint8)
    return {"image": image, "vector": vector}


def is_sb3_version_gte(target_version):
    """Check if current SB3 version is >= target version (major, minor, patch)."""
    return SB3_VERSION_TUPLE >= target_version


def _compression_methods():
    methods = ["rle", "rle-old", "rle-jit", "gzip", "igzip", "none"]
    if has_zstd():
        methods.insert(0, "zstd")
    if has_lz4():
        methods.extend(["lz4-frame/", "lz4-block/"])
    expanded = []
    for compress_method in methods:
        suffixes = [""]
        if compress_method.startswith("rle"):
            pass
        elif compress_method.startswith("igzip"):
            suffixes = ["0"]
        elif compress_method.startswith("zstd"):
            suffixes = ["-5"]
        else:
            suffixes = ["1"]
        for suffix in suffixes:
            expanded.append(compress_method + suffix)
    return expanded


def get_tests():
    """Get all test configs to run."""
    for compression_method in _compression_methods():
        for n_envs in [1, 2]:
            yield ("dict", None, compression_method, n_envs, 1)
        for n_stack in [1, 4]:
            for env_id in ENV_TO_TEST:
                for n_envs in [1, 2]:
                    yield ("atari", env_id, compression_method, n_envs, n_stack)


def _decompress_rollout(buffer, step_idx: int, env_idx: int) -> np.ndarray:
    compressed = buffer.observations[step_idx][env_idx]
    return buffer._decompress(compressed).reshape(buffer.obs_shape)


def _decompress_dict_rollout(buffer, step_idx: int, env_idx: int, key: str) -> np.ndarray:
    compressed = buffer.observations[key][step_idx][env_idx]
    return buffer._decompress(compressed, arr_configs=buffer.flatten_configs[key]).reshape(buffer.obs_shape[key])


def _decompress_replay(buffer, step_idx: int, env_idx: int, *, next_obs: bool = False) -> np.ndarray:
    if next_obs:
        if buffer.optimize_memory_usage:
            step_idx = (step_idx + 1) % buffer.buffer_size
            compressed = buffer.observations[step_idx][env_idx]
        else:
            compressed = buffer.next_observations[step_idx][env_idx]
    else:
        compressed = buffer.observations[step_idx][env_idx]
    return buffer._decompress(compressed).reshape(buffer.obs_shape)


def _decompress_dict_replay(buffer, step_idx: int, env_idx: int, key: str, *, next_obs: bool = False) -> np.ndarray:
    if next_obs:
        if buffer.optimize_memory_usage:
            step_idx = (step_idx + 1) % buffer.buffer_size
            compressed = buffer.observations[key][step_idx][env_idx]
        else:
            compressed = buffer.next_observations[key][step_idx][env_idx]
    else:
        compressed = buffer.observations[key][step_idx][env_idx]
    return buffer._decompress(compressed, arr_configs=buffer.flatten_configs[key]).reshape(buffer.obs_shape[key])


def _assert_obs_equal(raw: np.ndarray, decompressed: np.ndarray) -> None:
    np.testing.assert_array_equal(raw, decompressed)


def _assert_dict_obs_equal(
    raw: dict[str, np.ndarray],
    buffer,
    step_idx: int,
    env_idx: int,
    *,
    buffer_type: str,
    next_obs: bool = False,
) -> None:
    for key in raw:
        if buffer_type == "rollout":
            decompressed = _decompress_dict_rollout(buffer, step_idx, env_idx, key)
        else:
            decompressed = _decompress_dict_replay(buffer, step_idx, env_idx, key, next_obs=next_obs)
        _assert_obs_equal(raw[key][env_idx], decompressed)


def _collect_atari_observations(env: RawObservationStore, rng, n_envs: int, n_actions: int):
    raw_samples = []
    env.reset()
    for _ in range(NUM_OBS_SAMPLES):
        raw_samples.append(env.last_obs.copy())
        for _ in range(FRAME_SKIP_INTERVAL):
            actions = rng.integers(0, n_actions, size=(n_envs,))
            env.step(actions)
    return raw_samples


def _collect_dict_observations(rng, n_envs: int, n_actions: int):
    raw_samples = []
    step = 0
    for _ in range(NUM_OBS_SAMPLES):
        raw_samples.append(_dict_obs(step, n_envs))
        for _ in range(FRAME_SKIP_INTERVAL):
            step += 1
            _ = rng.integers(0, n_actions, size=(n_envs,))
    return raw_samples


def _fill_rollout_buffer(buffer, raw_samples, n_envs: int, rng):
    n_actions = buffer.action_space.n
    for step_idx, raw_obs in enumerate(raw_samples):
        buffer.add(
            obs=raw_obs,
            action=rng.integers(0, n_actions, size=(n_envs,)),
            reward=np.zeros(n_envs, dtype=np.float32),
            episode_start=np.zeros(n_envs, dtype=bool),
            value=torch.zeros(n_envs),
            log_prob=torch.zeros(n_envs),
        )
        if isinstance(raw_obs, dict):
            _assert_dict_obs_equal(raw_obs, buffer, step_idx, env_idx=0, buffer_type="rollout")
            if n_envs > 1:
                _assert_dict_obs_equal(raw_obs, buffer, step_idx, env_idx=1, buffer_type="rollout")
        else:
            for env_idx in range(n_envs):
                decompressed = _decompress_rollout(buffer, step_idx, env_idx)
                _assert_obs_equal(raw_obs[env_idx], decompressed)


def _fill_replay_buffer(buffer, raw_samples, n_envs: int, rng, *, space_kind: str):
    n_actions = buffer.action_space.n
    for step_idx, raw_obs in enumerate(raw_samples):
        if space_kind == "dict":
            next_step = step_idx + 1 if step_idx + 1 < len(raw_samples) else step_idx + FRAME_SKIP_INTERVAL + 1
            next_obs = _dict_obs(next_step, n_envs)
        else:
            next_obs = raw_obs if step_idx + 1 >= len(raw_samples) else raw_samples[step_idx + 1]
        buffer.add(
            obs=raw_obs,
            next_obs=next_obs,
            action=rng.integers(0, n_actions, size=(n_envs,)),
            reward=np.zeros(n_envs, dtype=np.float32),
            done=np.zeros(n_envs, dtype=bool),
            infos=[{} for _ in range(n_envs)],
        )
        if isinstance(raw_obs, dict):
            _assert_dict_obs_equal(raw_obs, buffer, step_idx, env_idx=0, buffer_type="replay")
            _assert_dict_obs_equal(next_obs, buffer, step_idx, env_idx=0, buffer_type="replay", next_obs=True)
            if n_envs > 1:
                _assert_dict_obs_equal(raw_obs, buffer, step_idx, env_idx=1, buffer_type="replay")
                _assert_dict_obs_equal(next_obs, buffer, step_idx, env_idx=1, buffer_type="replay", next_obs=True)
        else:
            for env_idx in range(n_envs):
                decompressed = _decompress_replay(buffer, step_idx, env_idx)
                _assert_obs_equal(raw_obs[env_idx], decompressed)
                decompressed_next = _decompress_replay(buffer, step_idx, env_idx, next_obs=True)
                _assert_obs_equal(next_obs[env_idx], decompressed_next)


def _make_buffer(
    buffer_type: str,
    space_kind: str,
    observation_space,
    n_envs: int,
    compression_method: str,
    buffer_dtypes: dict,
):
    # ReplayBuffer divides buffer_size by n_envs; rollout stores one row per step.
    buffer_size = NUM_OBS_SAMPLES * n_envs if buffer_type == "replay" else NUM_OBS_SAMPLES
    kwargs = dict(
        buffer_size=buffer_size,
        observation_space=observation_space,
        action_space=spaces.Discrete(2),
        n_envs=n_envs,
        compression_method=compression_method,
        dtypes=buffer_dtypes,
    )
    if space_kind == "dict":
        if buffer_type == "rollout":
            return CompressedDictRolloutBuffer(**kwargs)
        return CompressedDictReplayBuffer(**kwargs)
    if buffer_type == "rollout":
        return CompressedRolloutBuffer(**kwargs)
    return CompressedReplayBuffer(**kwargs)


@pytest.mark.parametrize("space_kind,env_id,compression_method,n_envs,n_stack", list(get_tests()))
def test_rollout(space_kind: str, env_id: str, compression_method: str, n_envs: int, n_stack: int):
    """Test rollout buffers."""
    compressed_buffer_test(
        space_kind,
        env_id,
        compression_method,
        n_envs,
        n_stack,
        buffer_type="rollout",
    )


@pytest.mark.parametrize("space_kind,env_id,compression_method,n_envs,n_stack", list(get_tests()))
def test_replay(space_kind: str, env_id: str, compression_method: str, n_envs: int, n_stack: int):
    """Test replay buffers."""
    compressed_buffer_test(
        space_kind,
        env_id,
        compression_method,
        n_envs,
        n_stack,
        buffer_type="replay",
    )


def compressed_buffer_test(
    space_kind: str,
    env_id: str,
    compression_method: str,
    n_envs: int,
    n_stack: int,
    buffer_type: str,
):
    """Unified test function for compressed buffers."""
    rng = np.random.default_rng(TEST_SEED)

    if space_kind == "dict":
        observation_space = _dict_space()
        buffer_dtypes = find_buffer_dtypes(
            obs_shape=observation_space["image"].shape,
            elem_dtype=observation_space["image"].dtype,
            compression_method=compression_method,
        )
        raw_samples = _collect_dict_observations(rng, n_envs, n_actions=2)
        env = None
    else:
        env = make_env(env_id=env_id, n_envs=n_envs, vec_env_cls=DummyVecEnv, framestack=n_stack)
        env = RawObservationStore(env)
        observation_space = env.observation_space
        buffer_dtypes = find_buffer_dtypes(
            obs_shape=observation_space.shape,
            elem_dtype=observation_space.dtype,
            compression_method=compression_method,
        )
        raw_samples = _collect_atari_observations(env, rng, n_envs, env.action_space.n)

    buffer = _make_buffer(
        buffer_type,
        space_kind,
        observation_space,
        n_envs,
        compression_method,
        buffer_dtypes,
    )

    if buffer_type == "rollout":
        _fill_rollout_buffer(buffer, raw_samples, n_envs, rng)
        buffer.compute_returns_and_advantage(last_values=torch.zeros(n_envs), dones=np.zeros(n_envs, dtype=bool))
        samples = next(buffer.get(batch_size=NUM_OBS_SAMPLES * n_envs))
        if space_kind == "dict":
            assert samples.observations["image"].shape == (NUM_OBS_SAMPLES * n_envs, 2, 2)
            assert samples.observations["vector"].shape == (NUM_OBS_SAMPLES * n_envs, 3)
            assert samples.observations["image"].dtype == torch.uint8
        else:
            assert samples.observations.shape[0] == NUM_OBS_SAMPLES * n_envs
        assert samples.actions.shape == (NUM_OBS_SAMPLES * n_envs, 1)
    else:
        _fill_replay_buffer(buffer, raw_samples, n_envs, rng, space_kind=space_kind)
        batch_size = min(4, NUM_OBS_SAMPLES * n_envs)
        samples = buffer.sample(batch_size=batch_size)
        if space_kind == "dict":
            assert samples.observations["image"].shape == (batch_size, 2, 2)
            assert samples.next_observations["vector"].shape == (batch_size, 3)
            assert samples.observations["image"].dtype == torch.uint8
        else:
            assert samples.observations.shape[0] == batch_size
        assert samples.actions.shape == (batch_size, 1)

    if env is not None:
        env.close()

    if not TEST_TRAINING.match(compression_method):
        return

    if space_kind == "dict":
        return

    _run_training_smoke_test(env_id, compression_method, n_envs, n_stack, buffer_type, buffer_dtypes)


def _run_training_smoke_test(
    env_id: str,
    compression_method: str,
    n_envs: int,
    n_stack: int,
    buffer_type: str,
    buffer_dtypes: dict,
):
    env = make_env(env_id=env_id, n_envs=n_envs, vec_env_cls=DummyVecEnv, framestack=n_stack)
    obs = env.observation_space

    if buffer_type == "replay":
        buffer_class = CompressedReplayBuffer
        uncompressed = ReplayBuffer
        model_class = DQN
        extra_args = dict(buffer_size=256)
        expected_dtype = obs.dtype
    elif buffer_type == "rollout":
        buffer_class = CompressedRolloutBuffer
        uncompressed = RolloutBuffer
        model_class = PPO
        extra_args = dict(n_steps=128)
        expected_dtype = obs.dtype if is_sb3_version_gte((2, 7, 1)) else np.float32
    else:
        raise NotImplementedError(f"What is {buffer_type}?")

    if compression_method == "none":
        extra_args[buffer_type + "_buffer_class"] = uncompressed
    else:
        extra_args[buffer_type + "_buffer_class"] = buffer_class
        extra_args[buffer_type + "_buffer_kwargs"] = dict(dtypes=buffer_dtypes, compression_method=compression_method)

    model = model_class(
        "CnnPolicy",
        env,
        verbose=0,
        batch_size=64,
        device="mps" if torch.mps.is_available() else "auto",
        **extra_args,
    )
    model.learn(total_timesteps=256)

    if buffer_type == "replay":
        last_obs = model.replay_buffer.sample(64).observations.cpu().numpy()
    else:
        last_obs = next(model.rollout_buffer.get()).observations.cpu().numpy()

    assert last_obs is not None
    assert last_obs.dtype == expected_dtype

    if DUMP_DEBUG_OBSERVATION:
        dump_dir = f"debug_obs/{buffer_type}"
        os.makedirs(dump_dir, exist_ok=True)
        save_path = f"{dump_dir}/{env_id.split('/')[-1]}_{compression_method}_{n_envs}_{n_stack}.npy"
        if os.path.exists(save_path):
            os.remove(save_path)
        np.save(save_path, last_obs)

    env.close()
