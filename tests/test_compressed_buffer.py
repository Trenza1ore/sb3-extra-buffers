import os
import pytest
import torch
import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.buffers import ReplayBuffer, RolloutBuffer
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_extra_buffers.compressed import CompressedReplayBuffer, CompressedRolloutBuffer, find_buffer_dtypes
from sb3_extra_buffers.training_utils.atari import make_env

ENV_TO_TEST = ["MsPacmanNoFrameskip-v4", "PongNoFrameskip-v4"]


def get_tests():
    for compress_method in ["rle", "rle-old", "rle-jit", "gzip", "igzip", "none"]:
        suffix = [""]
        if "igzip" in compress_method:
            suffix = ["0", "3"]
        if "gzip" in compress_method:
            suffix = ["1", "5", "9"]
        for compress_suffix in suffix:
            for n_stack in [1, 4]:
                for env_id in ENV_TO_TEST:
                    for n_env in [1, 2]:
                        yield (env_id, compress_method + compress_suffix, n_env, n_stack)


@pytest.mark.parametrize("env_id,compression_method,n_envs,n_stack", list(get_tests()))
def test_rollout(env_id: str, compression_method: str, n_envs: int, n_stack: int):
    compressed_buffer_test(env_id, compression_method, n_envs, n_stack, buffer_type="rollout")


@pytest.mark.parametrize("env_id,compression_method,n_envs,n_stack", list(get_tests()))
def test_replay(env_id: str, compression_method: str, n_envs: int, n_stack: int):
    compressed_buffer_test(env_id, compression_method, n_envs, n_stack, buffer_type="replay")


def compressed_buffer_test(env_id: str, compression_method: str, n_envs: int, n_stack: int, buffer_type: str):
    print(f"Testing {(env_id, compression_method, n_envs, n_stack, buffer_type)}", flush=True)

    obs = make_env(env_id=env_id, n_envs=1, framestack=n_stack).observation_space
    buffer_dtypes = find_buffer_dtypes(obs_shape=obs.shape, elem_dtype=obs.dtype,
                                       compression_method=compression_method)

    # if using pytest, avoid using SubprocVecEnv
    env = make_env(env_id=env_id, n_envs=n_envs, vec_env_cls=DummyVecEnv, framestack=n_stack)
    if buffer_type == "replay":
        def collect_data(model: DQN):
            return model.replay_buffer.sample(1000).observations.cpu().numpy()
        buffer_class = CompressedReplayBuffer
        uncompressed = ReplayBuffer
        model_class = DQN
        extra_args = dict(buffer_size=1000)
        expected_dtype = np.float32
        uncompressed_dtype = obs.dtype
    elif buffer_type == "rollout":
        def collect_data(model: PPO):
            return next(model.rollout_buffer.get()).observations.cpu().numpy()
        buffer_class = CompressedRolloutBuffer
        uncompressed = RolloutBuffer
        model_class = PPO
        extra_args = dict(n_steps=384)
        expected_dtype = np.float32
        uncompressed_dtype = expected_dtype
    else:
        raise NotImplementedError(f"What is {buffer_type}?")
    if compression_method == "none":
        extra_args[buffer_type+"_buffer_class"] = uncompressed
        expected_dtype = uncompressed_dtype
    else:
        extra_args[buffer_type+"_buffer_class"] = buffer_class
        extra_args[buffer_type+"_buffer_kwargs"] = dict(dtypes=buffer_dtypes, compression_method=compression_method)

    model = model_class(
        "CnnPolicy",
        env,
        verbose=1,
        batch_size=64,
        device="mps" if torch.mps.is_available() else "auto",
        **extra_args
    )

    # Train briefly
    model.learn(total_timesteps=1000)

    # Retrieve latest observation from PPO
    last_obs = collect_data(model)

    # Check basic properties
    assert last_obs is not None, "No observations stored"
    assert last_obs.dtype == expected_dtype, f"Expected {expected_dtype} observations, got {last_obs.dtype}"

    # Dump to disk for manual inspection
    dump_dir = f"debug_obs/{buffer_type}"
    os.makedirs(dump_dir, exist_ok=True)
    save_path = f"{dump_dir}/{env_id.split('/')[-1]}_{compression_method}_{n_envs}_{n_stack}.npy"
    if os.path.exists(save_path):
        os.remove(save_path)
    np.save(save_path, last_obs)
