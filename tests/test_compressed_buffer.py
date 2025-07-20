import os
import torch
import pytest
import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.buffers import ReplayBuffer, RolloutBuffer
from sb3_extra_buffers.compressed import CompressedReplayBuffer, CompressedRolloutBuffer, find_buffer_dtypes
from sb3_extra_buffers.training_utils.atari import make_env

ENV_TO_TEST = "MsPacmanNoFrameskip-v4"
STORAGE_DTYPES = dict(elem_type=np.uint8, runs_type=np.uint16)
METHODS_TO_TEST = ["rle", "rle-jit", "gzip", "igzip", "none"]
BATCH_SIZE = 64


def get_tests():
    for compress_method in METHODS_TO_TEST:
        suffix = [""]
        if "igzip" in compress_method:
            suffix = ["0", "3"]
        if "gzip" in compress_method:
            suffix = ["1", "5", "9"]
        for compress_suffix in suffix:
            for n_env in [1, 2]:
                for n_stack in [1, 4]:
                    for buffer_cls in ["replay", "rollout"]:
                        yield (ENV_TO_TEST, compress_method + compress_suffix, n_env, n_stack, buffer_cls)


@pytest.mark.parametrize("env_id,compression_method,n_envs,n_stack,buffer_cls", list(get_tests()))
def test_compressed_buffer(env_id, compression_method: str, n_envs: int, n_stack: int, buffer_cls: str):
    print(f"Testing {(env_id, compression_method, n_envs, n_stack, buffer_cls)}", flush=True)
    obs = make_env(env_id=env_id, n_envs=1, framestack=n_stack).observation_space
    buffer_dtypes = find_buffer_dtypes(obs_shape=obs.shape, elem_dtype=obs.dtype,
                                       compression_method=compression_method)

    env = make_env(env_id=env_id, n_envs=n_envs, framestack=n_stack)
    if buffer_cls == "replay":
        def collect_data(model: DQN):
            return model.replay_buffer.sample(1000).observations.cpu().numpy()
        buffer_class = CompressedReplayBuffer
        uncompressed = ReplayBuffer
        model_class = DQN
        extra_args = dict(buffer_size=1000)
        expected_dtype = np.float32
    elif buffer_cls == "rollout":
        def collect_data(model: PPO):
            return next(model.rollout_buffer.get()).observations.cpu().numpy()
        buffer_class = CompressedRolloutBuffer
        uncompressed = RolloutBuffer
        model_class = PPO
        extra_args = dict()
        expected_dtype = np.float32
    else:
        raise NotImplementedError(f"What is {buffer_cls}?")
    if compression_method != "none":
        extra_args[buffer_cls+"_buffer_class"] = buffer_class
        extra_args[buffer_cls+"_buffer_kwargs"] = dict(dtypes=buffer_dtypes, compression_method=compression_method)
    else:
        extra_args[buffer_cls+"_buffer_class"] = uncompressed

    model = model_class(
        "CnnPolicy",
        env,
        verbose=1,
        batch_size=BATCH_SIZE,
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
    os.makedirs("debug_obs", exist_ok=True)
    save_path = f"debug_obs/{env_id.split('/')[-1]}_{buffer_cls}_{compression_method}_{n_envs}_{n_stack}_last_obs.npy"
    if os.path.exists(save_path):
        os.remove(save_path)
    np.save(save_path, last_obs)
