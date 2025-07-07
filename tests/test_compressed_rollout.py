import os
import torch
import pytest
import ale_py
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecFrameStack
from sb3_extra_buffers.compressed import CompressedRolloutBuffer, find_smallest_dtype
from sb3_extra_buffers.compressed.compression_methods import HAS_NUMBA
from tests.feat_extract import CustomCNN

STORAGE_DTYPES = dict(elem_type=np.uint8, runs_type=np.uint16)
METHODS_TO_TEST = ["rle", "rle-jit", "gzip", "igzip", "none"]
ENV_TO_TEST = "ALE/Pong-v5"
BATCH_SIZE = 64


def get_tests():
    all_enums = []
    for compress_method in METHODS_TO_TEST:
        suffix = [""]
        if "igzip" in compress_method:
            suffix = ["0", "3"]
        if "gzip" in compress_method:
            suffix = ["1", "5", "9"]
        for compress_suffix in suffix:
            for n_env in [1, 2]:
                for n_stack in [1, 4]:
                    all_enums.append((ENV_TO_TEST, compress_method + compress_suffix, n_env, n_stack))
    return all_enums


@pytest.mark.parametrize("env_id,compression_method,n_envs,n_stack", get_tests())
def test_compressed_rollout_buffer(env_id, compression_method: str, n_envs: int, n_stack: int):
    def make_env():
        gym.register_envs(ale_py)
        return gym.make(env_id, obs_type="grayscale")

    storage_dtypes = STORAGE_DTYPES.copy()
    flatten_len = int(np.prod(make_env().observation_space.shape)) * n_stack
    storage_dtypes["runs_type"] = find_smallest_dtype(flatten_len, signed=False)

    if HAS_NUMBA and "jit" in compression_method:
        # Pre-JIT Numba to avoid fork issues
        from sb3_extra_buffers.compressed.compression_methods.compression_methods_numba import init_jit
        init_jit(**storage_dtypes)

    # Set up SubprocVecEnv with 2 grayscale Atari environments
    if n_envs > 1:
        env = SubprocVecEnv([make_env for _ in range(n_envs)])
    else:
        env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=n_stack)

    cnn_cls = CustomCNN
    policy_kwargs = {
        "normalize_images": False,
        "features_extractor_class": cnn_cls,
        "features_extractor_kwargs": {"features_dim": 128}
    }

    # Create PPO model using CompressedRolloutBuffer
    seed_num = 1234
    torch.manual_seed(seed_num)
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        n_steps=128,
        batch_size=BATCH_SIZE,
        rollout_buffer_class=CompressedRolloutBuffer,
        rollout_buffer_kwargs=dict(dtypes=STORAGE_DTYPES, compression_method=compression_method),
        policy_kwargs=policy_kwargs,
        device="mps" if torch.mps.is_available() else "auto",
        seed=seed_num
    )

    # Train briefly
    model.learn(total_timesteps=256)

    # Retrieve latest observation from PPO
    last_obs = next(model.rollout_buffer.get()).observations.cpu().numpy()

    # Check basic properties
    assert last_obs is not None, "No observations stored"
    assert last_obs.dtype == np.float32, f"Expected float32 observations, got {last_obs.dtype}"

    # Dump to disk for manual inspection
    os.makedirs("debug_obs", exist_ok=True)
    save_path = f"debug_obs/{env_id.split('/')[-1]}_{n_envs}_{n_stack}_{compression_method}_last_obs.npy"
    if os.path.exists(save_path):
        os.remove(save_path)
    np.save(save_path, last_obs)


if __name__ == "__main__":
    test_compressed_rollout_buffer(ENV_TO_TEST, "rle", 1, 4)
