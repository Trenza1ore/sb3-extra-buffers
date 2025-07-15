import os
import torch
import pytest
import numpy as np
from stable_baselines3 import DQN
from example_train_replay import make_env, ENV_TO_TEST
from sb3_extra_buffers.compressed import CompressedReplayBuffer, find_smallest_dtype
from sb3_extra_buffers.compressed.compression_methods import has_numba

STORAGE_DTYPES = dict(elem_type=np.uint8, runs_type=np.uint16)
METHODS_TO_TEST = ["rle", "rle-jit", "gzip", "igzip", "none"]
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


@pytest.mark.parametrize("env_id,compression_method,n_envs", get_tests())
def test_compressed_replay_buffer(env_id, compression_method: str, n_envs: int):
    storage_dtypes = STORAGE_DTYPES.copy()
    flatten_len = int(np.prod(make_env(n_envs=1).observation_space.shape))
    storage_dtypes["runs_type"] = find_smallest_dtype(flatten_len, signed=False)

    if has_numba() and "jit" in compression_method:
        # Pre-JIT Numba to avoid fork issues
        from sb3_extra_buffers.compressed.compression_methods.compression_methods_numba import init_jit
        init_jit(**storage_dtypes)

    env = make_env(n_envs=n_envs)

    policy_kwargs = {
        "normalize_images": False
    }

    # Create PPO model using CompressedRolloutBuffer
    seed_num = 1234
    env.seed(seed_num)
    torch.manual_seed(seed_num)
    extra_args = dict(
        replay_buffer_class=CompressedReplayBuffer,
        replay_buffer_kwargs=dict(dtypes=STORAGE_DTYPES, compression_method=compression_method)
    )
    if compression_method == "none":
        extra_args.clear()
    model = DQN(
        "CnnPolicy",
        env,
        verbose=1,
        batch_size=BATCH_SIZE,
        buffer_size=1000,
        policy_kwargs=policy_kwargs,
        device="mps" if torch.mps.is_available() else "auto",
        seed=seed_num,
        **extra_args
    )

    # Train briefly
    model.learn(total_timesteps=256)

    # Retrieve latest observation from PPO
    last_obs = model.replay_buffer.sample(1000).observations.cpu().numpy()

    # Check basic properties
    assert last_obs is not None, "No observations stored"
    # assert last_obs.dtype == np.float32, f"Expected float32 observations, got {last_obs.dtype}"

    # Dump to disk for manual inspection
    os.makedirs("debug_obs", exist_ok=True)
    save_path = f"debug_obs/{env_id.split('/')[-1]}_{n_envs}_replay_{compression_method}_last_obs.npy"
    if os.path.exists(save_path):
        os.remove(save_path)
    np.save(save_path, last_obs)


if __name__ == "__main__":
    test_compressed_replay_buffer(ENV_TO_TEST, "rle", 1)
