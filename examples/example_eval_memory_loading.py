import os
import gc
import sys
import time
import platform
from math import ceil
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from sb3_extra_buffers.compressed import CompressedRolloutBuffer, DummyCls
from sb3_extra_buffers.compressed.base import BaseCompressedBuffer
from sb3_extra_buffers.compressed import find_buffer_dtypes
from sb3_extra_buffers.training_utils.atari import make_env
from sb3_extra_buffers.vec_buf import DummyVecRolloutBuffer
from examples.example_train_atari import ATARI_GAME

FINAL_MODEL_PATH = "ppo_MsPacman_4.zip"
FRAMESTACK = 4
N_ENVS = 4
RENDER_GAMES = False
CLEAR_SCREEN = True
BUFFERSIZE = 40_000
COMPRESSION_METHODS = ["none", "rle", "rle-old", "rle-jit",
                       "igzip0", "igzip1", "igzip3", "gzip1", "gzip3",
                       "zstd1", "zstd3", "zstd5", "zstd-1", "zstd-3", "zstd-5",
                       "zstd10", "zstd15", "zstd22", "zstd-20", "zstd-50", "zstd-100",
                       "lz4-frame/1", "lz4-frame/5", "lz4-frame/9", "lz4-frame/12",
                       "lz4-block/1", "lz4-block/5", "lz4-block/9", "lz4-block/16"]

if __name__ == "__main__":
    device = "mps" if th.mps.is_available() else "cuda" if th.cuda.is_available() else "cpu"
    buffer_device = device
    render_mode = "human" if RENDER_GAMES else "rgb_array"
    vec_env = make_env(env_id=ATARI_GAME, n_envs=N_ENVS, framestack=FRAMESTACK, render_mode=render_mode)
    vec_env_obs = vec_env.observation_space
    buffer_dtype = find_buffer_dtypes(vec_env_obs.shape, compression_method="rle-jit")
    if CLEAR_SCREEN:
        os.system("cls" if platform.system() == "Windows" else "clear")

    PER_ENV = ceil(BUFFERSIZE / N_ENVS)
    buffer_config = dict(buffer_size=PER_ENV, observation_space=vec_env.observation_space,
                         action_space=vec_env.action_space, n_envs=vec_env.num_envs,
                         device=buffer_device)
    compression_config = dict(dtypes=buffer_dtype)
    buffer_dict = dict(baseline=RolloutBuffer(**buffer_config))
    buffer_dict.update({
        compression_method: CompressedRolloutBuffer(
            compression_method=compression_method, **buffer_config, **compression_config
        ) for compression_method in COMPRESSION_METHODS
    })

    vec_buffer = DummyVecRolloutBuffer(**buffer_config, buffers=list(buffer_dict.values()))
    model_path = FINAL_MODEL_PATH
    model = PPO.load(FINAL_MODEL_PATH, env=vec_env, device=device, custom_objects=dict(replay_buffer_class=DummyCls))
    _, callback = model._setup_learn(model._total_timesteps, progress_bar=True)
    callback.on_training_start(dict(total_timesteps=BUFFERSIZE), {})
    model.collect_rollouts(env=vec_env, callback=callback, rollout_buffer=vec_buffer,
                           n_rollout_steps=PER_ENV)
    callback._on_training_end()
    del vec_buffer, model
    try:
        getattr(th, device).empty_cache()
    except Exception as e:
        print(e)
    gc.collect()
    time.sleep(10)

    base_size = -1
    sort_dict = dict()
    save_dir = f"debug_obs/load_eval/{ATARI_GAME}"
    os.makedirs(save_dir, exist_ok=True)
    for k, v in list(buffer_dict.items()):
        raw_size = sys.getsizeof(v.observations)
        buffer = np.ravel(v.observations)
        if isinstance(v, BaseCompressedBuffer):
            raw_size += sum(sys.getsizeof(b) for b in buffer)
            size_vs_base = raw_size / base_size * 100
        else:
            base_size = int(raw_size)
            size_vs_base = 100.0

        # Display the size of buffer in reasonable unit
        size = raw_size
        size_str = "ERRORR"
        for size_unit in "KMGTP":
            size /= 1024
            if size < 1024:
                if size < 100:
                    size_str = f"{size:4.1f}{size_unit}B" if size > 10 else f"{size:4.2f}{size_unit}B"
                else:
                    size_str = f"{round(size):4d}{size_unit}B"
                break

        # Prepare content for printing
        pos = int(v.pos)
        if k == "baseline":
            print(f"{pos} steps for each env, {4*pos} steps in total.")

        t = time.time_ns()
        [x for x in v.get(batch_size=64)]
        ll = (time.time_ns() - t) / 1000_000

        sort_dict[f"| {k:15s} | {size_str} | {size_vs_base:5.1f}% | {ll:6.1f}ms |"] = (ll, size_vs_base)
        del buffer_dict[k], buffer, v
        gc.collect()

    # Print out formatted table
    print(f"Device: {device}")
    print(f"|{'Compression':^17s}|{'Memory':^8s}|{'Memory %':^8s}|{'Latency':^10s}|")
    print(f"|{'-'*17}|{'-'*8}|{'-'*8}|{'-'*10}|")
    sorted_list = sorted(sort_dict.items(), key=lambda x: x[1][1])
    for k, v in sorted(sorted_list, key=lambda x: x[1][0]):
        print(k)
