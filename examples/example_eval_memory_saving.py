import os
import gc
import sys
import platform
import numpy as np
import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from sb3_extra_buffers.compressed import CompressedReplayBuffer, DummyCls
from sb3_extra_buffers.compressed import find_buffer_dtypes
from sb3_extra_buffers.training_utils.buffer_warmup import eval_model
from sb3_extra_buffers.training_utils.atari import make_env
from examples.example_train_replay import FINAL_MODEL_PATH, ATARI_GAME, FRAMESTACK

N_EVAL_EPISODES = 50
N_ENVS = 4
RENDER_GAMES = False
CLEAR_SCREEN = True
BUFFERSIZE = 40_000
COMPRESSION_METHODS = ["none", "rle",
                       "igzip0", "igzip1", "igzip3", "gzip0", "gzip1", "gzip3",
                       "zstd1", "zstd3", "zstd5", "zstd-1", "zstd-3", "zstd-5",
                       "zstd10", "zstd15", "zstd22", "zstd-20", "zstd-50", "zstd-100",
                       "lz4-frame/1", "lz4-frame/5", "lz4-frame/9", "lz4-frame/12",
                       "lz4-block/1", "lz4-block/5", "lz4-block/9", "lz4-block/16"]

if __name__ == "__main__":
    device = "mps" if th.mps.is_available() else "auto"
    render_mode = "human" if RENDER_GAMES else "rgb_array"
    vec_env = make_env(env_id=ATARI_GAME, n_envs=N_ENVS, framestack=FRAMESTACK, render_mode=render_mode)
    vec_env_obs = vec_env.observation_space
    buffer_dtype = find_buffer_dtypes(vec_env_obs.shape, compression_method="rle-jit")
    if CLEAR_SCREEN:
        os.system("cls" if platform.system() == "Windows" else "clear")

    buffer_config = dict(buffer_size=BUFFERSIZE, observation_space=vec_env.observation_space,
                         action_space=vec_env.action_space, n_envs=vec_env.num_envs,
                         optimize_memory_usage=True, handle_timeout_termination=False)
    compression_config = dict(dtypes=buffer_dtype, output_dtype="raw")
    buffer_dict = dict(baseline=ReplayBuffer(**buffer_config))
    buffer_dict.update({
        compression_method: CompressedReplayBuffer(
            compression_method=compression_method, **buffer_config, **compression_config
        ) for compression_method in COMPRESSION_METHODS
    })

    all_buffers = list(buffer_dict.values())

    model_path = FINAL_MODEL_PATH
    model = DQN.load(FINAL_MODEL_PATH, device=device, custom_objects=dict(replay_buffer_class=DummyCls))
    eval_rewards, buffer_latency = eval_model(N_EVAL_EPISODES, vec_env, model, close_env=True, buffer=all_buffers)
    Q1, Q2, Q3 = (round(np.percentile(eval_rewards, x)) for x in [25, 50, 75])
    reward_avg, reward_std = np.mean(eval_rewards), np.std(eval_rewards)
    reward_min, reward_max = round(np.min(eval_rewards)), round(np.max(eval_rewards))
    relative_IQR = (Q3 - Q1) / Q2
    print(f"Evaluated {N_EVAL_EPISODES} episodes, mean reward: {reward_avg:.1f} +/- {reward_std:.2f}")
    print(f"Q1: {Q1:4d} | Q2: {Q2:4d} | Q3: {Q3:4d} | Relative IQR: {relative_IQR:4.2f}", end=" | ")
    print(f"Min: {reward_min} | Max: {reward_max}")
    del all_buffers

    base_size = -1
    sort_dict = dict()
    save_dir = f"debug_obs/size_eval/{ATARI_GAME}"
    os.makedirs(save_dir, exist_ok=True)
    for (k, v), l in zip(list(buffer_dict.items()), buffer_latency):
        raw_size = sys.getsizeof(v.observations)
        buffer = np.ravel(v.observations)
        if isinstance(v, CompressedReplayBuffer):
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
        assert v.full, f"Buffer not filled! pos: {v.pos}"
        pos = int(v.pos + v.observations.shape[0])
        if k == "baseline":
            print(f"{pos} steps for each env, {4*pos} steps in total.")
        # else:
        #     np.save(f"{save_dir}/{k.replace('/', '-')}.npy", buffer)
        del buffer_dict[k], buffer, v
        gc.collect()
        sort_dict[f"| {k:15s} | {size_str} | {size_vs_base:5.1f}% | {l:7.1f}s |"] = l

    # Print out formatted table
    print(f"|{'Compression':^17s}|{'Memory':^8s}|{'Memory %':^8s}|{'Latency':^10s}|")
    print(f"|{'-'*17}|{'-'*8}|{'-'*8}|{'-'*10}|")
    for k, v in sorted(sort_dict.items(), key=lambda x: x[1]):
        print(k)
