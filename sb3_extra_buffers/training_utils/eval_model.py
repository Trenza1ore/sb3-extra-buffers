from typing import Union, Iterable, get_args
from sb3_extra_buffers import ReplayLike, NumberType

import gc
import time
import numpy as np
import torch as th
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv
try:
    from tqdm.rich import tqdm
except ImportError:
    from tqdm import tqdm


def process_outcome(infos: list[dict]) -> tuple[np.ndarray[float], np.ndarray[bool]]:
    nan = float("nan")
    reward = np.asarray([info.get("episode", {}).get("r", nan) for info in infos], dtype=float)
    done = np.zeros_like(reward, dtype="?")
    done[np.isfinite(reward)] = True
    return reward, done


def eval_model(n_eps: int, eval_env: VecEnv, eval_model: BaseAlgorithm, close_env: bool = True,
               buffer: Union[ReplayLike, Iterable[ReplayLike], None] = None) -> tuple[list[NumberType], list]:
    if not (isinstance(buffer, Iterable) or (buffer is None)):
        buffer = [buffer]
    buffer_latency = [0.0] * len(buffer)
    buffer_ok_types = get_args(ReplayLike)
    assert buffer is None or all(isinstance(b, buffer_ok_types) for b in buffer)

    # Prepare for warming up buffer
    pbar = tqdm(total=n_eps, desc=f"Eval ({n_eps})")
    finished_eps_count = 0
    eval_n_envs = eval_env.num_envs
    buffer_n_envs = buffer[0].n_envs if buffer else 1
    reshape_factor = eval_n_envs // buffer_n_envs
    reshape_iter = list(range(reshape_factor))

    done = np.zeros(eval_n_envs, dtype='?')
    obs = eval_env.reset()
    eps_rewards = []

    # Evaluation loop
    while finished_eps_count < n_eps:
        action, _ = eval_model.predict(obs, deterministic=True)
        new_obs, reward, done, info = eval_env.step(action)

        if done.any():
            # Since done can sometimes lie (i.e. MsPacman), do some checking
            real_reward, done = process_outcome(info)
            new_finished_episodes = done.sum()
            finished_eps_count += new_finished_episodes
            pbar.update(new_finished_episodes)
            eps_rewards.extend(real_reward[np.isfinite(real_reward)])

        for b_idx, b in enumerate(buffer):
            t = time.time()
            for i in reshape_iter:
                j = i * buffer_n_envs
                k = j + buffer_n_envs
                b.add(obs[j:k, :, ...], new_obs[j:k, :, ...], action[j:k], reward[j:k], done[j:k], info[j:k])
            buffer_latency[b_idx] += time.time() - t

        obs = new_obs

    # Close warm-up environments
    if close_env:
        pbar.set_description_str("Closing environments")
        eval_env.close()

    # Cache clean-up for compute device
    eval_device_type = eval_model.device.type
    if eval_device_type != "cpu":
        pbar.set_description_str(f"Cleaning {eval_device_type} cache")
        try:
            getattr(th, eval_device_type).empty_cache()
        except Exception as e:
            print(f"Failed to clean cache by calling torch.{eval_device_type}.empty_cache(): {e}")

    # Garbage collection
    pbar.set_description_str("Garbage collection")
    del eval_env, eval_model, obs, new_obs, reward, done, info, action
    gc.collect()
    pbar.set_description_str("Done")
    pbar.close()

    return eps_rewards, buffer_latency
