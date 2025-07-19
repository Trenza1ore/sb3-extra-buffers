from typing import Optional, Union

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import VecEnv
from sb3_extra_buffers.training_utils.eval_model import eval_model


def warm_up(buffer: ReplayBuffer, n_envs: int, warmup_env: VecEnv, warmup_model: BaseAlgorithm,
            warmup_episodes: Optional[int] = None, mean_ep_len: Union[int, float, None] = None) -> float:
    """Perform buffer warm up with set model"""

    # Calculate/validate number of episodes for buffer warm-up
    if not isinstance(warmup_episodes, int) or warmup_episodes < 1:
        if isinstance(mean_ep_len, (int, float)) and mean_ep_len > 0:
            warmup_episodes = round(buffer.buffer_size * n_envs * 0.9 / mean_ep_len)
        else:
            raise ValueError("Please provide either warmup_episodes or mean_ep_len.")

    # Check compatibility of vectorized environments
    warmup_n_envs = warmup_env.num_envs
    assert warmup_episodes >= warmup_n_envs, f"Number of environments ({warmup_n_envs}) > episodes ({warmup_episodes})!"
    assert warmup_n_envs % n_envs == 0 and warmup_n_envs >= n_envs, \
        f"warmup_n_envs value ({warmup_n_envs}) incompatible with n_envs ({n_envs})"

    mean_reward = eval_model(n_eps=warmup_episodes, eval_env=warmup_env, eval_model=warmup_model,
                             eval_n_envs=warmup_n_envs, buffer_n_envs=n_envs, buffer=buffer)

    return mean_reward
