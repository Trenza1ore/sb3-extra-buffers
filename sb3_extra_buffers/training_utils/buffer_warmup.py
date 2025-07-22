from typing import Optional, Union, Iterable
from sb3_extra_buffers import ReplayLike, NumberType

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv
from sb3_extra_buffers.training_utils.eval_model import eval_model


def warm_up(buffer: Union[ReplayLike, Iterable[ReplayLike]], n_envs: int, warmup_env: VecEnv,
            warmup_model: BaseAlgorithm, warmup_episodes: Optional[int] = None,
            mean_ep_len: Union[int, float, None] = None) -> tuple[list[NumberType], list[NumberType]]:
    """Perform buffer warm up with set model"""
    if not (isinstance(buffer, Iterable) or (buffer is None)):
        buffer = [buffer]
    # Calculate/validate number of episodes for buffer warm-up
    if not isinstance(warmup_episodes, int) or warmup_episodes < 1:
        if isinstance(mean_ep_len, (int, float)) and mean_ep_len > 0:
            warmup_episodes = round(buffer[0].buffer_size * n_envs * 0.9 / mean_ep_len)
        else:
            raise ValueError("Please provide either warmup_episodes or mean_ep_len.")

    # Check compatibility of vectorized environments
    warmup_n_envs = warmup_env.num_envs
    assert warmup_episodes >= warmup_n_envs, f"Number of environments ({warmup_n_envs}) > episodes ({warmup_episodes})!"
    assert warmup_n_envs % n_envs == 0 and warmup_n_envs >= n_envs, \
        f"warmup_n_envs value ({warmup_n_envs}) incompatible with n_envs ({n_envs})"

    mean_reward, buffer_latency = eval_model(n_eps=warmup_episodes, eval_env=warmup_env, eval_model=warmup_model,
                                             eval_n_envs=warmup_n_envs, buffer_n_envs=n_envs, buffer=buffer)

    return mean_reward, buffer_latency
