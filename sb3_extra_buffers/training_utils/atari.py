__all__ = ["make_env"]

from typing import Optional

import ale_py
import gymnasium as gym
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env import VecEnvWrapper, VecTransposeImage, VecFrameStack


def make_env(env_id: str, n_envs: int, vec_env_cls: VecEnv = SubprocVecEnv, framestack: int = 4,
             seed: Optional[int] = None, **kwargs) -> VecEnvWrapper:
    gym.register_envs(ale_py)
    if n_envs == 1:
        vec_env_cls = DummyVecEnv
    env = make_atari_env(env_id=env_id, n_envs=n_envs, seed=seed, env_kwargs=kwargs, vec_env_cls=vec_env_cls)
    if framestack > 1:
        env = VecFrameStack(env, n_stack=framestack)
    return VecTransposeImage(env)
