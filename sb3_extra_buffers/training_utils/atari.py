"""Atari environment factory helpers."""

__all__ = ["make_env"]

from typing import Optional

import ale_py
import gymnasium as gym
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    SubprocVecEnv,
    VecEnv,
    VecEnvWrapper,
    VecFrameStack,
    VecTransposeImage,
)


def make_env(
    env_id: str,
    n_envs: int,
    vec_env_cls: VecEnv = SubprocVecEnv,
    framestack: int = 4,
    seed: Optional[int] = None,
    **kwargs,
) -> VecEnvWrapper:
    """Create a vectorized Atari environment with optional frame stacking.

    Args:
        env_id: Gymnasium environment identifier.
        n_envs: Number of parallel environments.
        vec_env_cls: Vectorized environment class (defaults to subprocess workers).
        framestack: Number of frames to stack; values below 2 disable stacking.
        seed: Optional random seed passed to the environment factory.
        **kwargs: Additional keyword arguments forwarded to ``make_atari_env``.

    Returns:
        A vectorized environment, transposed for channel-first image policies.
    """
    gym.register_envs(ale_py)
    if n_envs == 1:
        vec_env_cls = DummyVecEnv
    env = make_atari_env(
        env_id=env_id,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=kwargs,
        vec_env_cls=vec_env_cls,
    )
    if framestack > 1:
        env = VecFrameStack(env, n_stack=framestack)
    return VecTransposeImage(env)
