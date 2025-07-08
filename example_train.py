import torch
import ale_py
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecTransposeImage
from sb3_extra_buffers.compressed import CompressedRolloutBuffer, HAS_NUMBA, find_smallest_dtype
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

COMPRESSION_METHOD = "rle-jit"
TRAIN_NENV = 8
EVAL_NENV = 8
ENV_TO_TEST = "ALE/Pong-v5"


def make_env(env_id: str = ENV_TO_TEST, **kwargs):
    if env_id.startswith("ALE/"):
        gym.register_envs(ale_py)
    env = gym.make(env_id, **kwargs)
    env = Monitor(env)
    env = AtariWrapper(env)
    return env


if __name__ == "__main__":
    flatten_obs_shape = np.prod(make_env().observation_space.shape)
    buffer_dtypes = dict(elem_type=np.uint8, runs_type=find_smallest_dtype(flatten_obs_shape))

    # Pre-JIT Numba to avoid fork issues
    if HAS_NUMBA and "jit" in COMPRESSION_METHOD:
        from sb3_extra_buffers.compressed.compression_methods.compression_methods_numba import init_jit
        init_jit(**buffer_dtypes)

    env = SubprocVecEnv([make_env for _ in range(TRAIN_NENV)])
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)

    eval_env = SubprocVecEnv([make_env for _ in range(EVAL_NENV)])
    eval_env = VecFrameStack(eval_env, n_stack=4)
    eval_env = VecTransposeImage(eval_env)

    # Create PPO model using CompressedRolloutBuffer
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        n_steps=128,
        batch_size=256,
        gae_lambda=0.9,
        gamma=0.99,
        n_epochs=4,
        ent_coef=0.01,
        learning_rate=2.5e-4,
        clip_range=0.1,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        rollout_buffer_class=CompressedRolloutBuffer,
        rollout_buffer_kwargs=dict(dtypes=buffer_dtypes, compression_method=COMPRESSION_METHOD, normalize_images=False),
        policy_kwargs=dict(normalize_images=False),
        device="mps" if torch.mps.is_available() else "auto"
    )

    # Evaluation callback (optional)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model",
        log_path="./logs/eval",
        n_eval_episodes=20,
        eval_freq=4096,
        deterministic=True,
        render=False,
    )

    # Training
    model.learn(total_timesteps=10_000_000, callback=eval_callback, progress_bar=True)

    # Save the final model
    model.save("ppo_pong.zip")

    # Cleanup
    env.close()
    eval_env.close()
