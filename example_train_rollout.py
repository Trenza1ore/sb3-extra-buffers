import torch
import ale_py
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack, VecTransposeImage, VecEnv
from sb3_extra_buffers.compressed import CompressedRolloutBuffer, has_numba, find_smallest_dtype

MODEL_TYPE = "ppo"
NUM_ENVS_TRAIN = 8
NUM_ENVS_EVAL = 8
FRAMESTACK = 4
TRAINING_STEPS = 10_000_000
COMPRESSION_METHOD = "rle-jit"
ENV_TO_TEST = "ALE/Pong-v5"
FINAL_MODEL_PATH = f"./{MODEL_TYPE}_pong.zip"


def make_env(env_id: str = ENV_TO_TEST, n_envs: int = NUM_ENVS_TRAIN, framestack: int = FRAMESTACK,
             vec_env_cls: VecEnv = SubprocVecEnv, **kwargs):
    if env_id.startswith("ALE/"):
        gym.register_envs(ale_py)
    if n_envs == 1:
        vec_env_cls = DummyVecEnv
    env = make_atari_env(env_id=env_id, n_envs=n_envs, env_kwargs=kwargs, vec_env_cls=vec_env_cls)
    if framestack > 1:
        env = VecFrameStack(env, n_stack=framestack)
    return VecTransposeImage(env)


if __name__ == "__main__":
    flatten_obs_shape = np.prod(make_env().observation_space.shape)
    buffer_dtypes = dict(elem_type=np.uint8, runs_type=find_smallest_dtype(flatten_obs_shape))

    # Pre-JIT Numba to avoid fork issues
    if has_numba() and "jit" in COMPRESSION_METHOD:
        from sb3_extra_buffers.compressed.compression_methods.compression_methods_numba import init_jit
        init_jit(**buffer_dtypes)

    env = make_env(env_id=ENV_TO_TEST, n_envs=NUM_ENVS_TRAIN, framestack=FRAMESTACK)
    if NUM_ENVS_EVAL > 0:
        eval_env = make_env(env_id=ENV_TO_TEST, n_envs=NUM_ENVS_EVAL, framestack=FRAMESTACK)
    else:
        eval_env = env

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
        best_model_save_path=f"./logs/{MODEL_TYPE}/best_model",
        log_path=f"./logs/{MODEL_TYPE}/eval",
        n_eval_episodes=20,
        eval_freq=4096,
        deterministic=True,
        render=False,
    )

    # Training
    model.learn(total_timesteps=TRAINING_STEPS, callback=eval_callback, progress_bar=True)

    # Save the final model
    model.save(FINAL_MODEL_PATH)

    # Cleanup
    env.close()
    eval_env.close()
