import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.callbacks import EvalCallback
from sb3_extra_buffers.compressed import CompressedRolloutBuffer, has_numba, find_smallest_dtype
from examples.make_atari_env import make_env

MODEL_TYPE = "ppo"
FRAMESTACK = 1
NUM_ENVS_TRAIN = 8
NUM_ENVS_EVAL = 8
TRAINING_STEPS = 10_000_000
COMPRESSION_METHOD = "rle-jit"
ENV_TO_TEST = "PongNoFrameskip-v4"
FINAL_MODEL_PATH = f"./{MODEL_TYPE}_{ENV_TO_TEST.removesuffix('NoFrameskip-v4')}_{FRAMESTACK}.zip"
BEST_MODEL_DIR = f"./logs/{ENV_TO_TEST}/{MODEL_TYPE}/best_model"
SEED = 4089164106

if __name__ == "__main__":
    flatten_obs_shape = np.prod(make_env(env_id=ENV_TO_TEST, n_envs=1, framestack=FRAMESTACK).observation_space.shape)
    buffer_dtypes = dict(elem_type=np.uint8, runs_type=find_smallest_dtype(flatten_obs_shape))

    # Pre-JIT Numba to avoid fork issues
    if has_numba() and "jit" in COMPRESSION_METHOD:
        from sb3_extra_buffers.compressed.compression_methods.compression_methods_numba import init_jit
        init_jit(**buffer_dtypes)

    env = make_env(env_id=ENV_TO_TEST, n_envs=NUM_ENVS_TRAIN, framestack=FRAMESTACK, seed=SEED)
    if NUM_ENVS_EVAL > 0:
        eval_env = make_env(env_id=ENV_TO_TEST, n_envs=NUM_ENVS_EVAL, framestack=FRAMESTACK, seed=SEED)
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
        learning_rate=get_linear_fn(2.5e-4, 0.0, 1.0),
        clip_range=get_linear_fn(0.1, 0.0, 1.0),
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        rollout_buffer_class=CompressedRolloutBuffer,
        rollout_buffer_kwargs=dict(dtypes=buffer_dtypes, compression_method=COMPRESSION_METHOD),
        policy_kwargs=dict(normalize_images=False),
        device="mps" if torch.mps.is_available() else "auto",
        seed=SEED,
    )
    env.seed(SEED)
    eval_env.seed(SEED)

    # Evaluation callback (optional)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        log_path=f"./logs/{ENV_TO_TEST}/{MODEL_TYPE}/eval",
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
