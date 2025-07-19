import torch
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from sb3_extra_buffers.compressed import CompressedReplayBuffer, has_numba, find_smallest_dtype
from sb3_extra_buffers.training_utils.atari import make_env

MODEL_TYPE = "dqn"
FRAMESTACK = 4
NUM_ENVS_TRAIN = 1
NUM_ENVS_EVAL = 8
BUFFER_SIZE = 100_000
TRAINING_STEPS = 10_000_000
EXPLORATION_STEPS = 1_000_000
LEARNING_STARTS = 100_000
COMPRESSION_METHOD = "rle-jit"
ENV_TO_TEST = "MsPacmanNoFrameskip-v4"
FINAL_MODEL_PATH = f"./{MODEL_TYPE}_{ENV_TO_TEST.removesuffix('NoFrameskip-v4')}_{FRAMESTACK}.zip"
BEST_MODEL_DIR = f"./logs/{ENV_TO_TEST}/{MODEL_TYPE}/best_model"
SEED = 1809550766


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

    # Create DQN model using CompressedRolloutBuffer
    model = DQN(
        "CnnPolicy",
        env,
        verbose=1,
        buffer_size=BUFFER_SIZE,
        batch_size=32,
        learning_starts=LEARNING_STARTS,
        target_update_interval=1000,
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=EXPLORATION_STEPS/TRAINING_STEPS,
        exploration_final_eps=0.01,
        learning_rate=1e-4,
        replay_buffer_class=CompressedReplayBuffer,
        replay_buffer_kwargs=dict(dtypes=buffer_dtypes, compression_method=COMPRESSION_METHOD),
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
        eval_freq=8192,
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
