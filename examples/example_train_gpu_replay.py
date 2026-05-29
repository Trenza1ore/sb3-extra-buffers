import torch as th
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback

from sb3_extra_buffers.gpu_buffers import GpuReplayBuffer, find_gpu_buffer_dtypes
from sb3_extra_buffers.gpu_buffers.utils import numpy_dtype_to_torch
from sb3_extra_buffers.training_utils.atari import make_env


def resolve_device() -> str:
    """Pick the best available Torch device for policy and buffer storage."""
    if th.mps.is_available():
        return "mps"
    if th.cuda.is_available():
        return "cuda"
    return "cpu"


MODEL_TYPE = "dqn"
FRAMESTACK = 4
NUM_ENVS_TRAIN = 1
NUM_ENVS_EVAL = 8
BUFFER_SIZE = 100_000
TRAINING_STEPS = 10_000_000
EXPLORATION_STEPS = 1_000_000
LEARNING_STARTS = 100_000
COMPRESSION_METHOD = "rle-jit"
ATARI_GAME = "MsPacmanNoFrameskip-v4"
FINAL_MODEL_PATH = f"./{MODEL_TYPE}_{ATARI_GAME.removesuffix('NoFrameskip-v4')}_{FRAMESTACK}_{resolve_device()}.zip"
BEST_MODEL_DIR = f"./logs/{resolve_device()}/{ATARI_GAME}/{MODEL_TYPE}/best_model"
SEED = 1809550766

if __name__ == "__main__":
    device = resolve_device()
    buffer_device = device

    obs = make_env(env_id=ATARI_GAME, n_envs=1, framestack=FRAMESTACK).observation_space
    buffer_dtypes = find_gpu_buffer_dtypes(
        obs_shape=obs.shape,
        elem_dtype=numpy_dtype_to_torch(obs.dtype),
        compression_method=COMPRESSION_METHOD,
    )

    env = make_env(env_id=ATARI_GAME, n_envs=NUM_ENVS_TRAIN, framestack=FRAMESTACK, seed=SEED)
    if NUM_ENVS_EVAL > 0:
        eval_env = make_env(env_id=ATARI_GAME, n_envs=NUM_ENVS_EVAL, framestack=FRAMESTACK, seed=SEED)
    else:
        eval_env = env

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
        exploration_fraction=EXPLORATION_STEPS / TRAINING_STEPS,
        exploration_final_eps=0.01,
        learning_rate=1e-4,
        replay_buffer_class=GpuReplayBuffer,
        replay_buffer_kwargs=dict(
            dtypes=buffer_dtypes,
            compression_method=COMPRESSION_METHOD,
            buffer_device=buffer_device,
        ),
        policy_kwargs=dict(normalize_images=False),
        device=device,
        seed=SEED,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        log_path=f"./logs/gpu/{ATARI_GAME}/{MODEL_TYPE}/eval",
        n_eval_episodes=20,
        eval_freq=8192,
        deterministic=True,
        render=False,
    )

    model.learn(total_timesteps=TRAINING_STEPS, callback=eval_callback, progress_bar=True)
    model.save(FINAL_MODEL_PATH)

    env.close()
    eval_env.close()
