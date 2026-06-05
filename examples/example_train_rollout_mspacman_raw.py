import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import get_linear_fn

from sb3_extra_buffers.training_utils.atari import make_env

MODEL_TYPE = "ppo"
FRAMESTACK = 4
NUM_ENVS_TRAIN = 8
NUM_ENVS_EVAL = 8
TRAINING_STEPS = 10_000_000
COMPRESSION_METHOD = "zstd-3"
ATARI_GAME = "MsPacmanNoFrameskip-v4"
FINAL_MODEL_PATH = f"./{MODEL_TYPE}_{ATARI_GAME.removesuffix('NoFrameskip-v4')}_{FRAMESTACK}_raw.zip"
BEST_MODEL_DIR = f"./logs/raw/{ATARI_GAME}/{MODEL_TYPE}/best_model"
SEED = 1970626835

if __name__ == "__main__":
    obs = make_env(env_id=ATARI_GAME, n_envs=1, framestack=FRAMESTACK).observation_space
    env = make_env(env_id=ATARI_GAME, n_envs=NUM_ENVS_TRAIN, framestack=FRAMESTACK, seed=SEED)
    if NUM_ENVS_EVAL > 0:
        eval_env = make_env(env_id=ATARI_GAME, n_envs=NUM_ENVS_EVAL, framestack=FRAMESTACK, seed=SEED)
    else:
        eval_env = env

    # Create PPO model
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=get_linear_fn(2.5e-4, 0, 1),
        n_steps=128,
        batch_size=256,
        clip_range=get_linear_fn(0.1, 0, 1),
        n_epochs=4,
        ent_coef=0.01,
        vf_coef=0.5,
        seed=SEED,
        device="mps" if torch.mps.is_available() else "auto",
    )

    # Evaluation callback (optional)
    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=20,
        eval_freq=8192,
        log_path=f"./logs/raw/{ATARI_GAME}/{MODEL_TYPE}/eval",
        best_model_save_path=BEST_MODEL_DIR,
    )

    # Training
    model.learn(total_timesteps=TRAINING_STEPS, callback=eval_callback, progress_bar=True)

    # Save the final model
    model.save(FINAL_MODEL_PATH)

    # Cleanup
    env.close()
    eval_env.close()
