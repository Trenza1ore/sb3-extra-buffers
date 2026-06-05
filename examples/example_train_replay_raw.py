import torch
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback

from sb3_extra_buffers.training_utils.atari import make_env

MODEL_TYPE = "dqn"
FRAMESTACK = 4
NUM_ENVS_TRAIN = 1
NUM_ENVS_EVAL = 8
BUFFER_SIZE = 100_000
TRAINING_STEPS = 10_000_000
EXPLORATION_STEPS = 1_000_000
LEARNING_STARTS = 100_000
ATARI_GAME = "MsPacmanNoFrameskip-v4"
FINAL_MODEL_PATH = f"./{MODEL_TYPE}_{ATARI_GAME.removesuffix('NoFrameskip-v4')}_{FRAMESTACK}_raw.zip"
BEST_MODEL_DIR = f"./logs/raw/{ATARI_GAME}/{MODEL_TYPE}/best_model"
SEED = 1809550766


if __name__ == "__main__":
    obs = make_env(env_id=ATARI_GAME, n_envs=1, framestack=FRAMESTACK).observation_space
    env = make_env(env_id=ATARI_GAME, n_envs=NUM_ENVS_TRAIN, framestack=FRAMESTACK, seed=SEED)
    if NUM_ENVS_EVAL > 0:
        eval_env = make_env(env_id=ATARI_GAME, n_envs=NUM_ENVS_EVAL, framestack=FRAMESTACK, seed=SEED)
    else:
        eval_env = env

    # Create DQN model
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
        device="mps" if torch.mps.is_available() else "auto",
        seed=SEED,
    )

    # Evaluation callback (optional)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        log_path=f"./logs/raw/{ATARI_GAME}/{MODEL_TYPE}/eval",
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
