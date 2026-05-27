import torch
from stable_baselines3 import DQN

from sb3_extra_buffers.training_utils.atari import make_env

MODEL_TYPE = "dqn"
FRAMESTACK = 4
NUM_ENVS_TRAIN = 1
NUM_ENVS_EVAL = 8
BUFFER_SIZE = 1000
TRAINING_STEPS = 1000
EXPLORATION_STEPS = 1000
LEARNING_STARTS = 100_000
ATARI_GAME = "MsPacmanNoFrameskip-v4"
SEED = 1809550766


if __name__ == "__main__":
    obs = make_env(env_id=ATARI_GAME, n_envs=1, framestack=FRAMESTACK).observation_space

    env = make_env(
        env_id=ATARI_GAME, n_envs=NUM_ENVS_TRAIN, framestack=FRAMESTACK, seed=SEED
    )
    if NUM_ENVS_EVAL > 0:
        eval_env = make_env(
            env_id=ATARI_GAME, n_envs=NUM_ENVS_EVAL, framestack=FRAMESTACK, seed=SEED
        )
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
        exploration_fraction=EXPLORATION_STEPS / TRAINING_STEPS,
        exploration_final_eps=0.01,
        learning_rate=1e-4,
        device="mps" if torch.mps.is_available() else "auto",
        seed=SEED,
    )

    # Save the final model
    model.load_replay_buffer("old_replay.pkl")
    print(model.replay_buffer.observations.shape)
    print(model.replay_buffer.dtypes)

    # Cleanup
    env.close()
    eval_env.close()
