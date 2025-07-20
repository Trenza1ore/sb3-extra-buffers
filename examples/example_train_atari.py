from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.callbacks import EvalCallback
from sb3_extra_buffers.compressed import CompressedRolloutBuffer, find_buffer_dtypes
from sb3_extra_buffers.training_utils.atari import make_env

ATARI_GAME = "MsPacmanNoFrameskip-v4"

if __name__ == "__main__":
    obs = make_env(env_id=ATARI_GAME, n_envs=1, framestack=4).observation_space
    compression = "rle-jit"
    buffer_dtypes = find_buffer_dtypes(obs_shape=obs.shape, elem_dtype=obs.dtype, compression_method=compression)

    env = make_env(env_id=ATARI_GAME, n_envs=8, framestack=4)
    eval_env = make_env(env_id=ATARI_GAME, n_envs=10, framestack=4)

    # Create PPO model using CompressedRolloutBuffer
    model = PPO("CnnPolicy", env, verbose=1, learning_rate=get_linear_fn(2.5e-4, 0, 1), n_steps=128,
                batch_size=256, clip_range=get_linear_fn(0.1, 0, 1), n_epochs=4, ent_coef=0.01, vf_coef=0.5,
                seed=1970626835, device="mps", rollout_buffer_class=CompressedRolloutBuffer,
                rollout_buffer_kwargs=dict(dtypes=buffer_dtypes, compression_method=compression))

    # Evaluation callback (optional)
    eval_callback = EvalCallback(eval_env, n_eval_episodes=20, eval_freq=8192, log_path=f"./logs/{ATARI_GAME}/ppo/eval",
                                 best_model_save_path=f"./logs/{ATARI_GAME}/ppo/best_model")

    # Training
    model.learn(total_timesteps=10_000_000, callback=eval_callback, progress_bar=True)

    # Save the final model
    model.save("ppo_MsPacman_4.zip")

    # Cleanup
    env.close()
    eval_env.close()
