import os
import platform
import numpy as np
import torch as th
from stable_baselines3 import PPO
from sb3_extra_buffers.compressed import DummyCls
from sb3_extra_buffers.training_utils.eval_model import eval_model
from sb3_extra_buffers.training_utils.atari import make_env
from examples.example_train_atari import ATARI_GAME

N_EVAL_EPISODES = 10_000
N_ENVS = 50
RENDER_GAMES = False
CLEAR_SCREEN = True

if __name__ == "__main__":
    device = "mps" if th.mps.is_available() else "auto"
    render_mode = "human" if RENDER_GAMES else "rgb_array"
    vec_env = make_env(env_id=ATARI_GAME, n_envs=N_ENVS, framestack=4, render_mode=render_mode)
    if CLEAR_SCREEN:
        os.system("cls" if platform.system() == "Windows" else "clear")
    for model_path in ["logs/MsPacmanNoFrameskip-v4/ppo/best_model/best_model.zip", "ppo_MsPacman_4.zip"]:
        model = PPO.load(model_path, device=device, custom_objects=dict(rollout_buffer_class=DummyCls))
        eval_rewards = eval_model(N_EVAL_EPISODES, vec_env, model, close_env=False)
        Q1, Q2, Q3 = (round(np.percentile(eval_rewards, x)) for x in [25, 50, 75])
        reward_avg, reward_std = np.mean(eval_rewards), np.std(eval_rewards)
        reward_min, reward_max = round(np.min(eval_rewards)), round(np.max(eval_rewards))
        relative_IQR = (Q3 - Q1) / Q2
        print(model_path)
        print(f"Evaluated {N_EVAL_EPISODES} episodes, mean reward: {reward_avg:.1f} +/- {reward_std:.2f}")
        print(f"Q1: {Q1:4d} | Q2: {Q2:4d} | Q3: {Q3:4d} | Relative IQR: {relative_IQR:4.2f}", end=" | ")
        print(f"Min: {reward_min} | Max: {reward_max}")
