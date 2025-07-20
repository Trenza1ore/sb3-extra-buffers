try:
    from tqdm.rich import tqdm
except ImportError:
    from tqdm import tqdm
import numpy as np
import torch as th
from stable_baselines3 import PPO
from sb3_extra_buffers.compressed import DummyCls
from sb3_extra_buffers.training_utils.atari import make_env
from examples.example_train_atari import ATARI_GAME

NUM_GAMES_TO_WATCH = 10
PAUSE_BETWEEN_GAMES = False
RENDER_GAMES = True

if __name__ == "__main__":
    device = "mps" if th.mps.is_available() else "auto"
    model = PPO.load("logs/MsPacmanNoFrameskip-v4/ppo/best_model/best_model.zip", device=device,
                     custom_objects=dict(rollout_buffer_class=DummyCls))
    render_mode = "human" if RENDER_GAMES else "rgb_array"
    vec_env = make_env(env_id=ATARI_GAME, n_envs=1, framestack=4, render_mode=render_mode)
    obs = vec_env.reset()

    # Play the games
    game_counter = NUM_GAMES_TO_WATCH
    pbar = tqdm(total=NUM_GAMES_TO_WATCH)
    scores = []
    while game_counter > 0:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        game_info = info[0]
        if done and not game_info["lives"]:
            score = game_info["episode"]["r"]
            print(f"Scored: {score}")
            scores.append(score)
            if PAUSE_BETWEEN_GAMES and game_counter > 1:
                input("Click enter when ready for next match: ")
            obs = vec_env.reset()
            game_counter -= 1
            pbar.update()
        if RENDER_GAMES:
            vec_env.render()

    # Closing stuffs
    pbar.close()
    vec_env.close()
    print(f"Average score: {np.mean(scores):.1f} +/- {np.std(scores):.2f}")
