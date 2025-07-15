try:
    from tqdm.rich import tqdm
except ImportError:
    from tqdm import tqdm
import torch as th
from stable_baselines3 import PPO
from examples.make_atari_env import make_env
from examples.example_train_rollout import BEST_MODEL_DIR, ENV_TO_TEST, FRAMESTACK

NUM_GAMES_TO_WATCH = 10
PAUSE_BETWEEN_GAMES = False
RENDER_GAMES = True

if __name__ == "__main__":
    device = "mps" if th.mps.is_available() else "auto"
    model = PPO.load(BEST_MODEL_DIR + "/best_model.zip", device=device)
    render_mode = "human" if RENDER_GAMES else "rgb_array"
    vec_env = make_env(env_id=ENV_TO_TEST, n_envs=1, framestack=FRAMESTACK, render_mode=render_mode)
    obs = vec_env.reset()

    # Play the games
    game_counter = NUM_GAMES_TO_WATCH
    win_count = 0
    scores = [0, 0]
    pbar = tqdm(total=NUM_GAMES_TO_WATCH)
    while game_counter > 0:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        if reward:
            if reward > 0:
                scores[1] += 1
            else:
                scores[0] += 1
        if done:
            print(f"\nMatch result:\nEnemy vs Agent\n{scores[0]:5d} vs {scores[1]:5d}")
            print("Agent won!" if scores[1] > scores[0] else "Agent lost...")
            if scores[1] > scores[0]:
                win_count += 1
            if PAUSE_BETWEEN_GAMES and game_counter > 1:
                input("Click enter when ready for next match: ")
            obs = vec_env.reset()
            game_counter -= 1
            scores = [0, 0]
            pbar.update()
        if RENDER_GAMES:
            vec_env.render()

    # Closing stuffs
    pbar.close()
    vec_env.close()
    print(f"{win_count / NUM_GAMES_TO_WATCH * 100:.1f}% win rate.")
