from functools import partial
from concurrent.futures import ThreadPoolExecutor, wait
from tqdm.rich import tqdm
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage
from example_train import make_env

NUM_GAMES_TO_WATCH = 100
PAUSE_BETWEEN_GAMES = False
RENDER_GAMES = False

if __name__ == "__main__":
    model = PPO.load("ppo_pong.zip")

    make_env_render = partial(make_env, render_mode="human")
    vec_env = DummyVecEnv([make_env_render if RENDER_GAMES else make_env])
    vec_env = VecFrameStack(vec_env, n_stack=4)
    vec_env = VecTransposeImage(vec_env)
    obs = vec_env.reset()

    game_counter = NUM_GAMES_TO_WATCH
    win_rate = 0
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
                win_rate += 1
            if PAUSE_BETWEEN_GAMES and game_counter > 1:
                input("Click enter when ready for next match: ")
            obs = vec_env.reset()
            game_counter -= 1
            scores = [0, 0]
            pbar.update()
        if RENDER_GAMES:
            vec_env.render()
    vec_env.close()
    print(f"{win_rate / NUM_GAMES_TO_WATCH * 100:.1f}% win rate.")
