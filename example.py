import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from sb3_extra_buffers.compressed import CompressedRolloutBuffer, find_smallest_dtype


env = gym.make("CartPole-v1", render_mode="human")
flatten_obs_shape = np.prod(env.observation_space.shape)
buffer_dtypes = dict(elem_type=np.uint8, runs_type=find_smallest_dtype(flatten_obs_shape))

model = PPO("MlpPolicy", env, verbose=1, rollout_buffer_class=CompressedRolloutBuffer,
            rollout_buffer_kwargs=dict(dtypes=buffer_dtypes, compression_method="rle"))
model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()

env.close()
