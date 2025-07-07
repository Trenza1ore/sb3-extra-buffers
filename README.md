# sb3-extra-buffers
Unofficial implementation of extra Stable-Baselines3 buffer classes, still a proof-of-concept in the current state.

**Main Goal:**
Reduce the memory consumption of memory buffers in Reinforcement Learning.

**Motivation:**
Reinforcement Learning is quite memory-hungry due to massive buffer sizes, so let's try to tackle it by not storing raw frame buffers in full `np.float32` directly and find something smaller instead. For any input data that are sparse and containing large contiguous region of repeating values, lossless compression techniques can be applied to reduce memory footprint.

**Applicable Input Types:**
- `Semantic Segmentation` masks (1 color channel)
- `Color Palette` game frames from retro video games
- `Grayscale` game frames from retro video games
## Installation
To install with `isal` and `numba` support:
```bash
pip install "sb3_extra_buffers[fast]"
```
Other install options:
```bash
pip install sb3_extra_buffers            # only installs minimum requirements
pip install "sb3_extra_buffers[isal]"    # only installs python-isal
pip install "sb3_extra_buffers[numba]"   # only installs numba
pip install "sb3_extra_buffers[atari]"   # installs gymnasium, ale-py
pip install "sb3_extra_buffers[vizdoom]" # installs gymnasium, vizdoom
```
## Project Structure
```
sb3_extra_buffers
    |- compressed
    |    |- CompressedRolloutBuffer: RolloutBuffer with compression
    |    |- CompressedReplayBuffer: ReplayBuffer with compression
    |
    |- recording
         |- RecordBuffer: A buffer for recording game states
         |- FramelessRecordBuffer: RecordBuffer but not recording game frames
         |- DummyRecordBuffer: Dummy RecordBuffer, records nothing
```
---
## Compressed Buffers
Defined in `sb3_extra_buffers.compressed`

**Implemented Compression Methods:**
- `rle` Uses Run-Length Encoding for compression.
- `rle-jit` JIT-compiled version of `rle`, uses `numba` library.
- `gzip` Compression via `gzip`.
- `igzip` Compression via `isal.igzip`, uses `python-isal` library.
- `none` No compression other than casting to `elem_type`.

**JIT Before Multi-Processing:**
When using `rle-jit`, remember to trigger JIT compilation before any multi-processing code is executed.
```python
# Code for other stuffs...
from sb3_extra_buffers.compressed.compression_methods import HAS_NUMBA

# Compressed-buffer-related settings
compression_method = "rle-jit"
storage_dtypes = dict(elem_type=np.uint8, runs_type=np.uint16)

# Pre-JIT Numba to avoid fork issues
if HAS_NUMBA and "jit" in compression_method:
    from sb3_extra_buffers.compressed.compression_methods.compression_methods_numba import init_jit
    init_jit(**storage_dtypes)

# Now, safe to initialize multi-processing environments!
env = SubprocVecEnv([make_env for _ in range(4)])
```

**Example Usage:**
```python
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
```
---
## Recording Buffers
Defined in `sb3_extra_buffers.recording`
Mainly used in combination with [SegDoom](https://github.com/Trenza1ore/SegDoom) to record stuff.
#### WIP
