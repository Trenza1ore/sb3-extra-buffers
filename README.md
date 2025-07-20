[![PyPI - Version](https://img.shields.io/pypi/v/sb3-extra-buffers)](https://pypi.org/project/sb3-extra-buffers/) [![Pepy Total Downloads](https://img.shields.io/pepy/dt/sb3-extra-buffers)](https://pepy.tech/projects/sb3-extra-buffers) ![PyPI - License](https://img.shields.io/pypi/l/sb3-extra-buffers) ![PyPI - Implementation](https://img.shields.io/pypi/implementation/sb3-extra-buffers?style=flat)

# sb3-extra-buffers
Unofficial implementation of extra Stable-Baselines3 buffer classes. Aims to reduce memory usage drastically with minimal overhead.

![SB3-Extra-Buffers Banner](https://github.com/user-attachments/assets/20e59c8f-d593-4107-8219-f8db0667d0a6)

**Links:**
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [SB3 Contrib (experimental features for SB3)](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib)
- [SBX (SB3 + JAX, uses SB3 buffers so can also benefit from compressed buffers here)](https://github.com/araffin/sbx)
- [RL Baselines3 Zoo (training framework for SB3)](https://github.com/DLR-RM/rl-baselines3-zoo)

**Main Goal:**
Reduce the memory consumption of memory buffers in Reinforcement Learning while adding minimal overhead.

**Current Progress & Available Features:**
See Issue https://github.com/Trenza1ore/sb3-extra-buffers/issues/1

**Motivation:**
Reinforcement Learning is quite memory-hungry due to massive buffer sizes, so let's try to tackle it by not storing raw frame buffers in full `np.float32` or `np.uint8` directly and find something smaller instead. For any input data that are sparse and containing large contiguous region of repeating values, lossless compression techniques can be applied to reduce memory footprint.

**Applicable Input Types:**
- `Semantic Segmentation` masks (1 color channel)
- `Color Palette` game frames from retro video games
- `Grayscale` observations
- `RGB (Color)` observations
- For noisy input with a lot of variation (mostly `RGB`), using `gzip1` or `igzip0` is recommended, run-length encoding won't work as great and can potentially even increase memory usage.

**Implemented Compression Methods:**
- `rle` Vectorized Run-Length Encoding for compression.
- `rle-jit` JIT-compiled version of `rle`, uses `numba` library.
- `gzip` Gzip compression via `gzip`. 
- `igzip` Intel accelerated variant via `isal.igzip`, uses `python-isal` library.
- `none` No compression other than casting to `elem_type` and storing as `bytes`.

> - `gzip` supports `0-9` compress levels, `0` is no compression, `1` is least compression
> - `igzip` supports `0-3` compress levels, `0` is least compression
> - Shorthands are supported, i.e. `igzip3` = `igzip` at level `3`

## Installation
Install via PyPI:
```bash
pip install "sb3-extra-buffers[fast,extra]"
```
Other install options:
```bash
pip install "sb3-extra-buffers"          # only installs minimum requirements
pip install "sb3-extra-buffers[extra]"   # installs extra dependencies for SB3
pip install "sb3-extra-buffers[fast]"    # installs python-isal and numba
pip install "sb3-extra-buffers[isal]"    # only installs python-isal
pip install "sb3-extra-buffers[numba]"   # only installs numba
pip install "sb3-extra-buffers[vizdoom]" # installs vizdoom
```

## Example Usage
```python
from stable_baselines3 import PPO
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.callbacks import EvalCallback
from sb3_extra_buffers.compressed import CompressedRolloutBuffer, find_buffer_dtypes
from sb3_extra_buffers.training_utils.atari import make_env

ATARI_GAME = "MsPacmanNoFrameskip-v4"

if __name__ == "__main__":
    # Get the most suitable dtypes for CompressedRolloutBuffer to use
    obs = make_env(env_id=ATARI_GAME, n_envs=1, framestack=4).observation_space
    compression = "rle-jit"  # or use "igzip1" since it's relatively noisy
    buffer_dtypes = find_buffer_dtypes(obs_shape=obs.shape, elem_dtype=obs.dtype, compression_method=compression)

    # Create vectorized environments after the find_buffer_dtypes call, which initializes jit
    env = make_env(env_id=ATARI_GAME, n_envs=8, framestack=4)
    eval_env = make_env(env_id=ATARI_GAME, n_envs=10, framestack=4)

    # Create PPO model with CompressedRolloutBuffer as rollout buffer class
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
```

## Current Project Structure
```
sb3_extra_buffers
    |- compressed
    |    |- CompressedRolloutBuffer: RolloutBuffer with compression
    |    |- CompressedReplayBuffer: ReplayBuffer with compression
    |    |- CompressedArray: Compressed numpy.ndarray subclass
    |    |- find_buffer_dtypes: Find suitable buffer dtypes and initialize jit
    |
    |- recording
    |    |- RecordBuffer: A buffer for recording game states
    |    |- FramelessRecordBuffer: RecordBuffer but not recording game frames
    |    |- DummyRecordBuffer: Dummy RecordBuffer, records nothing
    |
    |- training_utils
         |- eval_model: Evaluate models in vectorized environment
         |- warmup: Perform buffer warmup for off-policy algorithms
```
## Example Scripts
[Example scripts](https://github.com/Trenza1ore/sb3-extra-buffers/tree/main/examples) have been included and tested to ensure working properly. 
#### Evaluation results for example training scripts:
**PPO on `PongNoFrameskip-v4`, trained for 10M steps using `rle-jit`, framestack: `None`**
```
(Best ) Evaluated 10000 episodes, mean reward: 21.0 +/- 0.00
Q1:   21 | Q2:   21 | Q3:   21 | Relative IQR: 0.00 | Min: 21 | Max: 21
(Final) Evaluated 10000 episodes, mean reward: 21.0 +/- 0.02
Q1:   21 | Q2:   21 | Q3:   21 | Relative IQR: 0.00 | Min: 20 | Max: 21
```
**PPO on `MsPacmanNoFrameskip-v4`, trained for 10M steps using `rle-jit`, framestack: `4`**
```
(Best ) Evaluated 10000 episodes, mean reward: 2667.0 +/- 290.00
Q1: 2300 | Q2: 2490 | Q3: 3000 | Relative IQR: 0.28 | Min: 2300 | Max: 3000
(Final) Evaluated 10000 episodes, mean reward: 2500.9 +/- 221.03
Q1: 2300 | Q2: 2390 | Q3: 2490 | Relative IQR: 0.08 | Min: 1420 | Max: 3000
```
**DQN on `MsPacmanNoFrameskip-v4`, trained for 10M steps using `rle-jit`, framestack: `4`**
```
(Best ) Evaluated 10000 episodes, mean reward: 3300.0 +/- 770.79
Q1: 2490 | Q2: 4020 | Q3: 4020 | Relative IQR: 0.38 | Min: 2460 | Max: 4020
(Final) Evaluated 10000 episodes, mean reward: 3379.2 +/- 453.78
Q1: 2690 | Q2: 3400 | Q3: 3880 | Relative IQR: 0.35 | Min: 1230 | Max: 4090
```
---
## Pytest
Make sure `pytest` and optionally `pytest-xdist` are already installed. Tests are compatible with `pytest-xdist` since `DummyVecEnv` is used for all tests.
```
# pytest
pytest tests -v --durations=0 --tb=short
# pytest-xdist
pytest tests -n auto -v --durations=0 --tb=short
```
---
## Compressed Buffers
Defined in `sb3_extra_buffers.compressed`

**JIT Before Multi-Processing:**
When using `rle-jit`, remember to trigger JIT compilation before any multi-processing code is executed via  `find_buffer_dtypes` or `init_jit`.
```python
# Code for other stuffs...

# Get observation space from environment
obs = make_env(env_id=ATARI_GAME, n_envs=1, framestack=4).observation_space

# Get the buffer datatype settings via find_buffer_dtypes
compression = "rle-jit"
buffer_dtypes = find_buffer_dtypes(obs_shape=obs.shape, elem_dtype=obs.dtype, compression_method=compression)

# Now, safe to initialize multi-processing environments!
env = SubprocVecEnv(...)
```
---
## Recording Buffers
Defined in `sb3_extra_buffers.recording`
Mainly used in combination with [SegDoom](https://github.com/Trenza1ore/SegDoom) to record stuff.
#### WIP
---
## Training Utils
Defined in `sb3_extra_buffers.training_utils`
Buffer warm-up and model evaluation
#### WIP
