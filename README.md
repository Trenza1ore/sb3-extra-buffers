[![PyPI version](https://badge.fury.io/py/sb3-extra-buffers.svg?icon=si%3Apython&icon_color=%2300faca)](https://badge.fury.io/py/sb3-extra-buffers) [![PyPI Downloads](https://static.pepy.tech/badge/sb3-extra-buffers)](https://pepy.tech/projects/sb3-extra-buffers)

# sb3-extra-buffers
Unofficial implementation of extra Stable-Baselines3 buffer classes. Aims to reduce memory usage drastically with minimal overhead.

**Links:**
- [This Project on PyPI](https://pypi.org/project/sb3-extra-buffers/)
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [SB3 Contrib (experimental features for SB3)](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib)
- [SBX (SB3 + JAX, uses SB3 buffers so can also benefit from compressed buffers here)](https://github.com/araffin/sbx)
- [RL Baselines3 Zoo (training framework for SB3)](https://github.com/DLR-RM/rl-baselines3-zoo)

**Main Goal:**
Reduce the memory consumption of memory buffers in Reinforcement Learning while adding minimal overhead.

**TO-DO List:**
- [x] Compressed Rollout / Replay Buffer
- [ ] Compressed variant for every buffer in SB3
- [ ] Compressed variant for every buffer in SB3-Contrib
- [x] Compressed Array, maybe can make porting easier?
- [x] Recording Buffers for game episodes
- [ ] Compressed Recording Buffers
- [x] Buffer warm-up and model evaluation utility functions
- [x] Example Atari train / eval scripts with compressed buffers
- [x] Report results for example Atari train / eval scripts
- [ ] Example ViZDoom train / eval scripts with compressed buffers
- [ ] Report results for example ViZDoom train / eval scripts
- [ ] Report memory saving
- [ ] Documentation & better readme
- [x] Define a standard bytes-out (compress) bytes-in (decompress) interface and store compressed obs in `np.ndarray[bytes]`

**Motivation:**
Reinforcement Learning is quite memory-hungry due to massive buffer sizes, so let's try to tackle it by not storing raw frame buffers in full `np.float32` or `np.uint8` directly and find something smaller instead. For any input data that are sparse and containing large contiguous region of repeating values, lossless compression techniques can be applied to reduce memory footprint.

**Applicable Input Types:**
- `Semantic Segmentation` masks (1 color channel)
- `Color Palette` game frames from retro video games
- `Grayscale` game frames from retro video games
- `RGB (Color)` game frames from retro video games
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
pip install "sb3-extra-buffers"            # only installs minimum requirements
pip install "sb3-extra-buffers[extra]"   # installs extra dependencies for SB3
pip install "sb3-extra-buffers[fast]"    # installs python-isal and numba
pip install "sb3-extra-buffers[isal]"    # only installs python-isal
pip install "sb3-extra-buffers[numba]"   # only installs numba
pip install "sb3-extra-buffers[vizdoom]" # installs vizdoom
```
## Current Project Structure
```
sb3_extra_buffers
    |- compressed
    |    |- CompressedRolloutBuffer: RolloutBuffer with compression
    |    |- CompressedReplayBuffer: ReplayBuffer with compression
    |    |- CompressedArray: Compressed numpy.ndarray subclass
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
**DQN on `MsPacmanNoFrameskip-v4`, trained for 10M steps using `rle-jit`, framestack: `4`**
```
(Best ) Evaluated 10000 episodes, mean reward: 3300.0 +/- 770.79
Q1: 2490 | Q2: 4020 | Q3: 4020 | Relative IQR: 0.38 | Min: 2460 | Max: 4020
(Final) Evaluated 10000 episodes, mean reward: 3379.2 +/- 453.78
Q1: 2690 | Q2: 3400 | Q3: 3880 | Relative IQR: 0.35 | Min: 1230 | Max: 4090
```
---
## Compressed Buffers
Defined in `sb3_extra_buffers.compressed`

**JIT Before Multi-Processing:**
When using `rle-jit`, remember to trigger JIT compilation before any multi-processing code is executed.
```python
# Code for other stuffs...
from sb3_extra_buffers.compressed.compression_methods import has_numba

# Compressed-buffer-related settings
compression_method = "rle-jit"
storage_dtypes = dict(elem_type=np.uint8, runs_type=np.uint16)

# Pre-JIT Numba to avoid fork issues
if has_numba() and "jit" in compression_method:
    from sb3_extra_buffers.compressed.compression_methods.compression_methods_numba import init_jit
    init_jit(**storage_dtypes)

# Now, safe to initialize multi-processing environments!
env = SubprocVecEnv(...)
```

**Example Usage:**
```python
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from sb3_extra_buffers.compressed import CompressedRolloutBuffer, find_smallest_dtype
from sb3_extra_buffers.training_utils.atari import make_env

ATARI_GAME = "MsPacmanNoFrameskip-v4"

if __name__ == "__main__":
    flatten_obs_shape = np.prod(make_env(env_id=ATARI_GAME, n_envs=1, framestack=4).observation_space.shape)
    buffer_dtypes = dict(elem_type=np.uint8, runs_type=find_smallest_dtype(flatten_obs_shape))

    # Make vec envs
    env = make_env(env_id=ATARI_GAME, n_envs=4, framestack=4)
    eval_env = make_env(env_id=ATARI_GAME, n_envs=10, framestack=4)

    # Create PPO model using CompressedRolloutBuffer
    model = PPO("CnnPolicy", env, verbose=1, n_steps=128, batch_size=256, n_epochs=4,
                rollout_buffer_class=CompressedRolloutBuffer,
                rollout_buffer_kwargs=dict(dtypes=buffer_dtypes, compression_method="rle"), device="mps")

    # Evaluation callback (optional)
    eval_callback = EvalCallback(eval_env, n_eval_episodes=10, eval_freq=8192, log_path=f"./logs/{ATARI_GAME}",
                                 best_model_save_path=f"./logs/{ATARI_GAME}/best_model")

    # Training
    model.learn(total_timesteps=1_000_000, callback=eval_callback, progress_bar=True)

    # Save the final model
    model.save(f"ppo-{ATARI_GAME}.zip")

    # Cleanup
    env.close()
    eval_env.close()
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
