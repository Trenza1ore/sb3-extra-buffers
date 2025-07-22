[![PyPI - Version](https://img.shields.io/pypi/v/sb3-extra-buffers)](https://pypi.org/project/sb3-extra-buffers/) [![Pepy Total Downloads](https://img.shields.io/pepy/dt/sb3-extra-buffers)](https://pepy.tech/projects/sb3-extra-buffers) ![PyPI - License](https://img.shields.io/pypi/l/sb3-extra-buffers) ![PyPI - Implementation](https://img.shields.io/pypi/implementation/sb3-extra-buffers?style=flat)

# sb3-extra-buffers
Unofficial implementation of extra Stable-Baselines3 buffer classes. Aims to reduce memory usage drastically with minimal overhead. Featured in [SB3 docs](https://stable-baselines3.readthedocs.io/en/master/misc/projects.html#sb3-extra-buffers-ram-expansions-are-overrated-just-compress-your-observations) :-)

![Banner Image](https://github.com/user-attachments/assets/e6e5cd2f-55d4-4686-abf7-773148d80ad2)


**Links:**
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [SB3 Contrib (experimental features for SB3)](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib)
- [SBX (SB3 + JAX, uses SB3 buffers so can also benefit from compressed buffers here)](https://github.com/araffin/sbx)
- [RL Baselines3 Zoo (training framework for SB3)](https://github.com/DLR-RM/rl-baselines3-zoo)

**Description:**
Tired of reading a cool RL paper and realizing that the author is storing a **MILLION** observations in their replay buffers? Yeah me too. This project has implemented several compressed buffer classes that replace Stable Baselines3's standard buffers like ReplayBuffer and RolloutBuffer. With as simple as 2-5 lines of extra code and **negligible overhead**, memory usage can be reduced by more than **95%**!

**Main Goal:**
Reduce the memory consumption of memory buffers in Reinforcement Learning while adding minimal overhead.

**Current Progress & Available Features:**
- Memory Saving: [reported here](#memory-usage-test-for-compressed-buffers-on-mspacmannoframeskip-v4)
- Progress Tracker Issue: https://github.com/Trenza1ore/sb3-extra-buffers/issues/1

**Motivation:**
Reinforcement Learning is quite memory-hungry due to massive buffer sizes, so let's try to tackle it by not storing raw frame buffers in full `np.float32` or `np.uint8` directly and find something smaller instead. For any input data that are sparse and containing large contiguous region of repeating values, lossless compression techniques can be applied to reduce memory footprint.

**Applicable Input Types:**
- `Semantic Segmentation` masks (1 color channel)
- `Color Palette` game frames from retro video games
- `Grayscale` observations
- `RGB (Color)` observations
- For noisy input with a lot of variation (mostly `RGB`), using `zstd` is recommended, run-length encoding won't work as great and can potentially even increase memory usage. [See benchmark](#memory-usage-test-for-compressed-buffers-on-mspacmannoframeskip-v4).

**Implemented Compression Methods:**
- `none` No compression other than casting to `elem_type` and storing as `bytes`.
- `rle` Vectorized Run-Length Encoding for compression.
- `rle-jit` JIT-compiled version of `rle`, uses [numba](https://numba.pydata.org) library.
- `gzip` Built-in gzip compression via `gzip`.
- `igzip` Intel accelerated variant via `isal.igzip`, uses [python-isal](https://github.com/pycompression/python-isal) library.
- **`zstd`** Zstandard compression via [python-zstd](https://github.com/sergey-dryabzhinsky/python-zstd). **(Recommended)**
- `lz4-frame` LZ4 (frame format) compression via [python-lz4](https://github.com/python-lz4/python-lz4).
- `lz4-block` LZ4 (block format) compression via [python-lz4](https://github.com/python-lz4/python-lz4).

> - `gzip` supports `0~9` compression levels, `0` is no compression, `1` is least compression
> - `igzip` supports `0~3` compression levels, `0` is least compression
> - `zstd` supports `1~22` standard compression levels and `-100~-1` ultra-fast compression levels, `-100` is fastest and `22` is slowest.
> - `lz4-frame` supports `0~16` standard compression levels and negative levels translates into acceleration factor.
> - `lz4-block` supports three modes, split into positive/zero/negative compression levels. `1~12` are in `high_compression` mode and negative levels translates into acceleration factor in `fast` mode, setting `0` enables `default` mode.
> - Shorthands are supported (for `lz4` methods including `/` is required):
>   - `pattern` = `^((?:[A-Za-z]+)|(?:[\w\-]+/))(\-?[0-9]+)$`
>   - `igzip3` = `igzip/3` = `igzip level 3`
>   - `zstd-5` = `zstd/-5` = `zstd level -5`
>   - `lz4-frame/5` = `lz4-frame level 5`

## Installation
Install via PyPI:
```bash
pip install "sb3-extra-buffers[fast,extra]"
```
Other install options:
```bash
pip install "sb3-extra-buffers"          # only installs minimum requirements
pip install "sb3-extra-buffers[extra]"   # installs extra dependencies for SB3
pip install "sb3-extra-buffers[fast]"    # installs python-isal, numba, zstd, lz4
pip install "sb3-extra-buffers[isal]"    # only installs python-isal
pip install "sb3-extra-buffers[numba]"   # only installs numba
pip install "sb3-extra-buffers[zstd]"    # only installs python-zstd
pip install "sb3-extra-buffers[lz4]"     # only installs python-lz4
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
### Benchmark for Compressed Buffers (on `MsPacmanNoFrameskip-v4`)
- **Frame Stack & Vec Envs**: both 4
- **Buffer Size**: 40,000 (split across 4 vectorized environments)
- **Loading Test**: Sample all trajectories from rollout buffers with batch size of 64, target device: `mps`. SB3's `RolloutBuffer` stores `np.float32` observations so it's 4x the size of `np.uint8`.
- **Settings**: The [example DQN / PPO model](#Example-Scripts) loaded and evaluated using the code in [examples](https://github.com/Trenza1ore/sb3-extra-buffers/blob/main/examples/), DQN for saving test, PPO for loading test. The **exact same** observations are stored into each buffer for fairness. `Latency` refers to the total number of seconds spent on adding observation to the specific buffer and `baseline` refers to using `ReplayBuffer` directly.
- **TLDR**:
  - `zstd` in general is very decent at save latency & memory saving, personally I recommend **`zstd-3`**.
  - `zstd-1` ~ `zstd-5` seems to be the sweet spot.
  - `gzip0` should be avoided, saving / loading has similar latency as `zstd-5`, but 13x bigger.
  - MsPacman at `84x84` resolution is too visually noisy for `rle` , although decompression isn't half-bad

| Compression     | Save Mem | Save Mem % | Save Latency | Load Mem | Load Mem % | Load Latency (s) |
|-----------------|----------|------------|---------------|----------|-------------|------------------|
| baseline        | 1.05GB   | 100.0%     | 0.9           | 4.21GB   | 100.0%      | 5.21             |
| none            | 1.05GB   | 100.1%     | 1.2           | 1.05GB   | 25.0%       | 8.70             |
| zstd-100        | 387MB    | 36.0%      | 1.8           | 413MB    | 9.6%        | 9.08             |
| zstd-50         | 306MB    | 28.4%      | 1.9           | 326MB    | 7.6%        | 8.95             |
| zstd-5          | 82.9MB   | 7.7%       | 2.1           | 89.1MB   | 2.1%        | 8.80             |
| lz4-frame/1     | 118MB    | 10.9%      | 2.1           | 127MB    | 2.9%        | 8.86             |
| zstd-20         | 181MB    | 16.8%      | 2.2           | 189MB    | 4.4%        | 8.91             |
| zstd-3          | 73.9MB   | 6.9%       | 2.3           | 78.7MB   | 1.8%        | 8.81             |
| zstd-1          | 66.0MB   | 6.1%       | 2.3           | 70.0MB   | 1.6%        | 8.79             |
| zstd1           | 61.3MB   | 5.7%       | 2.7           | 64.7MB   | 1.5%        | 8.90             |
| zstd3           | 59.4MB   | 5.5%       | 3.0           | 63.1MB   | 1.5%        | 8.91             |
| igzip0          | 129MB    | 12.0%      | 3.4           | 136MB    | 3.1%        | 9.60             |
| rle             | 811MB    | 75.3%      | 4.0           | 849MB    | 19.7%       | 14.7             |
| rle-jit         | 811MB    | 75.3%      | 4.0           | 849MB    | 19.7%       | 9.10             |
| rle-old         | 811MB    | 75.3%      | 4.0           | 849MB    | 19.7%       | 104              |
| lz4-block/1     | 83.2MB   | 7.7%       | 4.6           | 89.8MB   | 2.1%        | 8.73             |
| igzip1          | 114MB    | 10.6%      | 5.0           | 121MB    | 2.8%        | 9.66             |
| zstd5           | 55.9MB   | 5.2%       | 5.4           | 59.3MB   | 1.4%        | 8.90             |
| lz4-block/5     | 75.1MB   | 7.0%       | 6.3           | 80.1MB   | 1.9%        | 8.76             |
| lz4-frame/5     | 75.9MB   | 7.0%       | 6.5           | 80.8MB   | 1.9%        | 8.72             |
| gzip1           | 104MB    | 9.6%       | 7.6           | 108MB    | 2.5%        | 9.75             |
| gzip3           | 81.9MB   | 7.6%       | 8.3           | 85.9MB   | 2.0%        | 9.44             |
| igzip3          | 81.5MB   | 7.6%       | 10.5          | 87.0MB   | 2.0%        | 9.59             |
| zstd10          | 52.8MB   | 4.9%       | 10.8          | 56.5MB   | 1.3%        | 8.89             |
| lz4-block/9     | 72.0MB   | 6.7%       | 20.0          | 76.9MB   | 1.8%        | 8.69             |
| lz4-frame/9     | 72.7MB   | 6.8%       | 20.0          | 77.6MB   | 1.8%        | 8.74             |
| lz4-block/16    | 71.3MB   | 6.6%       | 57.9          | 76.2MB   | 1.8%        | 8.69             |
| lz4-frame/12    | 72.0MB   | 6.7%       | 58.4          | 77.0MB   | 1.8%        | 8.77             |
| zstd15          | 48.5MB   | 4.5%       | 99.8          | 52.0MB   | 1.2%        | 8.86             |
| zstd22          | 47.6MB   | 4.4%       | 590.7         | 51.0MB   | 1.2%        | 8.96             |



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
