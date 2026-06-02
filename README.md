[![PyPI - Version](https://img.shields.io/pypi/v/sb3-extra-buffers)](https://pypi.org/project/sb3-extra-buffers/) [![Pepy Total Downloads](https://img.shields.io/pepy/dt/sb3-extra-buffers?uuid=12076e13275f4891a087b22b9c031a9e)](https://pepy.tech/projects/sb3-extra-buffers) [![PyPI - License](https://img.shields.io/pypi/l/sb3-extra-buffers)](https://github.com/Trenza1ore/sb3-extra-buffers/blob/main/LICENSE) [![Tests](https://github.com/Trenza1ore/sb3-extra-buffers/actions/workflows/tests.yml/badge.svg)](https://github.com/Trenza1ore/sb3-extra-buffers/actions/workflows/tests.yml)

[![Documentation](https://img.shields.io/badge/Documentation-brown?style=for-the-badge&logo=readthedocs&link=https%3A%2F%2Fcompressed-rl.readthedocs.io%2Fen%2Flatest%2Findex.html)](https://compressed-rl.readthedocs.io)

# sb3-extra-buffers
Unofficial implementation of extra Stable-Baselines3 buffer classes. Aims to reduce memory usage drastically with minimal overhead. Featured in [SB3 docs](https://stable-baselines3.readthedocs.io/en/master/misc/projects.html#sb3-extra-buffers-ram-expansions-are-overrated-just-compress-your-observations) :-)

![Banner Image](https://github.com/user-attachments/assets/e6e5cd2f-55d4-4686-abf7-773148d80ad2)

**Links:**
- [My Master's Thesis (which started this as a side project)](https://arxiv.org/abs/2511.11703)
- [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3)
- [SB3 Contrib (experimental features for SB3)](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib)
- [SBX (SB3 + JAX, uses SB3 buffers so can also benefit from compressed buffers here)](https://github.com/araffin/sbx)
- [RL Baselines3 Zoo (training framework for SB3)](https://github.com/DLR-RM/rl-baselines3-zoo)

**Description:**
Tired of reading a cool RL paper and realizing that the author is storing a **MILLION** observations in their replay buffers? Yeah me too. This project has implemented several compressed buffer classes that replace Stable Baselines3's standard buffers like ReplayBuffer and RolloutBuffer. With as simple as 2-5 lines of extra code and **negligible overhead**, memory usage can be reduced by more than **95%**! Please consider [citing this project](#cite-this-project) if this is helpful to your research!

**Main Goal:**
Reduce the memory consumption of memory buffers in Reinforcement Learning while adding minimal overhead.

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

**Current Progress & Available Features:**
- Memory Saving: [reported here](https://compressed-rl.readthedocs.io/latest/benchmarks.html)
- Training Speed: [reported here](https://compressed-rl.readthedocs.io/latest/speed.html)
- Validation Experiments: [reported here](https://compressed-rl.readthedocs.io/latest/validation.html)
- Progress Tracker Issue: https://github.com/Trenza1ore/sb3-extra-buffers/issues/1

**Motivation:**
Reinforcement Learning is quite memory-hungry due to massive buffer sizes, so let's try to tackle it by not storing raw frame buffers in full `np.float32` or `np.uint8` directly and find something smaller instead. For any input data that are sparse and containing large contiguous region of repeating values, lossless compression techniques can be applied to reduce memory footprint.

**Applicable Input Types:**
- `Semantic Segmentation` masks (1 color channel)
- `Color Palette` game frames from retro video games
- `Grayscale` observations
- `RGB (Color)` observations
- For noisy input with a lot of variation (mostly `RGB`), using `zstd` is recommended, run-length encoding won't work as great and can potentially even increase memory usage. [See benchmark](https://compressed-rl.readthedocs.io/latest/benchmarks.html).

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
                seed=1970626835, rollout_buffer_class=CompressedRolloutBuffer,
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
    |    |- CompressedDictRolloutBuffer: DictRolloutBuffer with compression
    |    |- CompressedDictReplayBuffer: DictReplayBuffer with compression
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
## Notes

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
## Cite This Project

If you use this project in your research or work, please cite:

```bibtex
@article{Huang2025EnhancingRL,
  title={Enhancing Reinforcement Learning in 3D Environments through Semantic Segmentation: A Case Study in ViZDoom},
  author={Jin Huang},
  journal={ArXiv},
  year={2025},
  volume={abs/2511.11703},
  url={https://arxiv.org/abs/2511.11703},
}
```
I really appreciate it~
