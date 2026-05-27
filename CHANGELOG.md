# Changelog

## [0.4.5](https://github.com/Trenza1ore/sb3-extra-buffers/releases/tag/v0.4.5) — 2026-05-27

Changes since `0.4.4`…`0.4.5`.

### Feature

- Added compressed version for `DictReplayBuffer` and `DictRolloutBuffer`.

### Breaking Changes

- `CompressedReplayBuffer` now has `output_dtype` defaulting to `raw` instead of `float`.
- `CompressedRolloutBuffer` now returns observation with environment's `dtype` to exhibit same behavior as [v2.7.1](https://github.com/DLR-RM/stable-baselines3/releases/tag/v2.7.1) of SB3. (When pre-2.7.1 SB3 is installed, we still cast it to `torch.float32` for backward-compatibility~)

---

## [0.4.4](https://github.com/Trenza1ore/sb3-extra-buffers/releases/tag/v0.4.4) — 2026-05-27

Changes since `0.4.3`…`0.4.4`.

### Changes by Human

- README: ArXiv link, BibTeX citation, and CI tests badge
- Added GitHub Actions CI (Python 3.9–3.13), and trusted PyPI publishing on release
- Formatted code w/ `ruff` (black + isort)

### By Copilot

- Fix Python 3.12+ f-string syntax and replace print with logger by @Copilot in [https://github.com/Trenza1ore/sb3-extra-buffers/pull/5](https://github.com/Trenza1ore/sb3-extra-buffers/pull/5)
- Modernize packaging with pyproject.toml and add CI workflow by @Copilot in [https://github.com/Trenza1ore/sb3-extra-buffers/pull/4](https://github.com/Trenza1ore/sb3-extra-buffers/pull/4)
- Refactor logging configuration and type definitions into dedicated modules by @Copilot in [https://github.com/Trenza1ore/sb3-extra-buffers/pull/6](https://github.com/Trenza1ore/sb3-extra-buffers/pull/6)
- Make np.save optional in tests via environment variable by @Copilot in [https://github.com/Trenza1ore/sb3-extra-buffers/pull/7](https://github.com/Trenza1ore/sb3-extra-buffers/pull/7)
- Add SB3 version-based dtype handling for rollout buffer tests by @Copilot in [https://github.com/Trenza1ore/sb3-extra-buffers/pull/8](https://github.com/Trenza1ore/sb3-extra-buffers/pull/8)

---

## [0.4.3](https://github.com/Trenza1ore/sb3-extra-buffers/releases/tag/v0.4.3) — 2025-08-06

Changes since `0.4.1`…`0.4.3`.

- Removed f-string formatting specific to Python 3.12 (breaks 3.9-3.11 support)
- Added py.typed (to provide type-checking support with tools like `mypy`)

---

## [0.4.1](https://github.com/Trenza1ore/sb3-extra-buffers/releases/tag/v0.4.1) — 2025-07-23

Changes since `0.4.0`…`0.4.1`.

### Benchmark for Compressed Buffers (on `MsPacmanNoFrameskip-v4`)

- **Frame Stack & Vec Envs**: both 4
- **Buffer Size**: 40,000 (split across 4 vectorized environments)
- **Loading Test**: Sample all trajectories from rollout buffers with batch size of 64, target device: `mps`. SB3's `RolloutBuffer` stores `np.float32` observations so it's 4x the size of `np.uint8`.
- **Settings**: The example DQN / PPO model loaded and evaluated using the code in [examples](https://github.com/Trenza1ore/sb3-extra-buffers/blob/main/examples/), DQN for saving test, PPO for loading test. The **exact same** observations are stored into each buffer for fairness. `Latency` refers to the total number of seconds spent on adding observation to the specific buffer and `baseline` refers to using `ReplayBuffer` directly.
- **TLDR**:
  - `zstd` in general is very decent at save latency & memory saving, personally I recommend `**zstd-3`**.
  - `zstd-1` ~ `zstd-5` seems to be the sweet spot.
  - `gzip0` should be avoided, saving / loading has similar latency as `zstd-5`, but 13x bigger.
  - MsPacman at `84x84` resolution is too visually noisy for `rle` , although decompression isn't half-bad


| Compression  | Save Mem | Save Mem % | Save Latency | Load Mem | Load Mem % | Load Latency |
| ------------ | -------- | ---------- | ------------ | -------- | ---------- | ------------ |
| baseline     | 1.05GB   | 100.0%     | 0.9          | 4.21GB   | 100.0%     | 5.21         |
| none         | 1.05GB   | 100.1%     | 1.2          | 1.05GB   | 25.0%      | 8.70         |
| zstd-100     | 387MB    | 36.0%      | 1.8          | 413MB    | 9.6%       | 9.08         |
| zstd-50      | 306MB    | 28.4%      | 1.9          | 326MB    | 7.6%       | 8.95         |
| zstd-5       | 82.9MB   | 7.7%       | 2.1          | 89.1MB   | 2.1%       | 8.80         |
| lz4-frame/1  | 118MB    | 10.9%      | 2.1          | 127MB    | 2.9%       | 8.86         |
| zstd-20      | 181MB    | 16.8%      | 2.2          | 189MB    | 4.4%       | 8.91         |
| zstd-3       | 73.9MB   | 6.9%       | 2.3          | 78.7MB   | 1.8%       | 8.81         |
| zstd-1       | 66.0MB   | 6.1%       | 2.3          | 70.0MB   | 1.6%       | 8.79         |
| zstd1        | 61.3MB   | 5.7%       | 2.7          | 64.7MB   | 1.5%       | 8.90         |
| zstd3        | 59.4MB   | 5.5%       | 3.0          | 63.1MB   | 1.5%       | 8.91         |
| igzip0       | 129MB    | 12.0%      | 3.4          | 136MB    | 3.1%       | 9.60         |
| rle          | 811MB    | 75.3%      | 4.0          | 849MB    | 19.7%      | 14.7         |
| rle-jit      | 811MB    | 75.3%      | 4.0          | 849MB    | 19.7%      | 9.10         |
| rle-old      | 811MB    | 75.3%      | 4.0          | 849MB    | 19.7%      | 104          |
| lz4-block/1  | 83.2MB   | 7.7%       | 4.6          | 89.8MB   | 2.1%       | 8.73         |
| igzip1       | 114MB    | 10.6%      | 5.0          | 121MB    | 2.8%       | 9.66         |
| zstd5        | 55.9MB   | 5.2%       | 5.4          | 59.3MB   | 1.4%       | 8.90         |
| lz4-block/5  | 75.1MB   | 7.0%       | 6.3          | 80.1MB   | 1.9%       | 8.76         |
| lz4-frame/5  | 75.9MB   | 7.0%       | 6.5          | 80.8MB   | 1.9%       | 8.72         |
| gzip1        | 104MB    | 9.6%       | 7.6          | 108MB    | 2.5%       | 9.75         |
| gzip3        | 81.9MB   | 7.6%       | 8.3          | 85.9MB   | 2.0%       | 9.44         |
| igzip3       | 81.5MB   | 7.6%       | 10.5         | 87.0MB   | 2.0%       | 9.59         |
| zstd10       | 52.8MB   | 4.9%       | 10.8         | 56.5MB   | 1.3%       | 8.89         |
| lz4-block/9  | 72.0MB   | 6.7%       | 20.0         | 76.9MB   | 1.8%       | 8.69         |
| lz4-frame/9  | 72.7MB   | 6.8%       | 20.0         | 77.6MB   | 1.8%       | 8.74         |
| lz4-block/16 | 71.3MB   | 6.6%       | 57.9         | 76.2MB   | 1.8%       | 8.69         |
| lz4-frame/12 | 72.0MB   | 6.7%       | 58.4         | 77.0MB   | 1.8%       | 8.77         |
| zstd15       | 48.5MB   | 4.5%       | 99.8         | 52.0MB   | 1.2%       | 8.86         |
| zstd22       | 47.6MB   | 4.4%       | 590.7        | 51.0MB   | 1.2%       | 8.96         |


---

## [0.4.0](https://github.com/Trenza1ore/sb3-extra-buffers/releases/tag/v0.4.0) — 2025-07-22

Changes since `0.3.2`…`0.4.0`.

- Added `zstd` support, the best choice of compression so far!
- Added `lz4` block & frame format support
- Fixed a bug in `eval_model` function
- Project featured in [SB3's doc](https://stable-baselines3.readthedocs.io/en/master/misc/projects.html#sb3-extra-buffers-ram-expansions-are-overrated-just-compress-your-observations) 🥳

---

## [0.3.2](https://github.com/Trenza1ore/sb3-extra-buffers/releases/tag/v0.3.2) — 2025-07-21

Changes since `0.3.1`…`0.3.2`.

- Expanded model evaluation options
- Benchmarked memory saving



---

## [0.3.1](https://github.com/Trenza1ore/sb3-extra-buffers/releases/tag/v0.3.1) — 2025-07-20

Changes since `0.2.3`…`0.3.1`.

- Improved jit-less RLE decompression performance (renamed old implementation `rle-old`)
- Added proper pytests
- Refactored a ton of code, hopefully improves maintainability (in v0.3.0, but didn't release that version)

---

## [0.2.3](https://github.com/Trenza1ore/sb3-extra-buffers/releases/tag/v0.2.3) — 2025-07-20

Changes since `0.2.2`…`0.2.3`.

- Implemented CompressedBuffer
- Refactored code for CompressedRolloutBuffer and CompressedReplayBuffer

---

## [0.2.2](https://github.com/Trenza1ore/sb3-extra-buffers/releases/tag/v0.2.2) — 2025-07-19

Initial release `0.2.2`.

First stable-ish version, implemented replay / rollout buffer support.

---

