Benchmarks
==========

This benchmark compared compressed buffers against the Stable-Baselines3
(pre-2.7.1) baseline on ``MsPacmanNoFrameskip-v4``.

Benchmark setup
---------------

- Frame stack: ``4``.
- Vectorized environments: ``4``.
- Buffer size: ``40,000`` transitions, split across vectorized environments.
- Hardware: M4 MacBook Air.
- Saving test: the same observations are added to each buffer for fairness.
- Loading test: rollout-buffer trajectories are sampled with batch size ``64``.
- Baseline: SB3 ``ReplayBuffer`` or ``RolloutBuffer`` without compression.

Important caveats:

- SB3's ``RolloutBuffer`` stores observations as ``np.float32``, which is 4x
  larger than ``np.uint8`` observations.
- ``igzip`` does not benefit from Intel SIMD acceleration on Apple Silicon.
- Transfer latency between CPU and GPU may differ on other systems.

Summary
-------

The main takeaway is that ``zstd`` gives a strong balance between memory saving
and latency. ``zstd-1`` through ``zstd-5`` are good first choices, and the README
recommends ``zstd-3`` as a practical default. ``gzip0`` should usually be
avoided because it keeps similar latency to ``zstd-5`` while using much more
memory.

Run-length encoding is more data-dependent. On the MsPacman benchmark, the
``84x84`` observations are noisy enough that ``rle`` saves much less memory than
``zstd``, although decompression remains usable.

Results
-------

.. list-table::
   :header-rows: 1

   * - Compression
     - Save Mem
     - Save Mem %
     - Save Latency
     - Load Mem
     - Load Mem %
     - Load Latency
   * - baseline
     - 1.05GB
     - 100.0%
     - 0.9
     - 4.21GB
     - 100.0%
     - 5.21
   * - none
     - 1.05GB
     - 100.1%
     - 1.2
     - 1.05GB
     - 25.0%
     - 8.70
   * - zstd-100
     - 387MB
     - 36.0%
     - 1.8
     - 413MB
     - 9.6%
     - 9.08
   * - zstd-50
     - 306MB
     - 28.4%
     - 1.9
     - 326MB
     - 7.6%
     - 8.95
   * - zstd-5
     - 82.9MB
     - 7.7%
     - 2.1
     - 89.1MB
     - 2.1%
     - 8.80
   * - lz4-frame/1
     - 118MB
     - 10.9%
     - 2.1
     - 127MB
     - 2.9%
     - 8.86
   * - zstd-20
     - 181MB
     - 16.8%
     - 2.2
     - 189MB
     - 4.4%
     - 8.91
   * - zstd-3
     - 73.9MB
     - 6.9%
     - 2.3
     - 78.7MB
     - 1.8%
     - 8.81
   * - zstd-1
     - 66.0MB
     - 6.1%
     - 2.3
     - 70.0MB
     - 1.6%
     - 8.79
   * - zstd1
     - 61.3MB
     - 5.7%
     - 2.7
     - 64.7MB
     - 1.5%
     - 8.90
   * - zstd3
     - 59.4MB
     - 5.5%
     - 3.0
     - 63.1MB
     - 1.5%
     - 8.91
   * - igzip0
     - 129MB
     - 12.0%
     - 3.4
     - 136MB
     - 3.1%
     - 9.60
   * - rle
     - 811MB
     - 75.3%
     - 4.0
     - 849MB
     - 19.7%
     - 14.7
   * - rle-jit
     - 811MB
     - 75.3%
     - 4.0
     - 849MB
     - 19.7%
     - 9.10
   * - rle-old
     - 811MB
     - 75.3%
     - 4.0
     - 849MB
     - 19.7%
     - 104
   * - lz4-block/1
     - 83.2MB
     - 7.7%
     - 4.6
     - 89.8MB
     - 2.1%
     - 8.73
   * - igzip1
     - 114MB
     - 10.6%
     - 5.0
     - 121MB
     - 2.8%
     - 9.66
   * - zstd5
     - 55.9MB
     - 5.2%
     - 5.4
     - 59.3MB
     - 1.4%
     - 8.90
   * - lz4-block/5
     - 75.1MB
     - 7.0%
     - 6.3
     - 80.1MB
     - 1.9%
     - 8.76
   * - lz4-frame/5
     - 75.9MB
     - 7.0%
     - 6.5
     - 80.8MB
     - 1.9%
     - 8.72
   * - gzip1
     - 104MB
     - 9.6%
     - 7.6
     - 108MB
     - 2.5%
     - 9.75
   * - gzip3
     - 81.9MB
     - 7.6%
     - 8.3
     - 85.9MB
     - 2.0%
     - 9.44
   * - igzip3
     - 81.5MB
     - 7.6%
     - 10.5
     - 87.0MB
     - 2.0%
     - 9.59
   * - zstd10
     - 52.8MB
     - 4.9%
     - 10.8
     - 56.5MB
     - 1.3%
     - 8.89
   * - lz4-block/9
     - 72.0MB
     - 6.7%
     - 20.0
     - 76.9MB
     - 1.8%
     - 8.69
   * - lz4-frame/9
     - 72.7MB
     - 6.8%
     - 20.0
     - 77.6MB
     - 1.8%
     - 8.74
   * - lz4-block/16
     - 71.3MB
     - 6.6%
     - 57.9
     - 76.2MB
     - 1.8%
     - 8.69
   * - lz4-frame/12
     - 72.0MB
     - 6.7%
     - 58.4
     - 77.0MB
     - 1.8%
     - 8.77
   * - zstd15
     - 48.5MB
     - 4.5%
     - 99.8
     - 52.0MB
     - 1.2%
     - 8.86
   * - zstd22
     - 47.6MB
     - 4.4%
     - 590.7
     - 51.0MB
     - 1.2%
     - 8.96

Training validation
-------------------

The table above measures buffer memory and sampling latency. To confirm that
compressed buffers work end-to-end in SB3 training, see
:doc:`validation` for evaluation results from the example PPO and DQN scripts
(10M steps on Atari with ``rle-jit``).
