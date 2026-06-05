GPU Buffers
===========

.. warning::

   **Advanced / experimental.** :mod:`sb3_extra_buffers.gpu_buffers` targets users who
   want observation tensors to live on a Torch device (CPU, CUDA, or MPS) with optional
   compression. The API is less stable than :mod:`sb3_extra_buffers.compressed`, behaviour
   varies by device and codec, and not every CPU compression backend has a GPU-native
   implementation yet.

.. caution::

   **You are responsible for heap sizing.** Compressed observations are stored in a single
   packed byte heap (:class:`~sb3_extra_buffers.gpu_buffers.raw_buffer.RawBuffer`) with
   per-cell ``start_idx`` / ``lengths``. If the heap runs out of space, compression raises
   an error. By default, buffers set heap capacity via
   :func:`~sb3_extra_buffers.gpu_buffers.utils.estimate_total_heap_bytes` (per-cell
   heuristic × number of cells). For large or high-entropy observations, set
   ``heap_bytes`` explicitly on :class:`~sb3_extra_buffers.gpu_buffers.gpu_replay.GpuReplayBuffer`
   or :class:`~sb3_extra_buffers.gpu_buffers.gpu_rollout.GpuRolloutBuffer` and validate with
   your observation distribution. The deprecated ``max_slot_bytes`` argument is interpreted
   as per-cell capacity and multiplied by the cell count.

Overview
--------

GPU buffers mirror the CPU compressed replay/rollout API but keep flattened observations on
``buffer_device`` end-to-end (add → store → sample/get). Non-observation fields (actions,
rewards, dones, etc.) remain NumPy arrays like the CPU buffers.

**Public entry points**

- :class:`~sb3_extra_buffers.gpu_buffers.gpu_replay.GpuReplayBuffer` — off-policy (DQN, SAC, …)
- :class:`~sb3_extra_buffers.gpu_buffers.gpu_rollout.GpuRolloutBuffer` — on-policy (PPO, …)
- :func:`~sb3_extra_buffers.gpu_buffers.base.find_gpu_buffer_dtypes` — Torch ``elem_type`` / ``runs_type`` helpers
- :data:`~sb3_extra_buffers.gpu_buffers.compression_methods.COMPRESSION_METHOD_MAP` — registered codecs
- :func:`~sb3_extra_buffers.gpu_buffers.compression_methods.has_zstd` — optional Zstd backend probe

**Compression methods**

+------------------+--------------------+--------------------------------------------------+
| Method           | Storage            | Notes                                            |
+==================+====================+==================================================+
| ``none``         | Dense tensor       | Full-size flat obs on ``buffer_device``           |
+------------------+--------------------+--------------------------------------------------+
| ``rle``          | Packed raw heap    | Run-length encoding in device byte storage       |
+------------------+--------------------+--------------------------------------------------+
| ``zstd``         | Packed raw heap    | CPU Zstd round-trip into heap bytes; needs       |
|                  |                    | ``pip install "sb3-extra-buffers[zstd]"``        |
+------------------+--------------------+--------------------------------------------------+
| ``zstd3``, etc.  | Packed raw heap    | Shorthand for ``compresslevel`` (same as CPU)    |
+------------------+--------------------+--------------------------------------------------+

Example scripts (Pong, ``PongNoFrameskip-v4``):

- ``examples/example_train_gpu_replay.py`` / ``examples/example_watch_gpu_replay.py``
- ``examples/example_train_gpu_rollout.py`` / ``examples/example_watch_gpu_rollout.py``

.. automodule:: sb3_extra_buffers.gpu_buffers
   :members:
   :undoc-members:

Metadata
--------

.. automodule:: sb3_extra_buffers.gpu_buffers.metadata
   :members:

Raw storage
-----------

.. automodule:: sb3_extra_buffers.gpu_buffers.raw_buffer
   :members:

Observation stores
------------------

.. automodule:: sb3_extra_buffers.gpu_buffers.observation_store
   :members:

Base helpers
------------

.. automodule:: sb3_extra_buffers.gpu_buffers.base
   :members:

Replay buffer
-------------

.. automodule:: sb3_extra_buffers.gpu_buffers.gpu_replay
   :members:

Rollout buffer
--------------

.. automodule:: sb3_extra_buffers.gpu_buffers.gpu_rollout
   :members:

Utilities
---------

.. automodule:: sb3_extra_buffers.gpu_buffers.utils
   :members:

Compression methods
-------------------

.. automodule:: sb3_extra_buffers.gpu_buffers.compression_methods
   :members:

Usage
-----

Replay buffer with RLE on CUDA:

.. code-block:: python

   import torch as th
   from stable_baselines3 import DQN
   from sb3_extra_buffers.gpu_buffers import GpuReplayBuffer, find_gpu_buffer_dtypes
   from sb3_extra_buffers.gpu_buffers.utils import numpy_dtype_to_torch

   device = "cuda" if th.cuda.is_available() else "cpu"
   dtypes = find_gpu_buffer_dtypes(
       env.observation_space.shape,
       elem_dtype=numpy_dtype_to_torch(env.observation_space.dtype),
       compression_method="rle",
   )
   model = DQN(
       "CnnPolicy",
       env,
       replay_buffer_class=GpuReplayBuffer,
       replay_buffer_kwargs=dict(
           dtypes=dtypes,
           compression_method="rle",
           buffer_device=device,
           # heap_bytes=...,  # override if estimate_total_heap_bytes is too small
       ),
       device=device,
   )

Rollout buffer with optional Zstd (when installed):

.. code-block:: python

   from sb3_extra_buffers.gpu_buffers import GpuRolloutBuffer, has_zstd

   compression_method = "zstd" if has_zstd() else "rle"
   dtypes = find_gpu_buffer_dtypes(obs_shape, compression_method=compression_method)
   model = PPO(
       "CnnPolicy",
       env,
       rollout_buffer_class=GpuRolloutBuffer,
       rollout_buffer_kwargs=dict(
           dtypes=dtypes,
           compression_method=compression_method,
           buffer_device=device,
       ),
       device=device,
   )

Heap layout and compaction
--------------------------

Compressed observations share one :class:`~sb3_extra_buffers.gpu_buffers.raw_buffer.RawBuffer`
backed by :class:`~sb3_extra_buffers.gpu_buffers.raw_buffer.SharedRawHeap`. Each cell has a
``start_idx`` and ``lengths`` entry; codec metadata (``pos_runs``, ``pos_elem``, ``run_length``)
is stored relative to that cell's byte offset.

**Compaction** packs live payloads and resets ``data_end`` to roughly ``sum(lengths)``:

- :class:`~sb3_extra_buffers.gpu_buffers.gpu_replay.GpuReplayBuffer` compacts when the write
  cursor wraps after the buffer is full.
- :class:`~sb3_extra_buffers.gpu_buffers.gpu_rollout.GpuRolloutBuffer` compacts when the
  rollout buffer becomes full (before ``get()``).

Default heap capacity is ``n_cells * estimate_max_slot_bytes(...)`` where ``n_cells`` depends
on buffer geometry (replay vs rollout, ``n_envs``, shared next-obs storage). Per-cell size uses
MsPacman **Save Mem %** ratios from the README benchmark, fitted per codec family and level
(see ``scripts/fit_heap_heuristic.py``), scaled by ``flat_len * elem_size * overalloc_factor``
(default **1.5**). All documented levels are accepted; unmeasured levels use polynomial
extrapolation clamped to each family's benchmark min/max ratio.

Supported level ranges (invalid levels raise ``ValueError``):

- ``gzip``: 0–9
- ``igzip``: 0–3
- ``zstd``: 1–22 or -100–-1 (bare ``zstd`` uses level -3)
- ``lz4-frame``: negatives and 0–16
- ``lz4-block``: negatives, 0, and 1–12

Re-fit after updating the benchmark table::

   uv pip install scipy
   uv run scripts/fit_heap_heuristic.py

Copy the printed constants into :mod:`sb3_extra_buffers.gpu_buffers.size_estimation`.

After compaction, ``data_end`` reflects actual compressed usage; peak allocation remains ``heap_bytes``.

If compression fails with a heap capacity error, increase ``heap_bytes``. When in doubt,
log compressed sizes from your environment and add headroom.
