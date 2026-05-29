GPU Buffers
===========

.. warning::

   **Advanced / experimental.** :mod:`sb3_extra_buffers.gpu_buffers` targets users who
   want observation tensors to live on a Torch device (CPU, CUDA, or MPS) with optional
   compression. The API is less stable than :mod:`sb3_extra_buffers.compressed`, behaviour
   varies by device and codec, and not every CPU compression backend has a GPU-native
   implementation yet.

.. caution::

   **You are responsible for slot sizing.** Compressed observations are stored in fixed-size
   :class:`~sb3_extra_buffers.gpu_buffers.raw_buffer.SlotArena` slots. If a codec produces
   more bytes than a slot allows, compression raises an error or data may be unusable.
   By default, buffers estimate per-slot capacity via
   :func:`~sb3_extra_buffers.gpu_buffers.utils.estimate_max_slot_bytes`, but that estimate
   is a heuristic (especially for ``zstd``). For large or high-entropy observations, set
   ``max_slot_bytes`` explicitly on :class:`~sb3_extra_buffers.gpu_buffers.gpu_replay.GpuReplayBuffer`
   or :class:`~sb3_extra_buffers.gpu_buffers.gpu_rollout.GpuRolloutBuffer` and validate with
   your observation distribution.

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
| ``rle``          | :class:`SlotArena` | Run-length encoding in device slot bytes       |
+------------------+--------------------+--------------------------------------------------+
| ``zstd``         | :class:`SlotArena` | CPU Zstd round-trip into slot bytes; needs     |
|                  |                    | ``pip install "sb3-extra-buffers[zstd]"``        |
+------------------+--------------------+--------------------------------------------------+
| ``zstd3``, etc.  | :class:`SlotArena` | Shorthand for ``compresslevel`` (same as CPU)  |
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
           # max_slot_bytes=...,  # override if estimate_max_slot_bytes is too small
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

Slot sizing
-----------

Each compressed observation occupies one slot in a shared byte tensor of shape
``(n_slots, max_slot_bytes)``. ``n_slots`` depends on buffer geometry (replay vs rollout,
``n_envs``, whether next-observations share storage). ``max_slot_bytes`` defaults to:

- ``none``: ``flat_len * elem_itemsize``
- ``rle``: raw size + run metadata upper bound
- ``zstd``: Zstd compress-bound approximation (``raw + raw//255 + 64``)

If compression fails with a slot capacity error, increase ``max_slot_bytes``. When in doubt,
log compressed sizes from your environment and add headroom.
