Usage Notes
===========

This page collects practical details that are easier to keep out of the
quickstart: when each compression method is useful, how shorthand levels work,
what optional dependencies do, and how the non-training buffers fit in.

Input types
-----------

Compression is most effective when the stored observations have repeated values
or structure. The README calls out these common cases:

- semantic segmentation masks,
- color-palette frames from retro games,
- grayscale observations,
- RGB observations when the image is not too noisy.

For noisy RGB observations, prefer ``zstd`` first. Run-length encoding can still
decompress quickly, but it may save much less memory and can even increase
storage for highly varied inputs.

Compression algorithms
----------------------

See :doc:`algorithms`

Buffer dtype settings
---------------------

Most examples pass the result of
:func:`sb3_extra_buffers.compressed.find_buffer_dtypes` into the buffer kwargs:

.. code-block:: python

   buffer_dtypes = find_buffer_dtypes(
       obs_shape=env.observation_space.shape,
       elem_dtype=env.observation_space.dtype,
       compression_method=compression,
   )

   rollout_buffer_kwargs = {
       "dtypes": buffer_dtypes,
       "compression_method": compression,
   }

The helper returns an ``elem_type`` and ``runs_type`` mapping. ``elem_type``
controls the stored observation element dtype. ``runs_type`` controls the
integer dtype used for run lengths in RLE-style compression.

Recording buffers
-----------------

Recording buffers are useful when you want to keep a fixed-size circular record
of frames, rewards, actions, and optional features for debugging or offline
inspection.

.. code-block:: python

   from sb3_extra_buffers.recording import RecordBuffer

   record_buffer = RecordBuffer(res=(84, 84), ch_num=4, size=40_000)
   record_buffer.add(frame, reward, action)

Use :class:`~sb3_extra_buffers.recording.FramelessRecordBuffer` when you only
need rewards/actions/features, or
:class:`~sb3_extra_buffers.recording.DummyRecordBuffer` when recording should be
disabled without changing the calling code.

Training utilities
------------------

The :mod:`sb3_extra_buffers.training_utils` package contains helpers used by the
examples and benchmarks:

- :func:`sb3_extra_buffers.training_utils.atari.make_env` builds Atari vector
  environments, applies optional frame stacking, and returns channel-first
  observations through ``VecTransposeImage``.
- :func:`sb3_extra_buffers.training_utils.eval_model.eval_model` evaluates a
  model and can fill one or more replay-like buffers with transitions.
- :func:`sb3_extra_buffers.training_utils.buffer_warmup.warm_up` uses a trained
  model and environment to pre-fill replay buffers before another training run.

Testing
-------

The test suite can be run with plain pytest:

.. code-block:: bash

   pytest tests -v --durations=0 --tb=short

The tests are compatible with ``pytest-xdist`` because ``DummyVecEnv`` is used
for test environments:

.. code-block:: bash

   pytest tests -n auto -v --durations=0 --tb=short

By default, tests may save observations under ``debug_obs/`` for manual
inspection. Disable that behavior in CI or low-disk environments with:

.. code-block:: bash

   DISABLE_TEST_OBSERVATIONS_SAVE=true pytest tests -v

Experimental helpers
--------------------

The :mod:`sb3_extra_buffers.vec_buf` and :mod:`sb3_extra_buffers.gpu_buffers`
packages expose experimental helpers. Treat their APIs as less stable than the
compressed and recording buffer APIs.

:mod:`sb3_extra_buffers.gpu_buffers` adds device-resident replay and rollout buffers.
See :doc:`api/gpu_buffers` for compression options (``none``, ``rle``, ``zstd``),
``buffer_device``, slot sizing, and the Pong training examples
(``examples/example_train_gpu_replay.py``, ``examples/example_train_gpu_rollout.py``).

.. warning::

   GPU buffers are intended for advanced use. Default slot byte capacity is estimated
   automatically; **you are responsible** for confirming it is large enough for your
   observations and codec, or for setting ``max_slot_bytes`` explicitly.
