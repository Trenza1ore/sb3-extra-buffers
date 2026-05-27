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

Compression methods
-------------------

Compressed buffers accept registered compression method names through the
``compression_method`` argument.

Common choices:

- ``none`` stores raw bytes after dtype conversion.
- ``rle`` uses NumPy run-length encoding.
- ``rle-jit`` uses the numba-backed run-length encoder.
- ``gzip`` uses Python's standard gzip backend.
- ``igzip`` uses Intel's accelerated gzip backend when ``isal`` is installed.
- ``zstd`` uses Zstandard and is a strong default for noisy image observations.
- ``lz4-frame`` and ``lz4-block`` use lz4 backends.

The recommended starting point from the benchmark is ``zstd-3``. For very sparse
or palette-like observations, ``rle`` and ``rle-jit`` are worth testing as well.

Compression levels can be included as shorthand strings:

.. code-block:: python

   compression_method = "zstd-3"
   compression_method = "igzip1"
   compression_method = "lz4-frame/5"

Supported level ranges:

- ``gzip`` supports levels ``0`` through ``9``.
- ``igzip`` supports levels ``0`` through ``3``.
- ``zstd`` supports standard levels ``1`` through ``22`` and ultra-fast levels
  ``-100`` through ``-1``.
- ``lz4-frame`` supports standard levels ``0`` through ``16``; negative levels
  map to acceleration factors.
- ``lz4-block`` supports default mode at ``0``, high-compression mode from
  ``1`` through ``12``, and fast mode through negative levels.

For lz4 methods, include the slash before the level, for example
``lz4-frame/5``. For the other methods, both explicit and compact forms are
accepted by the parser, such as ``igzip/3`` and ``igzip3``.

Optional backends
-----------------

Install optional extras for faster or additional compression implementations:

.. code-block:: bash

   pip install "sb3-extra-buffers[fast]"

The ``fast`` extra installs ``isal``, ``numba``, ``zstd``, and ``lz4``.

If an optional backend is unavailable, the package either falls back to a
compatible implementation or raises a clear import error for explicit JIT
initialization.

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
