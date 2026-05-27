Compressed Buffers
==================

.. automodule:: sb3_extra_buffers.compressed
   :members:
   :undoc-members:

Core classes
------------

.. autosummary::

   CompressedReplayBuffer
   CompressedDictReplayBuffer
   CompressedRolloutBuffer
   CompressedDictRolloutBuffer
   CompressedArray

Helpers
-------

.. autosummary::

   find_buffer_dtypes
   init_jit
   find_smallest_dtype
   has_igzip
   has_numba

Implementation modules
----------------------

.. automodule:: sb3_extra_buffers.compressed.base
   :members:

.. automodule:: sb3_extra_buffers.compressed.compressed_replay
   :members:

.. automodule:: sb3_extra_buffers.compressed.compressed_rollout
   :members:

.. automodule:: sb3_extra_buffers.compressed.compressed_array
   :members:

.. automodule:: sb3_extra_buffers.compressed.utils
   :members:

.. automodule:: sb3_extra_buffers.compressed.compression_methods
   :members:
