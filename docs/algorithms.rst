Supported Compression Algorithms
================================

Compressed buffers accept registered compression method names through the
``compression_method`` argument.

Implemented compression methods
-------------------------------

- ``none`` — No compression other than casting to ``elem_type`` and storing as
  ``bytes``.
- ``rle`` — Vectorized run-length encoding for compression.
- ``rle-jit`` — JIT-compiled version of ``rle``; uses the
  `numba <https://numba.pydata.org>`__ library.
- ``gzip`` — Built-in gzip compression via Python's ``gzip`` module.
- ``igzip`` — Intel-accelerated variant via ``isal.igzip``; uses
  `python-isal <https://github.com/pycompression/python-isal>`__.
- **``zstd``** — Zstandard compression via
  `python-zstd <https://github.com/sergey-dryabzhinsky/python-zstd>`__.
  **(Recommended)**
- ``lz4-frame`` — LZ4 (frame format) compression via
  `python-lz4 <https://github.com/python-lz4/python-lz4>`__.
- ``lz4-block`` — LZ4 (block format) compression via
  `python-lz4 <https://github.com/python-lz4/python-lz4>`__.

The recommended starting point from the :doc:`benchmarks` is ``zstd-3``. For very
sparse or palette-like observations, ``rle`` and ``rle-jit`` are worth testing as
well.

Compression levels
------------------

- ``gzip`` supports levels ``0`` through ``9``; ``0`` is no compression, ``1`` is
  least compression.
- ``igzip`` supports levels ``0`` through ``3``; ``0`` is least compression.
- ``zstd`` supports standard levels ``1`` through ``22`` and ultra-fast levels
  ``-100`` through ``-1``; ``-100`` is fastest and ``22`` is slowest.
- ``lz4-frame`` supports standard levels ``0`` through ``16``; negative levels map
  to an acceleration factor.
- ``lz4-block`` supports three modes split by sign of the level:

  - ``1`` through ``12`` — ``high_compression`` mode.
  - ``0`` — ``default`` mode.
  - Negative levels — ``fast`` mode; the level maps to an acceleration factor.

Shorthand method names
----------------------

Level suffixes can be appended as shorthand strings (for ``lz4`` methods, a
``/`` before the level is required):

.. code-block:: text

   pattern = ^((?:[A-Za-z]+)|(?:[\w\-]+/))(\-?[0-9]+)$

Examples:

- ``igzip3`` = ``igzip/3`` = igzip level 3
- ``zstd-5`` = ``zstd/-5`` = zstd level -5
- ``lz4-frame/5`` = lz4-frame level 5

In Python:

.. code-block:: python

   compression_method = "zstd-3"
   compression_method = "igzip3"
   compression_method = "lz4-frame/5"

Optional backends
-----------------

Install optional extras for faster or additional compression implementations:

.. code-block:: bash

   pip install "sb3-extra-buffers[fast]"

The ``fast`` extra installs ``isal``, ``numba``, ``zstd``, and ``lz4``.
