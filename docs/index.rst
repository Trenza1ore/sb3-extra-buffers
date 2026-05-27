sb3-extra-buffers documentation
===============================

.. rubric:: Memory-efficient Stable-Baselines3 buffers

``sb3-extra-buffers`` is an unofficial collection of extra
`Stable-Baselines3 <https://github.com/DLR-RM/stable-baselines3>`__ buffer
classes. Its main goal is simple: keep large reinforcement-learning buffers
small enough to use comfortably by compressing observations before they are
stored.

The package is designed for reinforcement-learning workloads with large replay
or rollout buffers, especially Atari-style image observations where storing raw
frames can dominate memory use. As shown in `benchmarks on Atari games <benchmarks>`, the
best configurations reduce memory use by more than 95% while keeping sampling
latency close to the uncompressed SB3 buffers.

Why compress buffers?
---------------------

Many RL runs store hundreds of thousands or millions of observations. For image
observations, the buffer can become the dominant memory cost long before the
policy network does. This project targets that bottleneck while preserving the
normal SB3 integration pattern:

- pass a custom buffer class to an SB3 algorithm,
- pass compression options through ``rollout_buffer_kwargs`` or
  ``replay_buffer_kwargs``,
- continue training with the normal SB3 workflow.

Lossless compression is most useful when observations contain repeated or
structured values. Good candidates include semantic-segmentation masks, color
palette game frames, grayscale observations, and many RGB observations. For
noisy RGB input, ``zstd`` is usually a stronger default than run-length
encoding.

Project links
-------------

- `My master's thesis that started the project <https://arxiv.org/abs/2511.11703>`__
- `Stable-Baselines3 project listing <https://stable-baselines3.readthedocs.io/en/master/misc/projects.html#sb3-extra-buffers-ram-expansions-are-overrated-just-compress-your-observations>`__
- `Source code <https://github.com/Trenza1ore/sb3-extra-buffers>`__
- `PyPI package <https://pypi.org/project/sb3-extra-buffers/>`__

Package layout
--------------

``sb3_extra_buffers.compressed``
   Compressed replay and rollout buffers, compressed arrays, dtype helpers, and
   compression backend discovery.

``sb3_extra_buffers.recording``
   Circular recording buffers for frames, rewards, actions, and optional
   features, plus frameless and dummy variants.

``sb3_extra_buffers.training_utils``
   Atari environment creation, evaluation, and replay-buffer warm-up helpers.

``sb3_extra_buffers.vec_buf`` and ``sb3_extra_buffers.gpu_buffers``
   Experimental helpers for multi-buffer delegation and GPU-oriented byte
   storage.

How do I know the compressed buffers are implemented correctly?
---------------------------------------------------------------

The repository includes tested example training scripts. After 10M steps with
``rle-jit`` compression, evaluation on 10,000 episodes produced the rewards
documented in :doc:`validation`.

Citing this project
-------------------

If you use this project in your research or work, please cite:

.. code-block:: bibtex

   @article{Huang2025EnhancingRL,
     title={Enhancing Reinforcement Learning in 3D Environments through Semantic Segmentation: A Case Study in ViZDoom},
     author={Hugo Huang},
     journal={ArXiv},
     year={2025},
     volume={abs/2511.11703},
     url={https://arxiv.org/abs/2511.11703},
   }

I really appreciate it :-)

What is included
----------------

.. topic:: Compressed buffers
   :class: seb-doc-card

   Use :class:`sb3_extra_buffers.compressed.CompressedReplayBuffer` and
   :class:`sb3_extra_buffers.compressed.CompressedRolloutBuffer` as SB3 buffer
   classes with compression backends such as RLE, gzip, igzip, zstd, and lz4.

.. topic:: Recording buffers
   :class: seb-doc-card

   Capture frames, rewards, actions, and optional features with
   :class:`sb3_extra_buffers.recording.RecordBuffer`, or use no-op variants when
   you only need the buffer interface.

.. topic:: Training helpers
   :class: seb-doc-card

   Utilities for Atari environment creation, policy evaluation, and replay
   buffer warm-up are available under :mod:`sb3_extra_buffers.training_utils`.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   quickstart
   algorithms
   usage
   benchmarks
   validation
   citation
   api/index
   Changelog <changelog>

----

.. rubric:: Indices

The :ref:`genindex` lists documented names in alphabetical order.
