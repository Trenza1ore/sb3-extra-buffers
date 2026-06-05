Training speed
==============

This page reports end-to-end training wall-clock time when using the default
Stable-Baselines3 buffers versus the compressed buffer classes from this package.
The runs mirror the example scripts
:download:`example_train_rollout.py <../examples/example_train_rollout.py>` and
:download:`example_train_replay.py <../examples/example_train_replay.py>`, with
only the buffer class and compression method changed for the comparison.

Benchmark setup
---------------

Hardware
~~~~~~~~

- Mac mini (Apple M4, 16 GB RAM)
- ``device="mps"``

Libraries
~~~~~~~~~

- Stable-Baselines3 2.8.0
- PyTorch 2.12.0
- sb3-extra-buffers 0.5.1

Each run trained for 10M environment steps. Compressed buffers used ``zstd-3``
with dtypes from :func:`~sb3_extra_buffers.compressed.find_buffer_dtypes`.

PPO on ``PongNoFrameskip-v4``
-----------------------------

Hyperparameters follow the preset on Huggingface:
`sb3/ppo-PongNoFrameskip-v4 <https://huggingface.co/sb3/ppo-PongNoFrameskip-v4>`__
and check :download:`example_train_rollout.py <../examples/example_train_rollout.py>` for code:

- Frame stack: ``1`` (no stacking)
- ``n_envs``: ``8`` (train and eval)
- ``n_steps``: ``128``
- ``batch_size``: ``256``
- ``n_epochs``: ``4``
- ``learning_rate``: linear schedule from ``2.5e-4`` to ``0``
- ``clip_range``: linear schedule from ``0.1`` to ``0``
- ``ent_coef``: ``0.01``
- ``vf_coef``: ``0.5``
- ``gae_lambda``: ``0.9``
- ``gamma``: ``0.99``
- ``max_grad_norm``: ``0.5``
- Policy: ``CnnPolicy`` with ``normalize_images=False``

.. list-table::
   :header-rows: 1

   * - Buffer
     - Wall-clock time
   * - SB3 ``RolloutBuffer``
     - 2:56:16
   * - ``CompressedRolloutBuffer`` (``zstd-3``)
     - 4:48:41

On this setup, ``zstd-3`` rollout compression added roughly 65% training time
over the default buffer while keeping the same SB3 training loop.

DQN on ``MsPacmanNoFrameskip-v4``
---------------------------------

Hyperparameters follow the preset on Huggingface:
`sb3/dqn-MsPacmanNoFrameskip-v4 <https://huggingface.co/sb3/dqn-MsPacmanNoFrameskip-v4>`__
and check :download:`example_train_replay.py <../examples/example_train_replay.py>` for code:

- Frame stack: ``4``
- ``n_envs``: ``1`` (train), ``8`` (eval)
- ``buffer_size``: ``100_000``
- ``batch_size``: ``32``
- ``learning_starts``: ``100_000``
- ``train_freq``: ``4``
- ``gradient_steps``: ``1``
- ``target_update_interval``: ``1000``
- ``learning_rate``: ``1e-4``
- ``exploration_fraction``: ``0.1``
- ``exploration_final_eps``: ``0.01``
- Policy: ``CnnPolicy``

.. list-table::
   :header-rows: 1

   * - Buffer
     - Wall-clock time
   * - SB3 ``ReplayBuffer``
     - 12:33:26
   * - ``CompressedReplayBuffer`` (``zstd-3``)
     - 12:44:16

For DQN replay compression, ``zstd-3`` added only about 11 minutes (~1.4%) on
top of a 12.5-hour run. Off-policy algorithms spend more time in gradient
updates than in buffer I/O, so compression overhead is smaller than for PPO.

See also
--------

- :doc:`benchmarks` for per-transition buffer memory and sampling latency
- :doc:`validation` for reward results from the example training scripts
