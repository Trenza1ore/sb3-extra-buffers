Quickstart
==========

``sb3-extra-buffers`` plugs into Stable-Baselines3 through the same buffer hooks
used by SB3's built-in buffers. Most integrations only need two changes:

1. choose a compressed buffer class,
2. pass compression settings through the algorithm's buffer kwargs.

Installation
------------

For a practical default install with the common fast compression backends and
SB3's optional RL dependencies, use:

.. code-block:: bash

   pip install "sb3-extra-buffers[fast,extra]"

The minimum install only includes Stable-Baselines3 and tqdm:

.. code-block:: bash

   pip install "sb3-extra-buffers"

Optional extras can be installed individually:

.. code-block:: bash

   pip install "sb3-extra-buffers[extra]"    # Stable-Baselines3 extras
   pip install "sb3-extra-buffers[fast]"     # isal, numba, zstd, lz4
   pip install "sb3-extra-buffers[isal]"
   pip install "sb3-extra-buffers[numba]"
   pip install "sb3-extra-buffers[zstd]"
   pip install "sb3-extra-buffers[lz4]"
   pip install "sb3-extra-buffers[vizdoom]"

Choosing compression algorithm
------------------------------

Check :doc:`algorithms`


Choosing dtypes
---------------

Compressed buffers store flattened observations and, for RLE-style compression,
run lengths. :func:`~sb3_extra_buffers.compressed.find_buffer_dtypes` is a convenient
helper function that chooses a small integer dtype for you based on the observation shape.
When using ``rle-jit``, the same helper also initializes ``numba``.

.. code-block:: python

   from sb3_extra_buffers.compressed import find_buffer_dtypes

   compression = "zstd-3"
   buffer_dtypes = find_buffer_dtypes(
       obs_shape=env.observation_space.shape,
       elem_dtype=env.observation_space.dtype,
       compression_method=compression,
   )

Compressed rollout buffers
--------------------------

Use :class:`~sb3_extra_buffers.compressed.CompressedRolloutBuffer` with on-policy
algorithms such as PPO.

.. code-block:: python

   import numpy as np
   from stable_baselines3 import PPO

   from sb3_extra_buffers.compressed import CompressedRolloutBuffer, find_buffer_dtypes

   compression = "rle-jit"
   buffer_dtypes = find_buffer_dtypes(
       obs_shape=env.observation_space.shape,
       elem_dtype=np.uint8,
       compression_method=compression,
   )

   model = PPO(
       "CnnPolicy",
       env,
       rollout_buffer_class=CompressedRolloutBuffer,
       rollout_buffer_kwargs={
           "dtypes": buffer_dtypes,
           "compression_method": compression,
       },
   )

Full PPO Atari example
----------------------

This mirrors the README example and shows the intended ordering when JIT
compression and multiprocessing environments are used together.

.. code-block:: python

   from stable_baselines3 import PPO
   from stable_baselines3.common.callbacks import EvalCallback
   from stable_baselines3.common.utils import get_linear_fn

   from sb3_extra_buffers.compressed import CompressedRolloutBuffer, find_buffer_dtypes
   from sb3_extra_buffers.training_utils.atari import make_env

   ATARI_GAME = "MsPacmanNoFrameskip-v4"

   if __name__ == "__main__":
       probe_env = make_env(env_id=ATARI_GAME, n_envs=1, framestack=4)
       obs_space = probe_env.observation_space
       probe_env.close()

       compression = "rle-jit"
       buffer_dtypes = find_buffer_dtypes(
           obs_shape=obs_space.shape,
           elem_dtype=obs_space.dtype,
           compression_method=compression,
       )

       env = make_env(env_id=ATARI_GAME, n_envs=8, framestack=4)
       eval_env = make_env(env_id=ATARI_GAME, n_envs=10, framestack=4)

       model = PPO(
           "CnnPolicy",
           env,
           verbose=1,
           learning_rate=get_linear_fn(2.5e-4, 0, 1),
           n_steps=128,
           batch_size=256,
           clip_range=get_linear_fn(0.1, 0, 1),
           n_epochs=4,
           ent_coef=0.01,
           vf_coef=0.5,
           seed=1970626835,
           rollout_buffer_class=CompressedRolloutBuffer,
           rollout_buffer_kwargs={
               "dtypes": buffer_dtypes,
               "compression_method": compression,
           },
       )

       eval_callback = EvalCallback(
           eval_env,
           n_eval_episodes=20,
           eval_freq=8192,
           log_path=f"./logs/{ATARI_GAME}/ppo/eval",
           best_model_save_path=f"./logs/{ATARI_GAME}/ppo/best_model",
       )

       model.learn(total_timesteps=10_000_000, callback=eval_callback, progress_bar=True)
       model.save("ppo_MsPacman_4.zip")
       env.close()
       eval_env.close()

Compressed replay buffers
-------------------------

Use :class:`~sb3_extra_buffers.compressed.CompressedReplayBuffer` with off-policy
algorithms such as DQN.

.. code-block:: python

   import numpy as np
   from stable_baselines3 import DQN

   from sb3_extra_buffers.compressed import CompressedReplayBuffer, find_buffer_dtypes

   compression = "zstd-3"
   buffer_dtypes = find_buffer_dtypes(
       obs_shape=env.observation_space.shape,
       elem_dtype=np.uint8,
       compression_method=compression,
   )

   model = DQN(
       "CnnPolicy",
       env,
       replay_buffer_class=CompressedReplayBuffer,
       replay_buffer_kwargs={
           "dtypes": buffer_dtypes,
           "compression_method": compression,
       },
   )

JIT warm-up
-----------

When using ``rle-jit``, call
:func:`~sb3_extra_buffers.compressed.find_buffer_dtypes` before creating
multiprocessing environments. This initializes the numba-compiled compression
functions in the parent process.

.. code-block:: python

   obs = make_env(env_id=ATARI_GAME, n_envs=1, framestack=4).observation_space
   compression = "rle-jit"
   buffer_dtypes = find_buffer_dtypes(
       obs_shape=obs.shape,
       elem_dtype=obs.dtype,
       compression_method=compression,
   )

   # Now it is safe to create SubprocVecEnv-based environments.
   env = make_env(env_id=ATARI_GAME, n_envs=8, framestack=4)
