Validation
==========

The repository includes example scripts for training and evaluating SB3 models
with compressed buffers. They are intended to verify that the buffer classes can
be used through normal SB3 algorithm constructors rather than through a separate
training loop. Browse the examples in the
`examples directory <https://github.com/Trenza1ore/sb3-extra-buffers/tree/main/examples>`__.

Evaluation results for example training scripts
-------------------------------------------------

The example scripts have been run and evaluated to confirm they train correctly.
Each run below used ``rle-jit`` compression and 10M environment steps.

PPO on ``PongNoFrameskip-v4``, no frame stack:

.. code-block:: text

   (Best ) Evaluated 10000 episodes, mean reward: 21.0 +/- 0.00
   Q1:   21 | Q2:   21 | Q3:   21 | Relative IQR: 0.00 | Min: 21 | Max: 21
   (Final) Evaluated 10000 episodes, mean reward: 21.0 +/- 0.02
   Q1:   21 | Q2:   21 | Q3:   21 | Relative IQR: 0.00 | Min: 20 | Max: 21

PPO on ``MsPacmanNoFrameskip-v4``, with frame stack ``4``:

.. code-block:: text

   (Best ) Evaluated 10000 episodes, mean reward: 2667.0 +/- 290.00
   Q1: 2300 | Q2: 2490 | Q3: 3000 | Relative IQR: 0.28 | Min: 2300 | Max: 3000
   (Final) Evaluated 10000 episodes, mean reward: 2500.9 +/- 221.03
   Q1: 2300 | Q2: 2390 | Q3: 2490 | Relative IQR: 0.08 | Min: 1420 | Max: 3000

DQN on ``MsPacmanNoFrameskip-v4``, with frame stack ``4``:

.. code-block:: text

   (Best ) Evaluated 10000 episodes, mean reward: 3300.0 +/- 770.79
   Q1: 2490 | Q2: 4020 | Q3: 4020 | Relative IQR: 0.38 | Min: 2460 | Max: 4020
   (Final) Evaluated 10000 episodes, mean reward: 3379.2 +/- 453.78
   Q1: 2690 | Q2: 3400 | Q3: 3880 | Relative IQR: 0.35 | Min: 1230 | Max: 4090
