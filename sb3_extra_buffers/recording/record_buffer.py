from typing import Optional

import numpy as np


class RecordBuffer:
    def __init__(self, res: tuple[int, int] = (240, 320), ch_num: int = 3, size: int = 40000,
                 dtypes: Optional[list[np.dtype]] = None):
        """A class for recording game states.

        Args:
            res (tuple[int, int], optional): resolution in form of (height, width). Defaults to (240, 320).
            ch_num (int): number of colour channels. Defaults to 3.
            size (int): maximum size, starts deleting old memories after maximum reached. Defaults to 40000.
            dtypes (list[object]): data type of frame, reward and (if any) features, datatype for features
            should be passed in as strings like "np.uint8" or "bool". Defaults to [np.uint8, np.float32].
        """
        self.max_size = size
        self.max_index = size - 1
        self.ch_num = ch_num
        if dtypes is None:
            dtypes = [np.uint8, np.float32]
        self.dtype = {
            "frame": (dtypes[0], (size, ch_num, *res)),
            "reward": (dtypes[1], (size, ))
        }

        if len(dtypes) > 2:
            self.dtype["features"] = (*dtypes[2:], )
            self.feature_num = len(self.dtype["features"])
            self.use_features = True
            if len(set(self.dtype["features"])) == 1:
                self.features = np.zeros((size, self.feature_num), dtype=self.dtype["features"][0])
            else:
                self.features = np.zeros(size, dtype=[
                    (str(i), self.dtype["features"][i]) for i in range(self.feature_num)
                ])
        else:
            self.feature_num = 0
            self.use_features = False

        self.frames = np.zeros((size, ch_num, *res), dtype=dtypes[0])
        self.rewards = np.zeros(size, dtype=dtypes[1])
        self.actions = np.zeros(size, dtype=np.uint8)

        self._ptr = -1

    # functions for ease of checking
    def __len__(self) -> int:
        return self._ptr + 1

    def __str__(self) -> str:
        return f"ReplayMemory(f:{self.dtype["frame"]}, r:{self.dtype["reward"]})"

    def __repr__(self) -> str:
        return self.__str__()

    # add is replaced by add_filled after the memory has been filled once
    def add(self, frame: np.ndarray[np.integer], reward: np.floating, action: np.uint8, features: tuple = None):
        """Add a single state into memory"""
        if self._ptr < self.max_index:
            self._ptr += 1
        else:
            self._ptr = 0
            self.add = self.add_filled
        self.frames[self._ptr, :, :, :] = frame.transpose(2, 0, 1)
        self.rewards[self._ptr] = reward
        self.actions[self._ptr] = action
        if self.use_features:
            self.features[self._ptr] = features

    def add_filled(self, frame: np.ndarray[np.integer], reward: np.floating, action: np.uint8, features: tuple):
        """Add a single state into memory"""
        self._ptr = self._ptr + 1 if self._ptr < self.max_index else 0
        self.frames[self._ptr, :, :, :] = frame.transpose(2, 0, 1)
        self.rewards[self._ptr] = reward
        self.actions[self._ptr] = action
        if self.use_features:
            self.features[self._ptr] = features
