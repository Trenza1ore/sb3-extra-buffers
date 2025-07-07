import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    batch_size: int = 32

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        obs_dim = len(observation_space.shape)
        if obs_dim == 2:
            ch_num = 1
        else:
            ch_num = obs_dim[-3]

        self.cnn = nn.Sequential(
            nn.Conv2d(ch_num, 32, kernel_size=5, stride=4, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Flatten()
        )

        with torch.no_grad():
            observations = torch.from_numpy(observation_space.sample())
            if len(observations.shape) == 2:
                observations = observations[None, None, :, :]
            flatten_dim = self.cnn(
                torch.as_tensor(observations).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(flatten_dim, features_dim), nn.BatchNorm1d(features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        match len(observations.shape):
            case 2:
                observations = observations[None, None, :, :]
            case 3:
                observations = observations[:, None, :, :]
        return self.linear(self.cnn(observations))
