from __future__ import annotations

from typing import Dict

from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from rl.state_encoder import StateZExtractor


class SharedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, state_z_extractor: StateZExtractor):
        super().__init__(observation_space, features_dim=state_z_extractor.features_dim)
        self.extractor = state_z_extractor

    def forward(self, observations: Dict):
        return self.extractor(observations)
