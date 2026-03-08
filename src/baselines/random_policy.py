"""Random baseline: uniformly random action selection."""

from __future__ import annotations

import numpy as np


class RandomPolicy:
    """Baseline that picks a uniformly random action each step."""

    name = "random_policy"

    def __init__(self, n_actions: int = 4, seed: int = 42):
        self._rng = np.random.default_rng(seed)
        self._n_actions = n_actions

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        return int(self._rng.integers(0, self._n_actions))
