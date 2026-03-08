"""
Baseline policies for AIRS evaluation.

Provides simple reference policies to compare against the RL agent:
  - AlwaysNoopPolicy:          always takes action 0 (do nothing)
  - RandomPolicy:              uniformly random action
  - RuleBasedThresholdPolicy:  deterministic threshold-based heuristic
"""

from __future__ import annotations

import numpy as np


class AlwaysNoopPolicy:
    """Baseline that never takes defensive action."""

    name = "always_noop"

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        return 0


class RandomPolicy:
    """Baseline that picks a uniformly random action each step."""

    name = "random_policy"

    def __init__(self, n_actions: int = 4, seed: int = 42):
        self._rng = np.random.default_rng(seed)
        self._n_actions = n_actions

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        return int(self._rng.integers(0, self._n_actions))


class RuleBasedThresholdPolicy:
    """Deterministic rule-based defender.

    Strategy:
      - threat_level > 0.7  →  isolate (action 3)
      - threat_level > 0.5  →  block IP (action 1)
      - threat_level > 0.3  →  rate limit (action 2)
      - otherwise           →  no-op (action 0)

    Assumes threat_level is at index 4 of the observation vector
    (or index 4 of the last 6-d frame in a stacked observation).
    """

    name = "rule_based_threshold"

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        # Use the last frame's threat_level (index 4 in each 6-d slice)
        threat = obs[-2] if len(obs) >= 6 else obs[4]  # obs[-2] = threat in last frame
        # Actually threat is at position 4 of each 6-d block
        n_features = 6
        if len(obs) >= n_features:
            last_frame_start = len(obs) - n_features
            threat = obs[last_frame_start + 4]

        if threat > 0.7:
            return 3  # isolate
        elif threat > 0.5:
            return 1  # block
        elif threat > 0.3:
            return 2  # rate limit
        return 0  # no-op


BASELINE_REGISTRY = {
    "always_noop": AlwaysNoopPolicy,
    "random_policy": RandomPolicy,
    "rule_based_threshold": RuleBasedThresholdPolicy,
}


def get_baseline(name: str, **kwargs):
    """Instantiate a baseline policy by name."""
    if name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline '{name}'. Choose from {list(BASELINE_REGISTRY)}")
    return BASELINE_REGISTRY[name](**kwargs)
