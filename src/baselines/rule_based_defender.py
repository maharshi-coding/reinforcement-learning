"""Rule-based baseline: deterministic threshold-based heuristic defender."""

from __future__ import annotations

import numpy as np


class RuleBasedThresholdPolicy:
    """Deterministic rule-based defender.

    Strategy (based on alert_level at index 4 of each 6-d frame):
        alert_level > 0.7  →  isolate_service (action 3)
        alert_level > 0.5  →  block_ip        (action 1)
        alert_level > 0.3  →  rate_limit      (action 2)
        otherwise          →  do_nothing      (action 0)
    """

    name = "rule_based_threshold"

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        # Extract alert_level from the last 6-d frame (index 4)
        n_features = 6
        if len(obs) >= n_features:
            last_frame_start = len(obs) - n_features
            threat = obs[last_frame_start + 4]
        else:
            threat = obs[4] if len(obs) > 4 else 0.0

        if threat > 0.7:
            return 3  # isolate
        elif threat > 0.5:
            return 1  # block
        elif threat > 0.3:
            return 2  # rate limit
        return 0  # no-op
