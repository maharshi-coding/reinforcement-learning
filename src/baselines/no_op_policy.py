"""No-Op baseline: defender never takes any action."""

from __future__ import annotations

import numpy as np


class AlwaysNoopPolicy:
    """Baseline that never takes defensive action (always action 0)."""

    name = "always_noop"

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        return 0
