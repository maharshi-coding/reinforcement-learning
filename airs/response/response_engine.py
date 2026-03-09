"""
Response Engine for AIRS.

Simulates four defensive actions in response to detected threats.
Each action has a defined cost (service disruption penalty) and
a defined effectiveness against each attack type.

Actions may succeed or fail based on configurable success probabilities,
making the environment more realistic (stochastic outcomes).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger("airs.response")


@dataclass
class ActionOutcome:
    """Result of applying a defensive action."""

    action_id: int
    action_name: str
    # How much the action reduces the threat level (0–1)
    threat_reduction: float
    # Disruption cost imposed on normal service availability (0–1)
    service_cost: float
    # Whether the action succeeded (stochastic)
    success: bool = True


# Default success probabilities per action
DEFAULT_SUCCESS_PROBS: dict[int, float] = {
    0: 1.0,   # no-op always "succeeds"
    1: 0.90,  # block_ip: 90%
    2: 0.80,  # rate_limit: 80%
    3: 0.85,  # isolate_service: 85%
}


class ResponseEngine:
    """Maps discrete actions to simulated defensive outcomes.

    Actions
    -------
    0  No-op          – observe only; zero cost, zero reduction
    1  Block IP        – stops brute-force well; limited vs flood
    2  Rate limiting   – halves flood traffic; moderate brute-force effect
    3  Isolate service – maximum threat reduction; highest service cost

    Stochastic Outcomes
    -------------------
    Each non-noop action has a configurable success probability.
    On failure, threat_reduction is scaled down to a residual fraction
    (default 10% of the normal reduction), but service cost is still incurred.
    """

    ACTION_NAMES = {
        0: "no_op",
        1: "block_ip",
        2: "rate_limit",
        3: "isolate_service",
    }

    # (threat_reduction, service_cost) per action_id
    _BASE_PARAMS = {
        0: (0.0,  0.00),   # no-op
        1: (0.55, 0.10),   # block IP
        2: (0.40, 0.15),   # rate limit
        3: (0.80, 0.40),   # isolate service
    }

    def __init__(
        self,
        success_probs: Optional[dict[int, float]] = None,
        failure_residual: float = 0.10,
        stochastic: bool = True,
        seed: Optional[int] = None,
    ):
        self._success_probs = dict(DEFAULT_SUCCESS_PROBS)
        if success_probs:
            self._success_probs.update(success_probs)
        self._failure_residual = failure_residual
        self._stochastic = stochastic
        self._rng = np.random.default_rng(seed)

    def apply(self, action_id: int, threat_level: float) -> ActionOutcome:
        """Apply a defensive action and return the outcome.

        The threat reduction scales with the current threat level so that
        strong actions are more impactful under high-threat conditions.
        If stochastic mode is enabled, actions may fail — reducing their
        effectiveness while still incurring service cost.

        Args:
            action_id:    Integer in {0, 1, 2, 3}.
            threat_level: Current normalised threat level in [0, 1].

        Returns:
            ActionOutcome describing the result.
        """
        if action_id not in self._BASE_PARAMS:
            raise ValueError(f"Unknown action_id {action_id}. Must be 0–3.")

        base_reduction, base_cost = self._BASE_PARAMS[action_id]

        # Determine success/failure
        prob = self._success_probs.get(action_id, 1.0)
        success = True
        if self._stochastic and action_id > 0:
            success = float(self._rng.random()) < prob

        # Scale reduction by threat: more effective when actually needed
        effective_reduction = min(base_reduction * (0.5 + threat_level), 1.0)

        # On failure: minimal residual reduction, but cost is still paid
        if not success:
            effective_reduction *= self._failure_residual
            logger.debug(
                "Action %s FAILED (prob=%.0f%%) — residual reduction %.3f",
                self.ACTION_NAMES[action_id], prob * 100, effective_reduction,
            )

        # Service cost increases slightly under high load
        effective_cost = min(base_cost * (1.0 + 0.2 * threat_level), 1.0)

        return ActionOutcome(
            action_id=action_id,
            action_name=self.ACTION_NAMES[action_id],
            threat_reduction=effective_reduction,
            service_cost=effective_cost,
            success=success,
        )

    def get_action_name(self, action_id: int) -> str:
        return self.ACTION_NAMES.get(action_id, "unknown")

    @property
    def num_actions(self) -> int:
        return len(self._BASE_PARAMS)

    @property
    def success_probs(self) -> dict[int, float]:
        return dict(self._success_probs)
