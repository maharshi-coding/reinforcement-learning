"""
Response Engine for AIRS.

Simulates four defensive actions in response to detected threats.
Each action has a defined cost (service disruption penalty) and
a defined effectiveness against each attack type.
"""

from dataclasses import dataclass


@dataclass
class ActionOutcome:
    """Result of applying a defensive action."""

    action_id: int
    action_name: str
    # How much the action reduces the threat level (0–1)
    threat_reduction: float
    # Disruption cost imposed on normal service availability (0–1)
    service_cost: float


class ResponseEngine:
    """Maps discrete actions to simulated defensive outcomes.

    Actions
    -------
    0  No-op          – observe only; zero cost, zero reduction
    1  Block IP        – stops brute-force well; limited vs flood
    2  Rate limiting   – halves flood traffic; moderate brute-force effect
    3  Isolate service – maximum threat reduction; highest service cost
    """

    ACTION_NAMES = {
        0: "no_op",
        1: "block_ip",
        2: "rate_limit",
        3: "isolate_service",
    }

    # (threat_reduction, service_cost) per action_id
    # Values are *base* estimates; actual values are adjusted by threat level.
    _BASE_PARAMS = {
        0: (0.0,  0.00),   # no-op
        1: (0.55, 0.10),   # block IP
        2: (0.40, 0.15),   # rate limit
        3: (0.80, 0.40),   # isolate service
    }

    def apply(self, action_id: int, threat_level: float) -> ActionOutcome:
        """Apply a defensive action and return the outcome.

        The threat reduction scales with the current threat level so that
        strong actions are more impactful under high-threat conditions.

        Args:
            action_id:    Integer in {0, 1, 2, 3}.
            threat_level: Current normalised threat level in [0, 1].

        Returns:
            ActionOutcome describing the result.
        """
        if action_id not in self._BASE_PARAMS:
            raise ValueError(f"Unknown action_id {action_id}. Must be 0–3.")

        base_reduction, base_cost = self._BASE_PARAMS[action_id]
        # Scale reduction by threat: more effective when actually needed
        effective_reduction = min(base_reduction * (0.5 + threat_level), 1.0)
        # Service cost increases slightly under high load
        effective_cost = min(base_cost * (1.0 + 0.2 * threat_level), 1.0)

        return ActionOutcome(
            action_id=action_id,
            action_name=self.ACTION_NAMES[action_id],
            threat_reduction=effective_reduction,
            service_cost=effective_cost,
        )

    def get_action_name(self, action_id: int) -> str:
        return self.ACTION_NAMES.get(action_id, "unknown")

    @property
    def num_actions(self) -> int:
        return len(self._BASE_PARAMS)
