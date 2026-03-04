"""
Attack Simulator for AIRS.

Generates three attack modes:
  - brute_force: failed login attempts spike with moderate traffic
  - flood:       traffic rate spikes with moderate login failures
  - adaptive:    dynamically switches strategy to evade defenses
"""

import random


class AttackSimulator:
    """Simulates different network attack patterns."""

    MODES = ("brute_force", "flood", "adaptive")

    # Intensity parameters keyed by (mode, intensity)
    # Each entry: (traffic_rate_range, failed_logins_range)
    _PARAMS = {
        "brute_force": {
            "low":    ((10, 30),  (20, 50)),
            "medium": ((30, 60),  (50, 150)),
            "high":   ((60, 100), (150, 300)),
        },
        "flood": {
            "low":    ((100, 300),  (0, 5)),
            "medium": ((300, 700),  (0, 10)),
            "high":   ((700, 1000), (0, 15)),
        },
        "adaptive": {
            "low":    ((10, 300),  (0, 50)),
            "medium": ((30, 700),  (0, 150)),
            "high":   ((60, 1000), (0, 300)),
        },
    }

    def __init__(self, mode: str = "brute_force", intensity: str = "medium"):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        if intensity not in ("low", "medium", "high"):
            raise ValueError("intensity must be low, medium, or high")
        self.mode = mode
        self.intensity = intensity
        self._step_count = 0
        self._switch_interval = random.randint(10, 30)  # steps before strategy switch
        self._current_adaptive_mode = random.choice(["brute_force", "flood"])

    # ------------------------------------------------------------------
    def step(self, last_action: int) -> dict:
        """Return simulated attack metrics for one timestep.

        Args:
            last_action: The defensive action taken in the previous step.
                         0=no-op, 1=block, 2=rate-limit, 3=isolate

        Returns:
            dict with keys: traffic_rate, failed_logins, is_attacking
        """
        self._step_count += 1

        effective_mode = self.mode
        if self.mode == "adaptive":
            # Switch strategy periodically or when blocked
            if (
                self._step_count % self._switch_interval == 0
                or last_action in (1, 3)
            ):
                self._current_adaptive_mode = (
                    "flood"
                    if self._current_adaptive_mode == "brute_force"
                    else "brute_force"
                )
                self._switch_interval = random.randint(8, 25)
            effective_mode = self._current_adaptive_mode

        params = self._PARAMS[effective_mode][self.intensity]
        traffic_rate = random.uniform(*params[0])
        failed_logins = random.uniform(*params[1])

        # Attacker backs off temporarily after a strong defensive action
        if last_action == 3:  # isolate
            traffic_rate *= 0.2
            failed_logins *= 0.2
        elif last_action == 1:  # block
            failed_logins *= 0.3
        elif last_action == 2:  # rate-limit
            traffic_rate *= 0.4

        return {
            "traffic_rate": max(0.0, traffic_rate),
            "failed_logins": max(0.0, failed_logins),
            "is_attacking": True,
        }

    def reset(self):
        """Reset internal state for a new episode."""
        self._step_count = 0
        self._switch_interval = random.randint(10, 30)
        self._current_adaptive_mode = random.choice(["brute_force", "flood"])
