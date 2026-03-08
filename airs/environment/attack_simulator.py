"""
Attack Simulator for AIRS.

Generates attack modes:
  - brute_force:  failed login attempts spike with moderate traffic
  - flood:        traffic rate spikes with moderate login failures
  - adaptive:     dynamically switches strategy to evade defenses
  - multi_stage:  three-phase attack (reconnaissance → exploitation → persistence)
"""

import random
from collections import deque


class AttackSimulator:
    """Simulates different network attack patterns."""

    MODES = ("brute_force", "flood", "adaptive", "multi_stage")

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
        # multi_stage reuses params per phase (see step logic)
        "multi_stage": {
            "low":    ((10, 300),  (0, 50)),
            "medium": ((30, 700),  (0, 150)),
            "high":   ((60, 1000), (0, 300)),
        },
    }

    # Multi-stage phase definitions: (phase_name, traffic_scale, login_scale, duration_range)
    _MULTI_STAGE_PHASES = [
        ("reconnaissance", 0.15, 0.10, (15, 40)),
        ("exploitation",   0.70, 0.80, (30, 80)),
        ("persistence",    0.40, 0.50, (40, 80)),
    ]

    def __init__(
        self,
        mode: str = "brute_force",
        intensity: str = "medium",
        defender_history_len: int = 10,
    ):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        if intensity not in ("low", "medium", "high"):
            raise ValueError("intensity must be low, medium, or high")
        self.mode = mode
        self.intensity = intensity
        self._step_count = 0
        self._switch_interval = random.randint(10, 30)
        self._current_adaptive_mode = random.choice(["brute_force", "flood"])

        # For adaptive attacker: track defender's last K actions
        self._defender_history_len = defender_history_len
        self._defender_history: deque[int] = deque(maxlen=defender_history_len)

        # Multi-stage state
        self._phase_idx = 0
        self._phase_step = 0
        self._phase_duration = 0
        if mode == "multi_stage":
            self._init_phase(0)

    def _init_phase(self, idx: int):
        """Initialise a multi-stage attack phase."""
        idx = idx % len(self._MULTI_STAGE_PHASES)
        self._phase_idx = idx
        self._phase_step = 0
        dur_range = self._MULTI_STAGE_PHASES[idx][3]
        self._phase_duration = random.randint(*dur_range)

    @property
    def current_phase(self) -> str:
        """Return the name of the current multi-stage phase (or mode name)."""
        if self.mode == "multi_stage":
            return self._MULTI_STAGE_PHASES[self._phase_idx][0]
        return self.mode

    # ------------------------------------------------------------------
    def step(self, last_action: int) -> dict:
        """Return simulated attack metrics for one timestep.

        Args:
            last_action: The defensive action taken in the previous step.
                         0=no-op, 1=block, 2=rate-limit, 3=isolate

        Returns:
            dict with keys: traffic_rate, failed_logins, is_attacking, phase
        """
        self._step_count += 1
        self._defender_history.append(last_action)

        effective_mode = self.mode

        # --- Adaptive mode: uses defender history for smarter switching ---
        if self.mode == "adaptive":
            recent_blocks = sum(1 for a in self._defender_history if a in (1, 3))
            block_ratio = recent_blocks / max(len(self._defender_history), 1)

            if (
                self._step_count % self._switch_interval == 0
                or block_ratio > 0.5
            ):
                self._current_adaptive_mode = (
                    "flood"
                    if self._current_adaptive_mode == "brute_force"
                    else "brute_force"
                )
                self._switch_interval = random.randint(8, 25)
            effective_mode = self._current_adaptive_mode

        # --- Multi-stage mode ---
        if self.mode == "multi_stage":
            self._phase_step += 1
            if self._phase_step >= self._phase_duration:
                self._init_phase(self._phase_idx + 1)

            phase_name, t_scale, l_scale, _ = self._MULTI_STAGE_PHASES[self._phase_idx]
            base_params = self._PARAMS["multi_stage"][self.intensity]
            traffic_rate = random.uniform(*base_params[0]) * t_scale
            failed_logins = random.uniform(*base_params[1]) * l_scale
        else:
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
            "phase": self.current_phase,
        }

    def reset(self):
        """Reset internal state for a new episode."""
        self._step_count = 0
        self._switch_interval = random.randint(10, 30)
        self._current_adaptive_mode = random.choice(["brute_force", "flood"])
        self._defender_history.clear()
        if self.mode == "multi_stage":
            self._init_phase(0)
