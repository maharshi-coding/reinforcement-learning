"""
NetworkSecurityEnv – Gymnasium-compatible environment for AIRS.

State Space (6-dimensional, all normalised to [0, 1])
------------------------------------------------------
0  traffic_rate        – normalised packet/request rate
1  failed_logins       – normalised failed-login count
2  cpu_usage           – CPU utilisation
3  memory_usage        – memory utilisation
4  threat_level        – composite threat score
5  last_action         – previous defensive action (normalised)

Action Space (Discrete 4)
-------------------------
0  No-op
1  Block IP
2  Rate limit
3  Isolate service

Reward Function
---------------
r = +threat_reduction * 10          (positive: mitigating the threat)
  - service_cost * 5                 (negative: disruption cost)
  - false_positive_penalty           (negative: acting when threat is low)
  - ineffective_penalty              (negative: no-op under high threat)
  + survival_bonus (0.1 per step)    (encourage staying alive)

Discount factor γ = 0.99 (recommended to allow delayed reward propagation)
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from airs.environment.attack_simulator import AttackSimulator
from airs.monitoring.monitor import SystemMonitor
from airs.response.response_engine import ResponseEngine


class NetworkSecurityEnv(gym.Env):
    """Gym-style environment simulating a network under attack."""

    metadata = {"render_modes": []}

    # Normalisation constants
    TRAFFIC_MAX = 1000.0
    LOGINS_MAX = 300.0

    # Episode length
    MAX_STEPS = 200

    # Threat thresholds
    HIGH_THREAT_THRESHOLD = 0.6
    LOW_THREAT_THRESHOLD = 0.2

    def __init__(
        self,
        attack_mode: str = "brute_force",
        intensity: str = "medium",
        use_real_system_metrics: bool = False,
    ):
        super().__init__()

        self.attack_mode = attack_mode
        self.intensity = intensity
        self.use_real_system_metrics = use_real_system_metrics

        self._attacker = AttackSimulator(mode=attack_mode, intensity=intensity)
        self._monitor = SystemMonitor()
        self._responder = ResponseEngine()

        # Observation: 6 floats in [0, 1]
        self.observation_space = spaces.Box(
            low=np.zeros(6, dtype=np.float32),
            high=np.ones(6, dtype=np.float32),
            dtype=np.float32,
        )

        # 4 discrete actions
        self.action_space = spaces.Discrete(4)

        self._last_action: int = 0
        self._step_count: int = 0
        self._episode_reward: float = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Build the normalised observation vector."""
        attack_metrics = self._attacker.step(self._last_action)
        traffic = attack_metrics["traffic_rate"]
        logins = attack_metrics["failed_logins"]

        if self.use_real_system_metrics:
            sys_metrics = self._monitor.get_system_metrics()
            cpu = sys_metrics["cpu_usage"]
            mem = sys_metrics["memory_usage"]
        else:
            # Simulate system load proportional to attack intensity
            cpu = float(np.clip(0.2 + traffic / self.TRAFFIC_MAX * 0.6, 0.0, 1.0))
            mem = float(np.clip(0.3 + logins / self.LOGINS_MAX * 0.4, 0.0, 1.0))

        threat = self._monitor.compute_threat_level(
            traffic, logins, cpu, mem,
            self.TRAFFIC_MAX, self.LOGINS_MAX,
        )

        obs = np.array(
            [
                traffic / self.TRAFFIC_MAX,
                logins / self.LOGINS_MAX,
                cpu,
                mem,
                threat,
                self._last_action / (self._responder.num_actions - 1),
            ],
            dtype=np.float32,
        )
        # Cache threat for reward computation
        self._current_threat = threat
        return np.clip(obs, 0.0, 1.0)

    def _compute_reward(self, action: int, outcome) -> float:
        """Compute the step reward.

        r = threat_reduction_reward
          - service_disruption_penalty
          - false_positive_penalty
          - ineffective_penalty
          + survival_bonus
        """
        threat = self._current_threat

        # Primary reward: how much threat was reduced
        threat_reduction_reward = outcome.threat_reduction * 10.0

        # Cost of disrupting service
        service_penalty = outcome.service_cost * 5.0

        # Penalise aggressive action when threat is actually low (false positive)
        if threat < self.LOW_THREAT_THRESHOLD and action > 0:
            false_positive_penalty = (action / 3.0) * 3.0
        else:
            false_positive_penalty = 0.0

        # Penalise inaction when threat is high
        if threat > self.HIGH_THREAT_THRESHOLD and action == 0:
            ineffective_penalty = threat * 5.0
        else:
            ineffective_penalty = 0.0

        # Small survival bonus each step (encourages episode completion)
        survival_bonus = 0.1

        reward = (
            threat_reduction_reward
            - service_penalty
            - false_positive_penalty
            - ineffective_penalty
            + survival_bonus
        )
        return float(reward)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._attacker.reset()
        self._last_action = 0
        self._step_count = 0
        self._episode_reward = 0.0
        self._current_threat = 0.0
        obs = self._get_obs()
        return obs, {}

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action {action}"

        self._step_count += 1
        outcome = self._responder.apply(action, self._current_threat)
        self._last_action = action

        obs = self._get_obs()
        reward = self._compute_reward(action, outcome)
        self._episode_reward += reward

        terminated = False
        truncated = self._step_count >= self.MAX_STEPS

        info = {
            "action_name": outcome.action_name,
            "threat_level": self._current_threat,
            "threat_reduction": outcome.threat_reduction,
            "service_cost": outcome.service_cost,
            "episode_reward": self._episode_reward,
            "step": self._step_count,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        pass
