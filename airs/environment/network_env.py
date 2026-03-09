"""
NetworkSecurityEnv – Gymnasium-compatible environment for AIRS.

State Space (variable-dimensional, all normalised to [0, 1])
-------------------------------------------------------------
Base observation (6 features per timestep):
0  traffic_rate        – normalised packet/request rate
1  failed_logins       – normalised failed-login count
2  cpu_usage           – CPU utilisation
3  memory_usage        – memory utilisation
4  threat_level        – composite threat score
5  last_action         – previous defensive action (normalised)

With temporal_window=N the observation is (6*N,) — stacking last N timesteps.

Action Space (Discrete 4)
-------------------------
0  No-op
1  Block IP
2  Rate limit
3  Isolate service

Reward Function
---------------
r = +threat_reduction × threat_weight
  - service_cost × service_cost_weight
  - false_positive_penalty
  - ineffective_penalty
  + survival_bonus
  - response_latency_penalty          (new)
  - downtime_threshold_penalty        (new)
"""

from __future__ import annotations

from collections import deque
from typing import Any, Optional

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
        # --- new realism knobs ---
        noisy_observations: bool = False,
        noise_std: float = 0.05,
        partial_observability: bool = False,
        mask_probability: float = 0.1,
        action_cooldown: int = 0,
        delayed_effect_steps: int = 0,
        resource_budget: Optional[int] = None,
        temporal_window: int = 1,
        # --- reward knobs ---
        reward_cfg: Optional[dict] = None,
    ):
        super().__init__()

        self.attack_mode = attack_mode
        self.intensity = intensity
        self.use_real_system_metrics = use_real_system_metrics

        # Realism features
        self.noisy_observations = noisy_observations
        self.noise_std = noise_std
        self.partial_observability = partial_observability
        self.mask_probability = mask_probability
        self.action_cooldown = action_cooldown
        self.delayed_effect_steps = delayed_effect_steps
        self.resource_budget = resource_budget
        self.temporal_window = max(1, temporal_window)

        self._attacker = AttackSimulator(mode=attack_mode, intensity=intensity)
        self._monitor = SystemMonitor()
        self._responder = ResponseEngine()

        # Reward config (defaults match original behaviour)
        rcfg = reward_cfg or {}
        self._threat_w = rcfg.get("threat_weight", 10.0)
        self._service_w = rcfg.get("service_cost_weight", 5.0)
        self._fp_w = rcfg.get("false_positive_weight", 3.0)
        self._ineff_w = rcfg.get("ineffective_weight", 5.0)
        self._survival = rcfg.get("survival_bonus", 0.1)
        self._latency_pen = rcfg.get("response_latency_penalty", 0.0)
        self._downtime_thresh = rcfg.get("downtime_threshold", 0.5)
        self._downtime_pen = rcfg.get("downtime_penalty", 2.0)

        # Observation: 6 * temporal_window floats in [0, 1]
        obs_dim = 6 * self.temporal_window
        self.observation_space = spaces.Box(
            low=np.zeros(obs_dim, dtype=np.float32),
            high=np.ones(obs_dim, dtype=np.float32),
            dtype=np.float32,
        )

        # 4 discrete actions
        self.action_space = spaces.Discrete(4)

        self._last_action: int = 0
        self._step_count: int = 0
        self._episode_reward: float = 0.0

        # Temporal window buffer
        self._obs_buffer: deque[np.ndarray] = deque(maxlen=self.temporal_window)

        # Operational constraints state
        self._cooldown_counter: int = 0
        self._actions_used: int = 0
        self._pending_actions: deque = deque()
        self._cumulative_service_cost: float = 0.0
        self._steps_since_first_high_threat: int = -1
        self._first_detection_step: int = -1
        self._breach_progress: float = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_single_obs(self) -> np.ndarray:
        """Build one 6-d normalised observation vector."""
        # Determine effective action (accounting for delays)
        effective_action = self._last_action
        if self.delayed_effect_steps > 0 and self._pending_actions:
            if self._pending_actions[0][0] <= self._step_count:
                _, ea = self._pending_actions.popleft()
                effective_action = ea

        attack_metrics = self._attacker.step(effective_action)
        traffic = attack_metrics["traffic_rate"]
        logins = attack_metrics["failed_logins"]

        if self.use_real_system_metrics:
            sys_metrics = self._monitor.get_system_metrics()
            cpu = sys_metrics["cpu_usage"]
            mem = sys_metrics["memory_usage"]
        else:
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
                self._last_action / max(self._responder.num_actions - 1, 1),
            ],
            dtype=np.float32,
        )
        obs = np.clip(obs, 0.0, 1.0)

        # Noisy observations
        if self.noisy_observations:
            noise = self.np_random.normal(0.0, self.noise_std, size=obs.shape).astype(np.float32)
            obs = np.clip(obs + noise, 0.0, 1.0)

        # Partial observability: randomly mask some features
        if self.partial_observability:
            mask = self.np_random.random(size=obs.shape) > self.mask_probability
            obs = obs * mask.astype(np.float32)

        self._current_threat = threat
        self._current_phase = attack_metrics.get("phase", self.attack_mode)
        return obs

    def _get_obs(self) -> np.ndarray:
        """Return the (possibly stacked) observation."""
        single = self._get_single_obs()
        self._obs_buffer.append(single)

        # Pad if we don't have enough history yet
        while len(self._obs_buffer) < self.temporal_window:
            self._obs_buffer.appendleft(np.zeros(6, dtype=np.float32))

        return np.concatenate(list(self._obs_buffer))

    def _apply_action(self, action: int) -> int:
        """Apply operational constraints and return the effective action."""
        # Resource budget
        if self.resource_budget is not None and action > 0:
            if self._actions_used >= self.resource_budget:
                action = 0  # forced no-op

        # Cooldown
        if self.action_cooldown > 0 and action > 0:
            if self._cooldown_counter > 0:
                action = 0  # forced no-op
            else:
                self._cooldown_counter = self.action_cooldown

        if action > 0:
            self._actions_used += 1

        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1

        # Delayed effects
        if self.delayed_effect_steps > 0 and action > 0:
            self._pending_actions.append(
                (self._step_count + self.delayed_effect_steps, action)
            )

        return action

    def _compute_reward(self, action: int, outcome: Any) -> float:
        """Compute the step reward with configurable weights.

        Core idea: ignoring attacks causes breach damage that escalates
        over time, making 'do nothing' non-viable under persistent attack.
        Taking action during calm periods incurs false-positive cost.
        """
        threat = self._current_threat

        threat_reduction_reward = outcome.threat_reduction * self._threat_w
        service_penalty = outcome.service_cost * self._service_w

        # --- False positive penalty: only when threat is genuinely low ---
        if threat < self.LOW_THREAT_THRESHOLD and action > 0:
            false_positive_penalty = (action / 3.0) * self._fp_w
        else:
            false_positive_penalty = 0.0

        # --- Unnecessary action penalty: defensive actions during calm ---
        # Encourages selective behavior — don't block/isolate when no active threat
        unnecessary_penalty = 0.0
        if action > 0 and threat < 0.1:
            # Heavier actions (isolate > rate_limit > block) get larger penalty
            unnecessary_penalty = (action / 3.0) * 1.5

        # --- Breach damage: doing nothing lets the attack progress ---
        # Threat accumulates as breach_progress; defensive actions reduce it.
        if action == 0:
            self._breach_progress = min(self._breach_progress + threat, 3.0)
        else:
            self._breach_progress = max(0.0, self._breach_progress - outcome.threat_reduction)

        breach_penalty = self._breach_progress * 1.0

        # --- Legacy ineffective penalty for very high threat ---
        if threat > self.HIGH_THREAT_THRESHOLD and action == 0:
            ineffective_penalty = threat * self._ineff_w
        else:
            ineffective_penalty = 0.0

        # --- Response latency penalty ---
        latency_penalty = 0.0
        if self._latency_pen > 0 and threat > self.HIGH_THREAT_THRESHOLD:
            if self._first_detection_step < 0:
                self._steps_since_first_high_threat += 1
                latency_penalty = self._steps_since_first_high_threat * self._latency_pen
            elif action > 0 and self._first_detection_step < 0:
                self._first_detection_step = self._step_count

        # --- Downtime threshold penalty ---
        downtime_penalty = 0.0
        self._cumulative_service_cost += outcome.service_cost
        if self._cumulative_service_cost > self._downtime_thresh:
            downtime_penalty = self._downtime_pen

        # --- Survival bonus: reduced when under threat ---
        survival_bonus = self._survival * (1.0 - threat)

        reward = (
            threat_reduction_reward
            - service_penalty
            - false_positive_penalty
            - unnecessary_penalty
            - ineffective_penalty
            - breach_penalty
            - latency_penalty
            - downtime_penalty
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
        self._current_phase = self.attack_mode
        self._obs_buffer.clear()
        self._cooldown_counter = 0
        self._actions_used = 0
        self._pending_actions.clear()
        self._cumulative_service_cost = 0.0
        self._steps_since_first_high_threat = -1
        self._first_detection_step = -1
        self._breach_progress = 0.0
        obs = self._get_obs()
        return obs, {}

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action {action}"

        self._step_count += 1
        action = self._apply_action(action)
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
            "action_success": outcome.success,
            "episode_reward": self._episode_reward,
            "step": self._step_count,
            "phase": self._current_phase,
            "cumulative_service_cost": self._cumulative_service_cost,
            "actions_used": self._actions_used,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        pass
