"""
IntrusionEnv – Gymnasium-compatible cybersecurity defense environment.

MDP Formulation
===============
State Space S (normalised to [0, 1]):
    s_t = [failed_logins, request_rate, cpu_usage, memory_usage, alert_level, last_action]

    With temporal_window=N the observation is (6*N,) — stacking last N timesteps.

Action Space A (Discrete 4):
    0 = do_nothing
    1 = block_ip
    2 = rate_limit
    3 = isolate_service

Transition Function P(s'|s,a):
    Stochastic transitions driven by attacker behaviour, defender actions, and
    system dynamics.  Implemented via AttackSimulator + ResponseEngine.

Reward Function R(s,a):
    r = security_gain - service_disruption_cost - response_latency_penalty
    See _compute_reward() for the full multi-objective formulation.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


# ────────────────────────────────────────────────────────────────────
# ActionOutcome dataclass
# ────────────────────────────────────────────────────────────────────

@dataclass
class ActionOutcome:
    """Result of applying a defensive action."""
    action_id: int
    action_name: str
    threat_reduction: float
    service_cost: float


# ────────────────────────────────────────────────────────────────────
# ResponseEngine
# ────────────────────────────────────────────────────────────────────

class ResponseEngine:
    """Maps discrete actions to simulated defensive outcomes.

    Actions:
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

    _BASE_PARAMS = {
        0: (0.0, 0.00),
        1: (0.55, 0.10),
        2: (0.40, 0.15),
        3: (0.80, 0.40),
    }

    def apply(self, action_id: int, threat_level: float) -> ActionOutcome:
        if action_id not in self._BASE_PARAMS:
            raise ValueError(f"Unknown action_id {action_id}. Must be 0–3.")
        base_reduction, base_cost = self._BASE_PARAMS[action_id]
        effective_reduction = min(base_reduction * (0.5 + threat_level), 1.0)
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


# ────────────────────────────────────────────────────────────────────
# SystemMonitor
# ────────────────────────────────────────────────────────────────────

class SystemMonitor:
    """Reads live system metrics and computes a scalar threat-level score."""

    _THREAT_WEIGHTS = np.array([0.30, 0.30, 0.15, 0.10, 0.15], dtype=np.float32)

    def get_system_metrics(self) -> dict:
        if _HAS_PSUTIL:
            cpu = psutil.cpu_percent(interval=None) / 100.0
            mem = psutil.virtual_memory().percent / 100.0
        else:
            cpu, mem = 0.2, 0.3
        return {"cpu_usage": float(cpu), "memory_usage": float(mem)}

    def compute_threat_level(
        self,
        traffic_rate: float,
        failed_logins: float,
        cpu_usage: float,
        memory_usage: float,
        traffic_max: float = 1000.0,
        logins_max: float = 300.0,
    ) -> float:
        t = min(traffic_rate / traffic_max, 1.0)
        f = min(failed_logins / logins_max, 1.0)
        c = float(np.clip(cpu_usage, 0.0, 1.0))
        m = float(np.clip(memory_usage, 0.0, 1.0))
        spike = min(t * f + 0.5 * c, 1.0)
        features = np.array([t, f, c, m, spike], dtype=np.float32)
        threat = float(np.dot(self._THREAT_WEIGHTS, features))
        return float(np.clip(threat, 0.0, 1.0))


# ────────────────────────────────────────────────────────────────────
# AttackSimulator
# ────────────────────────────────────────────────────────────────────

class AttackSimulator:
    """Generates synthetic attack traffic patterns.

    Attack Modes:
        brute_force  – failed login attempts spike
        flood        – traffic rate spikes
        adaptive     – switches strategy based on defender history
        multi_stage  – 3 phases: reconnaissance → exploitation → persistence
    """

    MODES = ("brute_force", "flood", "adaptive", "multi_stage")

    _PARAMS = {
        "brute_force": {
            "low": ((10, 30), (20, 50)),
            "medium": ((30, 60), (50, 150)),
            "high": ((60, 100), (150, 300)),
        },
        "flood": {
            "low": ((100, 300), (0, 5)),
            "medium": ((300, 700), (0, 10)),
            "high": ((700, 1000), (0, 15)),
        },
        "adaptive": {
            "low": ((10, 300), (0, 50)),
            "medium": ((30, 700), (0, 150)),
            "high": ((60, 1000), (0, 300)),
        },
        "multi_stage": {
            "low": ((10, 300), (0, 50)),
            "medium": ((30, 700), (0, 150)),
            "high": ((60, 1000), (0, 300)),
        },
    }

    _MULTI_STAGE_PHASES = [
        ("reconnaissance", 0.15, 0.10, (15, 40)),
        ("exploitation", 0.70, 0.80, (30, 80)),
        ("persistence", 0.40, 0.50, (40, 80)),
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
        self._defender_history_len = defender_history_len
        self._defender_history: deque[int] = deque(maxlen=defender_history_len)
        self._phase_idx = 0
        self._phase_step = 0
        self._phase_duration = 0
        if mode == "multi_stage":
            self._init_phase(0)

    def _init_phase(self, idx: int):
        idx = idx % len(self._MULTI_STAGE_PHASES)
        self._phase_idx = idx
        self._phase_step = 0
        dur_range = self._MULTI_STAGE_PHASES[idx][3]
        self._phase_duration = random.randint(*dur_range)

    @property
    def current_phase(self) -> str:
        if self.mode == "multi_stage":
            return self._MULTI_STAGE_PHASES[self._phase_idx][0]
        return self.mode

    def step(self, last_action: int) -> dict:
        self._step_count += 1
        self._defender_history.append(last_action)
        effective_mode = self.mode

        if self.mode == "adaptive":
            recent_blocks = sum(1 for a in self._defender_history if a in (1, 3))
            block_ratio = recent_blocks / max(len(self._defender_history), 1)
            if self._step_count % self._switch_interval == 0 or block_ratio > 0.5:
                self._current_adaptive_mode = (
                    "flood" if self._current_adaptive_mode == "brute_force" else "brute_force"
                )
                self._switch_interval = random.randint(8, 25)
            effective_mode = self._current_adaptive_mode

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

        # Attacker backs off after strong defense
        if last_action == 3:
            traffic_rate *= 0.2
            failed_logins *= 0.2
        elif last_action == 1:
            failed_logins *= 0.3
        elif last_action == 2:
            traffic_rate *= 0.4

        return {
            "traffic_rate": max(0.0, traffic_rate),
            "failed_logins": max(0.0, failed_logins),
            "is_attacking": True,
            "phase": self.current_phase,
        }

    def reset(self):
        self._step_count = 0
        self._switch_interval = random.randint(10, 30)
        self._current_adaptive_mode = random.choice(["brute_force", "flood"])
        self._defender_history.clear()
        if self.mode == "multi_stage":
            self._init_phase(0)


# ────────────────────────────────────────────────────────────────────
# IntrusionEnv  (Gymnasium Environment)
# ────────────────────────────────────────────────────────────────────

class IntrusionEnv(gym.Env):
    """Gymnasium environment simulating a network under attack.

    State Space (6 features per timestep, all normalised to [0, 1]):
        0  failed_logins   – normalised failed-login count
        1  request_rate    – normalised packet/request rate
        2  cpu_usage       – CPU utilisation
        3  memory_usage    – memory utilisation
        4  alert_level     – composite threat score
        5  last_action     – previous defensive action (normalised)

    Action Space (Discrete 4):
        0  do_nothing
        1  block_ip
        2  rate_limit
        3  isolate_service
    """

    metadata = {"render_modes": []}

    TRAFFIC_MAX = 1000.0
    LOGINS_MAX = 300.0
    MAX_STEPS = 200

    HIGH_THREAT_THRESHOLD = 0.6
    LOW_THREAT_THRESHOLD = 0.2

    def __init__(
        self,
        attack_mode: str = "brute_force",
        intensity: str = "medium",
        use_real_system_metrics: bool = False,
        noisy_observations: bool = False,
        noise_std: float = 0.05,
        partial_observability: bool = False,
        mask_probability: float = 0.1,
        action_cooldown: int = 0,
        delayed_effect_steps: int = 0,
        resource_budget: Optional[int] = None,
        temporal_window: int = 1,
        reward_cfg: Optional[dict] = None,
    ):
        super().__init__()

        self.attack_mode = attack_mode
        self.intensity = intensity
        self.use_real_system_metrics = use_real_system_metrics

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

        rcfg = reward_cfg or {}
        self._threat_w = rcfg.get("threat_weight", 10.0)
        self._service_w = rcfg.get("service_cost_weight", 5.0)
        self._fp_w = rcfg.get("false_positive_weight", 3.0)
        self._ineff_w = rcfg.get("ineffective_weight", 5.0)
        self._survival = rcfg.get("survival_bonus", 0.1)
        self._latency_pen = rcfg.get("response_latency_penalty", 0.0)
        self._downtime_thresh = rcfg.get("downtime_threshold", 0.5)
        self._downtime_pen = rcfg.get("downtime_penalty", 2.0)

        obs_dim = 6 * self.temporal_window
        self.observation_space = spaces.Box(
            low=np.zeros(obs_dim, dtype=np.float32),
            high=np.ones(obs_dim, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(4)

        self._last_action: int = 0
        self._step_count: int = 0
        self._episode_reward: float = 0.0
        self._obs_buffer: deque[np.ndarray] = deque(maxlen=self.temporal_window)
        self._cooldown_counter: int = 0
        self._actions_used: int = 0
        self._pending_actions: deque = deque()
        self._cumulative_service_cost: float = 0.0
        self._steps_since_first_high_threat: int = -1
        self._first_detection_step: int = -1

    # ──────────── Internal helpers ────────────

    def _get_single_obs(self) -> np.ndarray:
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
            traffic, logins, cpu, mem, self.TRAFFIC_MAX, self.LOGINS_MAX,
        )

        # Observation: [failed_logins, request_rate, cpu, mem, alert_level, last_action]
        obs = np.array(
            [
                logins / self.LOGINS_MAX,
                traffic / self.TRAFFIC_MAX,
                cpu,
                mem,
                threat,
                self._last_action / max(self._responder.num_actions - 1, 1),
            ],
            dtype=np.float32,
        )
        obs = np.clip(obs, 0.0, 1.0)

        if self.noisy_observations:
            noise = self.np_random.normal(0.0, self.noise_std, size=obs.shape).astype(np.float32)
            obs = np.clip(obs + noise, 0.0, 1.0)

        if self.partial_observability:
            mask = self.np_random.random(size=obs.shape) > self.mask_probability
            obs = obs * mask.astype(np.float32)

        self._current_threat = threat
        self._current_phase = attack_metrics.get("phase", self.attack_mode)
        return obs

    def _get_obs(self) -> np.ndarray:
        single = self._get_single_obs()
        self._obs_buffer.append(single)
        while len(self._obs_buffer) < self.temporal_window:
            self._obs_buffer.appendleft(np.zeros(6, dtype=np.float32))
        return np.concatenate(list(self._obs_buffer))

    def _apply_action(self, action: int) -> int:
        if self.resource_budget is not None and action > 0:
            if self._actions_used >= self.resource_budget:
                action = 0
        if self.action_cooldown > 0 and action > 0:
            if self._cooldown_counter > 0:
                action = 0
            else:
                self._cooldown_counter = self.action_cooldown
        if action > 0:
            self._actions_used += 1
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1
        if self.delayed_effect_steps > 0 and action > 0:
            self._pending_actions.append(
                (self._step_count + self.delayed_effect_steps, action)
            )
        return action

    def _compute_reward(self, action: int, outcome: ActionOutcome) -> float:
        """Multi-objective reward:
            reward = security_gain
                   - service_disruption_cost
                   - false_positive_penalty
                   - ineffective_penalty
                   - response_latency_penalty
                   - downtime_threshold_penalty
                   + survival_bonus

        Reward Table:
            Event                  |  Reward contribution
            ─────────────────────  |  ────────────────────
            attack blocked         |  +10  (threat_reduction × threat_weight)
            successful mitigation  |  +15  (high reduction at high threat)
            false positive         |  −5   (action on low threat)
            service downtime       |  −10  (cumulative cost exceeds threshold)
            attack success         |  −15  (no action at high threat)
        """
        threat = self._current_threat

        threat_reduction_reward = outcome.threat_reduction * self._threat_w
        service_penalty = outcome.service_cost * self._service_w

        if threat < self.LOW_THREAT_THRESHOLD and action > 0:
            false_positive_penalty = (action / 3.0) * self._fp_w
        else:
            false_positive_penalty = 0.0

        if threat > self.HIGH_THREAT_THRESHOLD and action == 0:
            ineffective_penalty = threat * self._ineff_w
        else:
            ineffective_penalty = 0.0

        latency_penalty = 0.0
        if self._latency_pen > 0 and threat > self.HIGH_THREAT_THRESHOLD:
            if self._first_detection_step < 0:
                self._steps_since_first_high_threat += 1
                latency_penalty = self._steps_since_first_high_threat * self._latency_pen
            elif action > 0 and self._first_detection_step < 0:
                self._first_detection_step = self._step_count

        downtime_penalty = 0.0
        self._cumulative_service_cost += outcome.service_cost
        if self._cumulative_service_cost > self._downtime_thresh:
            downtime_penalty = self._downtime_pen

        reward = (
            threat_reduction_reward
            - service_penalty
            - false_positive_penalty
            - ineffective_penalty
            - latency_penalty
            - downtime_penalty
            + self._survival
        )
        return float(reward)

    # ──────────── Gymnasium API ────────────

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
            "episode_reward": self._episode_reward,
            "step": self._step_count,
            "phase": self._current_phase,
            "cumulative_service_cost": self._cumulative_service_cost,
            "actions_used": self._actions_used,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        pass
