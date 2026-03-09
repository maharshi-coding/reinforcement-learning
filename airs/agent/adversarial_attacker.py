"""
Adversarial Attacker Agent for AIRS Self-Play.

A second RL agent trained to *maximize* damage to the network while
the defender agent tries to minimize it.  The attacker controls traffic
and login patterns; the defender selects defensive actions.

Self-play loop:
  1. Attacker picks an attack strategy for the next N steps.
  2. Defender observes the resulting metrics and chooses actions.
  3. Attacker is rewarded when the defender fails; defender is rewarded
     when the attack is mitigated.
  Both agents improve simultaneously.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from airs.environment.network_env import NetworkSecurityEnv
from airs.monitoring.monitor import SystemMonitor
from airs.response.response_engine import ResponseEngine

logger = logging.getLogger("airs.adversarial")


# ======================================================================
# Attacker Environment  (wraps around a frozen or live defender)
# ======================================================================

class AttackerEnv(gym.Env):
    """Gymnasium env where the RL agent plays as the *attacker*.

    Observation  (6-d, normalised [0,1])
    -----------
    0  last_traffic_rate    – traffic rate produced last step
    1  last_failed_logins   – failed logins produced last step
    2  cpu_usage            – simulated CPU load
    3  memory_usage         – simulated memory load
    4  defender_last_action – last defensive action (normalised)
    5  threat_level         – current threat level

    Action Space (Discrete 6)
    -------------------------
    Combination of intensity × focus:
      0  low traffic  / low logins       (stealth)
      1  low traffic  / high logins      (brute force)
      2  high traffic / low logins       (flood)
      3  high traffic / high logins      (full assault)
      4  medium traffic / medium logins  (balanced)
      5  back-off / quiet               (evasion)
    """

    metadata = {"render_modes": []}

    TRAFFIC_MAX = 1000.0
    LOGINS_MAX = 300.0
    MAX_STEPS = 200

    # Each attack action → (traffic_range, logins_range)
    _ATTACK_PROFILES: dict[int, tuple[tuple[float, float], tuple[float, float]]] = {
        0: ((10, 50),   (5, 20)),     # stealth
        1: ((20, 80),   (100, 300)),   # brute force
        2: ((500, 1000), (0, 10)),     # flood
        3: ((500, 1000), (100, 300)),  # full assault
        4: ((100, 400), (30, 100)),    # balanced
        5: ((0, 10),    (0, 5)),       # evasion / back-off
    }

    def __init__(
        self,
        defender_predict_fn=None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            defender_predict_fn: Callable(obs_6d) -> int action.
                If None, a random defender is used.
        """
        super().__init__()

        self._defender_fn = defender_predict_fn or (lambda _obs: np.random.randint(0, 4))
        self._monitor = SystemMonitor()
        self._responder = ResponseEngine(stochastic=True, seed=seed)

        self.observation_space = spaces.Box(
            low=np.zeros(6, dtype=np.float32),
            high=np.ones(6, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(6)

        self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._last_traffic = 0.0
        self._last_logins = 0.0
        self._defender_last_action = 0
        self._threat = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._last_traffic = 0.0
        self._last_logins = 0.0
        self._defender_last_action = 0
        self._threat = 0.0
        return self._build_obs(), {}

    def _build_obs(self) -> np.ndarray:
        cpu = float(np.clip(0.2 + self._last_traffic / self.TRAFFIC_MAX * 0.6, 0, 1))
        mem = float(np.clip(0.3 + self._last_logins / self.LOGINS_MAX * 0.4, 0, 1))
        obs = np.array([
            self._last_traffic / self.TRAFFIC_MAX,
            self._last_logins / self.LOGINS_MAX,
            cpu, mem,
            self._defender_last_action / 3.0,
            self._threat,
        ], dtype=np.float32)
        return np.clip(obs, 0.0, 1.0)

    def step(self, attack_action: int):
        self._step_count += 1

        # --- Generate traffic based on attacker's choice ---
        profile = self._ATTACK_PROFILES[int(attack_action)]
        traffic = float(self._rng.uniform(*profile[0]))
        logins = float(self._rng.uniform(*profile[1]))

        # Defender response may reduce the attack's effect
        if self._defender_last_action == 3:  # isolate
            traffic *= 0.2
            logins *= 0.2
        elif self._defender_last_action == 1:  # block
            logins *= 0.3
        elif self._defender_last_action == 2:  # rate-limit
            traffic *= 0.4

        self._last_traffic = max(0.0, traffic)
        self._last_logins = max(0.0, logins)

        cpu = float(np.clip(0.2 + traffic / self.TRAFFIC_MAX * 0.6, 0, 1))
        mem = float(np.clip(0.3 + logins / self.LOGINS_MAX * 0.4, 0, 1))

        self._threat = self._monitor.compute_threat_level(
            traffic, logins, cpu, mem,
            self.TRAFFIC_MAX, self.LOGINS_MAX,
        )

        # --- Ask defender what it does ---
        defender_obs = np.array([
            traffic / self.TRAFFIC_MAX,
            logins / self.LOGINS_MAX,
            cpu, mem,
            self._threat,
            self._defender_last_action / 3.0,
        ], dtype=np.float32)
        defender_obs = np.clip(defender_obs, 0.0, 1.0)

        defender_action = int(self._defender_fn(defender_obs))
        outcome = self._responder.apply(defender_action, self._threat)
        self._defender_last_action = defender_action

        # --- Attacker reward: opposite of defender reward ---
        # High threat that isn't mitigated = attacker success
        # Defender service cost = attacker gain
        attacker_reward = (
            self._threat * 5.0                       # high threat is good for attacker
            - outcome.threat_reduction * 8.0          # effective defense hurts attacker
            + outcome.service_cost * 3.0              # defender disruption helps attacker
        )

        # Penalise evasion slightly to encourage attacking
        if attack_action == 5:
            attacker_reward -= 0.5

        obs = self._build_obs()
        truncated = self._step_count >= self.MAX_STEPS

        info = {
            "threat_level": self._threat,
            "defender_action": defender_action,
            "defender_success": outcome.success,
            "traffic_rate": self._last_traffic,
            "failed_logins": self._last_logins,
        }

        return obs, float(attacker_reward), False, truncated, info


# ======================================================================
# Self-Play Training Loop
# ======================================================================

class SelfPlayTrainer:
    """Alternating self-play between a PPO defender and a PPO attacker.

    Each round:
      1. Freeze the attacker, train the defender for ``defender_steps``.
      2. Freeze the defender, train the attacker for ``attacker_steps``.
    Repeat for ``rounds`` iterations.
    """

    DEFENDER_PPO_KWARGS: dict[str, Any] = {
        "learning_rate": 2.5e-4,
        "n_steps": 1024,
        "batch_size": 128,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "policy_kwargs": {"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
        "device": "cpu",
        "verbose": 0,
    }

    ATTACKER_PPO_KWARGS: dict[str, Any] = {
        "learning_rate": 3e-4,
        "n_steps": 1024,
        "batch_size": 128,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.02,  # slightly higher to encourage exploration
        "policy_kwargs": {"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
        "device": "cpu",
        "verbose": 0,
    }

    def __init__(
        self,
        rounds: int = 10,
        defender_steps: int = 20_000,
        attacker_steps: int = 20_000,
        seed: int = 42,
    ):
        self.rounds = rounds
        self.defender_steps = defender_steps
        self.attacker_steps = attacker_steps
        self.seed = seed

        # Build defender env (standard AIRS env with a scripted attacker for now)
        self._defender_env = DummyVecEnv([
            lambda: Monitor(NetworkSecurityEnv(attack_mode="adaptive", intensity="medium"))
        ])

        self._defender = PPO("MlpPolicy", self._defender_env, seed=seed, **self.DEFENDER_PPO_KWARGS)
        self._attacker: Optional[PPO] = None

        self.history: list[dict] = []

    def _build_attacker_env(self, defender_predict_fn):
        """Create an attacker env with the current defender frozen."""
        def _init():
            return Monitor(AttackerEnv(defender_predict_fn=defender_predict_fn, seed=self.seed))
        return DummyVecEnv([_init])

    def _build_adversarial_defender_env(self, attacker_predict_fn):
        """Create a defender env driven by the learned attacker.

        We wrap NetworkSecurityEnv but override internal attack generation
        with queries to the trained attacker.  For simplicity we use the
        AttackerEnv in "reverse" — the defender is the agent being trained.
        """
        from airs.environment.network_env import NetworkSecurityEnv as _Env

        class _AdversarialDefenderEnv(_Env):
            """NetworkSecurityEnv whose attack traffic is generated by the attacker."""

            def __init__(self, atk_fn, **kwargs):
                super().__init__(**kwargs)
                self._atk_fn = atk_fn
                self._atk_obs = np.zeros(6, dtype=np.float32)

            def _get_single_obs(self):
                # Ask the attacker for its desired attack profile
                atk_action = int(self._atk_fn(self._atk_obs))
                profile = AttackerEnv._ATTACK_PROFILES.get(atk_action, ((50, 200), (10, 50)))
                rng = np.random.default_rng()
                traffic = float(rng.uniform(*profile[0]))
                logins = float(rng.uniform(*profile[1]))

                # Defender response dampens attack
                effective_action = self._last_action
                if effective_action == 3:
                    traffic *= 0.2
                    logins *= 0.2
                elif effective_action == 1:
                    logins *= 0.3
                elif effective_action == 2:
                    traffic *= 0.4

                if self.use_real_system_metrics:
                    sys_metrics = self._monitor.get_system_metrics()
                    cpu, mem = sys_metrics["cpu_usage"], sys_metrics["memory_usage"]
                else:
                    cpu = float(np.clip(0.2 + traffic / self.TRAFFIC_MAX * 0.6, 0, 1))
                    mem = float(np.clip(0.3 + logins / self.LOGINS_MAX * 0.4, 0, 1))

                threat = self._monitor.compute_threat_level(
                    traffic, logins, cpu, mem,
                    self.TRAFFIC_MAX, self.LOGINS_MAX,
                )
                obs = np.array([
                    traffic / self.TRAFFIC_MAX,
                    logins / self.LOGINS_MAX,
                    cpu, mem, threat,
                    self._last_action / max(self._responder.num_actions - 1, 1),
                ], dtype=np.float32)
                obs = np.clip(obs, 0.0, 1.0)

                self._current_threat = threat
                self._current_phase = "adversarial"

                # Update attacker's observation so it can see defender's state
                self._atk_obs = obs.copy()
                return obs

        def _init():
            env = _AdversarialDefenderEnv(
                atk_fn=attacker_predict_fn,
                attack_mode="adaptive", intensity="medium",
            )
            return Monitor(env)

        return DummyVecEnv([_init])

    def train(self) -> dict:
        """Run the full self-play training loop.

        Returns:
            Dictionary with training history.
        """
        for rnd in range(1, self.rounds + 1):
            logger.info("=== Self-play round %d/%d ===", rnd, self.rounds)

            # --- Phase 1: Train defender (attacker frozen or scripted) ---
            if self._attacker is not None:
                def _atk_fn(obs):
                    a, _ = self._attacker.predict(obs, deterministic=False)
                    return int(a)
                adv_env = self._build_adversarial_defender_env(_atk_fn)
                self._defender.set_env(adv_env)

            self._defender.learn(total_timesteps=self.defender_steps, reset_num_timesteps=False)
            logger.info("  Defender trained for %d steps", self.defender_steps)

            # --- Phase 2: Train attacker (defender frozen) ---
            def _def_fn(obs):
                a, _ = self._defender.predict(obs, deterministic=False)
                return int(a)

            atk_env = self._build_attacker_env(_def_fn)
            if self._attacker is None:
                self._attacker = PPO("MlpPolicy", atk_env, seed=self.seed + 1, **self.ATTACKER_PPO_KWARGS)
            else:
                self._attacker.set_env(atk_env)

            self._attacker.learn(total_timesteps=self.attacker_steps, reset_num_timesteps=False)
            logger.info("  Attacker trained for %d steps", self.attacker_steps)

            self.history.append({"round": rnd})

        return {"rounds": self.rounds, "history": self.history}

    def save(self, defender_path: str, attacker_path: str):
        """Save both agents to disk."""
        import os
        os.makedirs(os.path.dirname(defender_path) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(attacker_path) or ".", exist_ok=True)
        self._defender.save(defender_path)
        if self._attacker:
            self._attacker.save(attacker_path)

    @property
    def defender(self) -> PPO:
        return self._defender

    @property
    def attacker(self) -> Optional[PPO]:
        return self._attacker
