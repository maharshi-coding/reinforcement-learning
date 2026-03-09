"""
MultiScenarioEnv – wrapper that randomises attack_mode and intensity each episode.

This allows a single agent to train across all 12 combinations of
(4 attack modes × 3 intensity levels), producing a robust generalist policy.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import gymnasium as gym

from airs.environment.network_env import NetworkSecurityEnv
from airs.environment.attack_simulator import AttackSimulator


ALL_ATTACK_MODES = list(AttackSimulator.MODES)          # 4
ALL_INTENSITIES = ["low", "medium", "high"]              # 3


class MultiScenarioEnv(gym.Env):
    """Gymnasium wrapper that creates a fresh NetworkSecurityEnv each reset
    with a randomly sampled (attack_mode, intensity) combination.

    The observation and action spaces match NetworkSecurityEnv exactly,
    so models trained on this env can be evaluated on any single scenario.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        attack_modes: Optional[list[str]] = None,
        intensities: Optional[list[str]] = None,
        env_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self._attack_modes = attack_modes or ALL_ATTACK_MODES
        self._intensities = intensities or ALL_INTENSITIES
        self._env_kwargs = dict(env_kwargs or {})

        # Build a reference env to copy space definitions
        ref = NetworkSecurityEnv(
            attack_mode=self._attack_modes[0],
            intensity=self._intensities[0],
            **self._env_kwargs,
        )
        self.observation_space = ref.observation_space
        self.action_space = ref.action_space
        self.TRAFFIC_MAX = ref.TRAFFIC_MAX
        self.LOGINS_MAX = ref.LOGINS_MAX
        self.MAX_STEPS = ref.MAX_STEPS

        self._env: Optional[NetworkSecurityEnv] = None
        self._rng = np.random.default_rng()

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Sample random scenario
        mode = self._rng.choice(self._attack_modes)
        intensity = self._rng.choice(self._intensities)

        # Create fresh env for this episode
        if self._env is not None:
            self._env.close()

        self._env = NetworkSecurityEnv(
            attack_mode=mode,
            intensity=intensity,
            **self._env_kwargs,
        )
        self._current_mode = mode
        self._current_intensity = intensity
        return self._env.reset(seed=seed, options=options)

    def step(self, action):
        return self._env.step(action)

    def render(self):
        pass

    def close(self):
        if self._env is not None:
            self._env.close()
