"""
AIRSAgent – thin wrapper around Stable-Baselines3 DQN for AIRS.

Why DQN over PPO?
-----------------
- Action space is discrete (4 actions): DQN is well-suited.
- Cybersecurity defense is a value-based problem where we want
  to learn Q(s, a) — the long-term value of each defensive action.
- DQN with experience replay stabilises training in this sparse-ish
  reward scenario better than on-policy PPO for small action spaces.
- PPO support is also included for comparison experiments.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor

from airs.environment.network_env import NetworkSecurityEnv


class RewardLoggerCallback(BaseCallback):
    """Logs episode reward statistics during training."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: list[float] = []
        self._current_episode_reward: float = 0.0

    def _on_step(self) -> bool:
        reward = self.locals.get("rewards", [0])[0]
        self._current_episode_reward += float(reward)

        dones = self.locals.get("dones", [False])
        if dones[0]:
            self.episode_rewards.append(self._current_episode_reward)
            self._current_episode_reward = 0.0
        return True


class AIRSAgent:
    """Wraps a Stable-Baselines3 agent for the NetworkSecurityEnv.

    Parameters
    ----------
    algorithm : str
        "dqn" (default) or "ppo".
    attack_mode : str
        Attack mode for the training environment.
    intensity : str
        Attack intensity level.
    model_path : str, optional
        Path to a saved model file to load.
    """

    ALGORITHMS = {"dqn": DQN, "ppo": PPO}

    # DQN hyperparameters (tuned for this environment)
    DQN_KWARGS = {
        "learning_rate": 1e-3,
        "buffer_size": 50_000,
        "learning_starts": 500,
        "batch_size": 64,
        "tau": 1.0,
        "gamma": 0.99,
        "train_freq": 4,
        "gradient_steps": 1,
        "target_update_interval": 500,
        "exploration_fraction": 0.3,
        "exploration_final_eps": 0.05,
        "verbose": 0,
    }

    # PPO hyperparameters
    PPO_KWARGS = {
        "learning_rate": 3e-4,
        "n_steps": 512,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "verbose": 0,
    }

    def __init__(
        self,
        algorithm: str = "dqn",
        attack_mode: str = "brute_force",
        intensity: str = "medium",
        model_path: Optional[str] = None,
    ):
        algorithm = algorithm.lower()
        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"algorithm must be one of {list(self.ALGORITHMS)}")

        self.algorithm = algorithm
        self.attack_mode = attack_mode
        self.intensity = intensity
        self.reward_logger = RewardLoggerCallback()

        env = NetworkSecurityEnv(attack_mode=attack_mode, intensity=intensity)
        self._env = Monitor(env)

        algo_cls = self.ALGORITHMS[algorithm]
        kwargs = self.DQN_KWARGS if algorithm == "dqn" else self.PPO_KWARGS

        if model_path and os.path.exists(model_path):
            self._model = algo_cls.load(model_path, env=self._env)
        else:
            self._model = algo_cls("MlpPolicy", self._env, **kwargs)

    # ------------------------------------------------------------------
    def train(self, total_timesteps: int = 50_000) -> "AIRSAgent":
        """Train the agent for the given number of timesteps.

        Args:
            total_timesteps: Total environment steps to train for.

        Returns:
            self (for chaining)
        """
        self._model.learn(
            total_timesteps=total_timesteps,
            callback=self.reward_logger,
            reset_num_timesteps=False,
        )
        return self

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """Predict the best action for the given observation.

        Args:
            obs:           Observation array of shape (6,).
            deterministic: Use greedy policy (True) or stochastic (False).

        Returns:
            action integer.
        """
        action, _ = self._model.predict(obs, deterministic=deterministic)
        return int(action)

    def save(self, path: str) -> None:
        """Save the model weights to disk."""
        self._model.save(path)

    @property
    def episode_rewards(self) -> list[float]:
        """Cumulative reward per completed training episode."""
        return self.reward_logger.episode_rewards
