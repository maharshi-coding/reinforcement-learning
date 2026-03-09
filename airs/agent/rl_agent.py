"""
AIRSAgent – wrapper around Stable-Baselines3 DQN/PPO for AIRS.

Supports:
- DQN and PPO algorithms
- Vectorised parallel environments (SubprocVecEnv / DummyVecEnv)
- Curriculum learning with progressive difficulty stages
- Best-model checkpointing and early stopping
- Multi-seed reproducible training
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from airs.environment.network_env import NetworkSecurityEnv
from airs.environment.multi_scenario_env import MultiScenarioEnv


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


def _make_env(
    attack_mode: str,
    intensity: str,
    seed: int = 0,
    env_kwargs: Optional[dict] = None,
):
    """Factory that returns a callable for SubprocVecEnv / DummyVecEnv."""
    def _init():
        kw = dict(env_kwargs or {})
        env = NetworkSecurityEnv(attack_mode=attack_mode, intensity=intensity, **kw)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def _make_multi_env(
    seed: int = 0,
    env_kwargs: Optional[dict] = None,
    attack_modes: Optional[list[str]] = None,
    intensities: Optional[list[str]] = None,
):
    """Factory that returns a callable for a MultiScenarioEnv."""
    def _init():
        kw = dict(env_kwargs or {})
        env = MultiScenarioEnv(
            attack_modes=attack_modes,
            intensities=intensities,
            env_kwargs=kw,
        )
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


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
    n_envs : int
        Number of parallel environments.
    seed : int
        Random seed for reproducibility.
    env_kwargs : dict, optional
        Extra keyword arguments forwarded to NetworkSecurityEnv.
    algo_kwargs : dict, optional
        Override default hyperparameters for the chosen algorithm.
    """

    ALGORITHMS = {"dqn": DQN, "ppo": PPO, "a2c": A2C}

    # DQN hyperparameters (tuned for multi-scenario generalization)
    DQN_KWARGS: dict[str, Any] = {
        "learning_rate": 5e-4,
        "buffer_size": 100_000,
        "learning_starts": 1000,
        "batch_size": 128,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": 4,
        "gradient_steps": 2,
        "target_update_interval": 250,
        "exploration_fraction": 0.4,
        "exploration_final_eps": 0.03,
        "policy_kwargs": {"net_arch": [256, 256]},
        "verbose": 0,
    }

    # PPO hyperparameters (tuned for stability)
    PPO_KWARGS: dict[str, Any] = {
        "learning_rate": 2.5e-4,
        "n_steps": 1024,
        "batch_size": 128,
        "n_epochs": 15,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "policy_kwargs": {"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
        "device": "cpu",
        "verbose": 0,
    }

    # A2C hyperparameters (tuned for fast convergence)
    A2C_KWARGS: dict[str, Any] = {
        "learning_rate": 7e-4,
        "n_steps": 256,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "normalize_advantage": True,
        "policy_kwargs": {"net_arch": {"pi": [256, 256], "vf": [256, 256]}},
        "device": "cpu",
        "verbose": 0,
    }

    def __init__(
        self,
        algorithm: str = "dqn",
        attack_mode: str = "brute_force",
        intensity: str = "medium",
        model_path: Optional[str] = None,
        n_envs: int = 1,
        seed: int = 42,
        env_kwargs: Optional[dict] = None,
        algo_kwargs: Optional[dict] = None,
        multi_scenario: bool = False,
    ):
        algorithm = algorithm.lower()
        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"algorithm must be one of {list(self.ALGORITHMS)}")

        self.algorithm = algorithm
        self.attack_mode = attack_mode
        self.intensity = intensity
        self.n_envs = n_envs
        self.seed = seed
        self.env_kwargs = env_kwargs or {}
        self.multi_scenario = multi_scenario
        self.reward_logger = RewardLoggerCallback()

        # Build vectorised environment
        if multi_scenario:
            self._env = self._build_multi_vec_env(n_envs, seed)
        else:
            self._env = self._build_vec_env(attack_mode, intensity, n_envs, seed)

        algo_cls = self.ALGORITHMS[algorithm]
        _default_kwargs = {"dqn": self.DQN_KWARGS, "ppo": self.PPO_KWARGS, "a2c": self.A2C_KWARGS}
        kwargs = dict(_default_kwargs[algorithm])
        if algo_kwargs:
            kwargs.update(algo_kwargs)
        kwargs["seed"] = seed

        # Handle model loading — try with and without .zip suffix
        loaded = False
        if model_path:
            for p in [model_path, model_path + ".zip"]:
                if os.path.exists(p):
                    self._model = algo_cls.load(p, env=self._env)
                    loaded = True
                    break

        if not loaded:
            self._model = algo_cls("MlpPolicy", self._env, **kwargs)

    def _build_vec_env(
        self,
        attack_mode: str,
        intensity: str,
        n_envs: int,
        seed: int,
    ):
        """Create a (possibly parallel) vectorised environment."""
        env_fns = [
            _make_env(attack_mode, intensity, seed=seed + i, env_kwargs=self.env_kwargs)
            for i in range(n_envs)
        ]
        if n_envs == 1:
            return DummyVecEnv(env_fns)
        return DummyVecEnv(env_fns)

    def _build_multi_vec_env(self, n_envs: int, seed: int):
        """Create vectorised multi-scenario environments."""
        env_fns = [
            _make_multi_env(seed=seed + i, env_kwargs=self.env_kwargs)
            for i in range(n_envs)
        ]
        return DummyVecEnv(env_fns)

    # ------------------------------------------------------------------
    def train(
        self,
        total_timesteps: int = 50_000,
        eval_freq: int = 5000,
        eval_episodes: int = 10,
        checkpoint_best: bool = True,
        early_stopping_patience: int = 5,
        model_save_path: str = "models/airs_agent",
    ) -> "AIRSAgent":
        """Train the agent with optional checkpointing and early stopping.

        Args:
            total_timesteps: Total environment steps to train for.
            eval_freq: Evaluate every N timesteps.
            eval_episodes: Episodes per evaluation.
            checkpoint_best: Save the model with best eval reward.
            early_stopping_patience: Stop after N evals w/o improvement.
            model_save_path: Directory prefix for model checkpoints.

        Returns:
            self (for chaining)
        """
        callbacks = [self.reward_logger]

        if checkpoint_best:
            eval_env = DummyVecEnv([
                _make_env(
                    self.attack_mode, self.intensity,
                    seed=self.seed + 1000,
                    env_kwargs=self.env_kwargs,
                )
            ])
            stop_cb = StopTrainingOnNoModelImprovement(
                max_no_improvement_evals=early_stopping_patience,
                verbose=1,
            )
            eval_cb = EvalCallback(
                eval_env,
                best_model_save_path=os.path.dirname(model_save_path) or "models",
                eval_freq=max(eval_freq // self.n_envs, 1),
                n_eval_episodes=eval_episodes,
                callback_after_eval=stop_cb,
                verbose=0,
            )
            callbacks.append(eval_cb)

        self._model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList(callbacks),
            reset_num_timesteps=False,
        )
        return self

    def train_curriculum(
        self,
        stages: list[dict],
        model_save_path: str = "models/airs_agent",
        **train_kwargs,
    ) -> "AIRSAgent":
        """Train with curriculum learning — progressively harder stages.

        Args:
            stages: list of dicts, each with keys ``intensity`` and ``timesteps``.
            model_save_path: Where to save checkpoints.
            **train_kwargs: Forwarded to ``train()``.

        Returns:
            self
        """
        for i, stage in enumerate(stages):
            intensity = stage.get("intensity", self.intensity)
            timesteps = stage.get("timesteps", 10_000)
            print(f"[AIRS] Curriculum stage {i+1}/{len(stages)}: "
                  f"intensity={intensity}, timesteps={timesteps:,}")

            # Rebuild env with new intensity
            self._env = self._build_vec_env(
                self.attack_mode, intensity, self.n_envs, self.seed,
            )
            self._model.set_env(self._env)
            self.intensity = intensity

            self.train(
                total_timesteps=timesteps,
                model_save_path=model_save_path,
                **train_kwargs,
            )
        return self

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """Predict the best action for the given observation."""
        action, _ = self._model.predict(obs, deterministic=deterministic)
        return int(action)

    def save(self, path: str) -> None:
        """Save the model weights to disk."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._model.save(path)

    @property
    def episode_rewards(self) -> list[float]:
        """Cumulative reward per completed training episode."""
        return self.reward_logger.episode_rewards
