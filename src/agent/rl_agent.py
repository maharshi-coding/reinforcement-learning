"""
AIRSAgent – wrapper around Stable-Baselines3 DQN/PPO.

Supports:
    - DQN and PPO algorithms
    - Vectorised parallel environments (DummyVecEnv / SubprocVecEnv)
    - Curriculum learning with progressive difficulty stages
    - Best-model checkpointing and early stopping
    - Multi-seed reproducible training
    - Model save / load (handles .zip suffix transparently)
"""

from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from src.environment.intrusion_env import IntrusionEnv


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
    """Factory returning a callable for VecEnv creation."""
    def _init():
        kw = dict(env_kwargs or {})
        env = IntrusionEnv(attack_mode=attack_mode, intensity=intensity, **kw)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


class AIRSAgent:
    """Wraps a Stable-Baselines3 agent for the IntrusionEnv.

    Parameters
    ----------
    algorithm : str
        ``"dqn"`` or ``"ppo"``.
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
        Extra keyword arguments forwarded to IntrusionEnv.
    algo_kwargs : dict, optional
        Override default hyperparameters for the chosen algorithm.
    """

    ALGORITHMS = {"dqn": DQN, "ppo": PPO}

    DQN_KWARGS: dict[str, Any] = {
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

    PPO_KWARGS: dict[str, Any] = {
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
        n_envs: int = 1,
        seed: int = 42,
        env_kwargs: Optional[dict] = None,
        algo_kwargs: Optional[dict] = None,
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
        self.reward_logger = RewardLoggerCallback()

        self._env = self._build_vec_env(attack_mode, intensity, n_envs, seed)

        algo_cls = self.ALGORITHMS[algorithm]
        kwargs = dict(self.DQN_KWARGS if algorithm == "dqn" else self.PPO_KWARGS)
        if algo_kwargs:
            kwargs.update(algo_kwargs)
        kwargs["seed"] = seed

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
        env_fns = [
            _make_env(attack_mode, intensity, seed=seed + i, env_kwargs=self.env_kwargs)
            for i in range(n_envs)
        ]
        if n_envs == 1:
            return DummyVecEnv(env_fns)
        return DummyVecEnv(env_fns)

    def train(
        self,
        total_timesteps: int = 50_000,
        eval_freq: int = 5000,
        eval_episodes: int = 10,
        checkpoint_best: bool = True,
        early_stopping_patience: int = 5,
        model_save_path: str = "models/airs_agent",
    ) -> "AIRSAgent":
        """Train the agent with optional checkpointing and early stopping."""
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
        """Train with curriculum learning — progressively harder stages."""
        for i, stage in enumerate(stages):
            intensity = stage.get("intensity", self.intensity)
            timesteps = stage.get("timesteps", 10_000)
            print(
                f"[AIRS] Curriculum stage {i+1}/{len(stages)}: "
                f"intensity={intensity}, timesteps={timesteps:,}"
            )
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
        action, _ = self._model.predict(obs, deterministic=deterministic)
        return int(action)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._model.save(path)

    @property
    def episode_rewards(self) -> list[float]:
        return self.reward_logger.episode_rewards
