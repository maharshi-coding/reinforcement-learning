#!/usr/bin/env python3
"""
train_universal.py – Train DQN, PPO, and A2C agents across ALL attack scenarios.

Supports:
  - Random multi-scenario training (default)
  - Curriculum training: low → medium → high → mixed
  - Learning curve logging (reward vs timestep)
  - 500k–1M timestep training

Usage:
  python scripts/train_universal.py                         # All 3 algorithms
  python scripts/train_universal.py --algorithm dqn         # DQN only
  python scripts/train_universal.py --algorithm a2c         # A2C only
  python scripts/train_universal.py --timesteps 500000      # 500k steps
  python scripts/train_universal.py --curriculum            # Curriculum training
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from airs.agent.rl_agent import AIRSAgent
from airs.config import load_config
from airs.environment.network_env import NetworkSecurityEnv
from airs.environment.multi_scenario_env import MultiScenarioEnv


# ── Learning Curve Callback ──────────────────────────────────────────

class LearningCurveCallback(BaseCallback):
    """Records (timestep, episode_reward) pairs for learning curve plots."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.timesteps: list[int] = []
        self.rewards: list[float] = []
        self._current_reward = 0.0

    def _on_step(self) -> bool:
        reward = self.locals.get("rewards", [0])[0]
        self._current_reward += float(reward)
        dones = self.locals.get("dones", [False])
        if dones[0]:
            self.timesteps.append(self.num_timesteps)
            self.rewards.append(self._current_reward)
            self._current_reward = 0.0
        return True


# ── Curriculum Environment Builder ───────────────────────────────────

def make_curriculum_env(intensities, env_kwargs, seed=42):
    """Create a MultiScenarioEnv restricted to specific intensities."""
    def _init():
        env = MultiScenarioEnv(intensities=intensities, env_kwargs=env_kwargs)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def make_multi_env(env_kwargs, seed=42):
    """Create a standard MultiScenarioEnv (all scenarios)."""
    def _init():
        env = MultiScenarioEnv(env_kwargs=env_kwargs)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


# ── Training Functions ───────────────────────────────────────────────

def train_curriculum(algorithm, timesteps, cfg, seed):
    """Train with curriculum: low → medium → high → mixed."""
    env_cfg = cfg.get("environment", {})
    reward_cfg = cfg.get("reward", {})
    paths_cfg = cfg.get("paths", {})

    model_path = os.path.join(paths_cfg.get("model_dir", "models"), f"{algorithm}_agent")
    output_dir = paths_cfg.get("results_dir", "results")
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    env_kwargs = {
        "noisy_observations": env_cfg.get("noisy_observations", False),
        "noise_std": env_cfg.get("noise_std", 0.05),
        "temporal_window": env_cfg.get("temporal_window", 1),
        "reward_cfg": reward_cfg,
    }

    # Curriculum stages: 25% low, 25% medium, 25% high, 25% mixed
    stage_steps = timesteps // 4
    stages = [
        {"name": "Stage 1: LOW intensity", "intensities": ["low"], "timesteps": stage_steps},
        {"name": "Stage 2: MEDIUM intensity", "intensities": ["medium"], "timesteps": stage_steps},
        {"name": "Stage 3: HIGH intensity", "intensities": ["high"], "timesteps": stage_steps},
        {"name": "Stage 4: MIXED (all)", "intensities": ["low", "medium", "high"], "timesteps": stage_steps},
    ]

    print(f"\n{'='*60}")
    print(f" Curriculum Training: {algorithm.upper()}")
    print(f" Total timesteps: {timesteps:,} across {len(stages)} stages")
    print(f"{'='*60}\n")

    t0 = time.time()
    lc_callback = LearningCurveCallback()

    # Create initial env
    env = DummyVecEnv([make_curriculum_env(stages[0]["intensities"], env_kwargs, seed)])

    # Build model
    agent = AIRSAgent(
        algorithm=algorithm,
        multi_scenario=True,
        seed=seed,
        env_kwargs=env_kwargs,
    )
    model = agent._model
    model.set_env(env)

    for i, stage in enumerate(stages):
        print(f"\n── {stage['name']} ({stage['timesteps']:,} steps) ──")

        # Rebuild env for this stage's intensity subset
        new_env = DummyVecEnv([make_curriculum_env(stage["intensities"], env_kwargs, seed)])
        model.set_env(new_env)

        eval_env = DummyVecEnv([make_multi_env(env_kwargs, seed + 1000)])
        stop_cb = StopTrainingOnNoModelImprovement(max_no_improvement_evals=20, verbose=0)
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=os.path.dirname(model_path) or "models",
            eval_freq=5000,
            n_eval_episodes=10,
            callback_after_eval=stop_cb,
            verbose=0,
        )

        model.learn(
            total_timesteps=stage["timesteps"],
            callback=CallbackList([lc_callback, eval_cb]),
            reset_num_timesteps=False,
        )

        # Report stage progress
        recent = lc_callback.rewards[-20:] if lc_callback.rewards else [0]
        print(f"  Avg reward (last 20 eps): {np.mean(recent):.1f}")

    model.save(model_path)
    elapsed = time.time() - t0

    print(f"\n[AIRS] {algorithm.upper()} curriculum training complete.")
    print(f"  Model saved: {model_path}")
    print(f"  Episodes: {len(lc_callback.rewards)}")
    print(f"  Time: {elapsed:.0f}s")

    return lc_callback


def train_standard(algorithm, timesteps, cfg, seed):
    """Train with random multi-scenario selection (original method)."""
    env_cfg = cfg.get("environment", {})
    reward_cfg = cfg.get("reward", {})
    paths_cfg = cfg.get("paths", {})

    model_path = os.path.join(paths_cfg.get("model_dir", "models"), f"{algorithm}_agent")
    output_dir = paths_cfg.get("results_dir", "results")
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    env_kwargs = {
        "noisy_observations": env_cfg.get("noisy_observations", False),
        "noise_std": env_cfg.get("noise_std", 0.05),
        "temporal_window": env_cfg.get("temporal_window", 1),
        "reward_cfg": reward_cfg,
    }

    algo_kwargs = dict(cfg.get("agent", {}).get(algorithm, {}))

    print(f"\n{'='*60}")
    print(f" Training {algorithm.upper()} – Universal Agent (Multi-Scenario)")
    print(f" Timesteps: {timesteps:,}")
    print(f"{'='*60}\n")

    t0 = time.time()
    lc_callback = LearningCurveCallback()

    agent = AIRSAgent(
        algorithm=algorithm,
        multi_scenario=True,
        seed=seed,
        env_kwargs=env_kwargs,
        algo_kwargs=algo_kwargs,
    )

    # Inject our learning curve callback into training
    model = agent._model
    env = DummyVecEnv([make_multi_env(env_kwargs, seed)])
    model.set_env(env)

    eval_env = DummyVecEnv([make_multi_env(env_kwargs, seed + 1000)])
    stop_cb = StopTrainingOnNoModelImprovement(max_no_improvement_evals=20, verbose=1)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.dirname(model_path) or "models",
        eval_freq=5000,
        n_eval_episodes=10,
        callback_after_eval=stop_cb,
        verbose=0,
    )

    model.learn(
        total_timesteps=timesteps,
        callback=CallbackList([lc_callback, eval_cb]),
    )

    model.save(model_path)
    elapsed = time.time() - t0

    n_eps = len(lc_callback.rewards)
    avg20 = np.mean(lc_callback.rewards[-20:]) if lc_callback.rewards else 0
    print(f"\n[AIRS] {algorithm.upper()} training complete.")
    print(f"  Model saved: {model_path}")
    print(f"  Episodes: {n_eps}")
    print(f"  Final avg reward (last 20): {avg20:.2f}")
    print(f"  Time: {elapsed:.0f}s")

    return lc_callback


# ── Learning Curve Plots ─────────────────────────────────────────────

def plot_learning_curves(all_curves, output_dir):
    """Plot reward vs timestep for all algorithms on one chart."""
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    colours = {"dqn": "#4C9AFF", "ppo": "#F97316", "a2c": "#10B981"}

    for algo, lc in all_curves.items():
        if not lc.timesteps:
            continue
        ts = np.array(lc.timesteps)
        rw = np.array(lc.rewards)

        # Smoothed with rolling window
        window = min(50, len(rw) // 4) if len(rw) > 10 else 1
        if window > 1:
            smoothed = np.convolve(rw, np.ones(window) / window, mode="valid")
            ts_smooth = ts[window - 1:]
        else:
            smoothed = rw
            ts_smooth = ts

        col = colours.get(algo, "grey")
        ax.plot(ts_smooth, smoothed, label=f"{algo.upper()} (smoothed)", color=col, linewidth=2)
        ax.fill_between(ts_smooth, smoothed * 0.9, smoothed * 1.1, alpha=0.1, color=col)

    ax.set_xlabel("Timesteps", fontsize=12)
    ax.set_ylabel("Episode Reward", fontsize=12)
    ax.set_title("Learning Curves: DQN vs PPO vs A2C", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    path = os.path.join(output_dir, "learning_curves.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved learning curves: {path}")

    # Also save per-algorithm curves
    for algo, lc in all_curves.items():
        if not lc.timesteps:
            continue

        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ts = np.array(lc.timesteps)
        rw = np.array(lc.rewards)
        col = colours.get(algo, "grey")

        ax2.plot(ts, rw, alpha=0.3, color=col, linewidth=0.5)
        window = min(50, len(rw) // 4) if len(rw) > 10 else 1
        if window > 1:
            smoothed = np.convolve(rw, np.ones(window) / window, mode="valid")
            ax2.plot(ts[window - 1:], smoothed, color=col, linewidth=2, label=f"Smoothed (w={window})")

        ax2.set_xlabel("Timesteps")
        ax2.set_ylabel("Episode Reward")
        ax2.set_title(f"{algo.upper()} – Learning Curve")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ppath = os.path.join(output_dir, f"learning_curve_{algo}.png")
        fig2.tight_layout()
        fig2.savefig(ppath, dpi=150)
        plt.close(fig2)
        print(f"  Saved: {ppath}")

    # Save raw data as JSON for reproducibility
    for algo, lc in all_curves.items():
        data_path = os.path.join(output_dir, f"learning_curve_{algo}.json")
        with open(data_path, "w") as f:
            json.dump({"timesteps": lc.timesteps, "rewards": lc.rewards}, f)


# ── Main ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train universal AIRS agents")
    p.add_argument("--algorithm", default=None, choices=["dqn", "ppo", "a2c"],
                   help="Train single algorithm (default: all three)")
    p.add_argument("--timesteps", default=None, type=int,
                   help="Total timesteps per algorithm (default: from config)")
    p.add_argument("--curriculum", action="store_true",
                   help="Use curriculum training (low→medium→high→mixed)")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--seed", default=42, type=int)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    train_cfg = cfg.get("training", {})
    paths_cfg = cfg.get("paths", {})
    timesteps = args.timesteps or train_cfg.get("total_timesteps", 100_000)
    output_dir = paths_cfg.get("results_dir", "results")

    algorithms = [args.algorithm] if args.algorithm else ["dqn", "ppo", "a2c"]
    train_fn = train_curriculum if args.curriculum else train_standard

    all_curves = {}
    for algo in algorithms:
        lc = train_fn(algo, timesteps, cfg, args.seed)
        all_curves[algo] = lc

    # Generate learning curve plots
    print(f"\nGenerating learning curves...")
    plot_learning_curves(all_curves, output_dir)

    print(f"\n{'='*60}")
    print(f" All training complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
