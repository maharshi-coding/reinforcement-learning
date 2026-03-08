"""
train_agent.py – Train an AIRS RL agent with MLflow experiment tracking.

Usage
-----
python src/training/train_agent.py --config configs/training_config.yaml
python src/training/train_agent.py --algorithm ppo --curriculum
python src/training/train_agent.py --algorithm dqn --attack_mode adaptive --timesteps 100000
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# Ensure project root is on path
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.agent.rl_agent import AIRSAgent
from src.visualization.visualizer import AIRSVisualizer

# Optional MLflow
try:
    import mlflow

    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False


# ────────────────────────────────────────────────────────────────────
# Config loader
# ────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ────────────────────────────────────────────────────────────────────
# Reproducibility
# ────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    """Set seeds for reproducibility across numpy, torch, random."""
    import random as stdlib_random

    stdlib_random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# ────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train AIRS RL agent")
    p.add_argument("--algorithm", default=None, choices=["dqn", "ppo"])
    p.add_argument(
        "--attack_mode",
        default=None,
        choices=["brute_force", "flood", "adaptive", "multi_stage"],
    )
    p.add_argument("--intensity", default=None, choices=["low", "medium", "high"])
    p.add_argument("--timesteps", default=None, type=int)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--model_path", default=None)
    p.add_argument("--n_envs", default=None, type=int)
    p.add_argument("--seed", default=None, type=int)
    p.add_argument(
        "--config",
        default="configs/training_config.yaml",
        help="Path to YAML config file",
    )
    p.add_argument(
        "--curriculum", action="store_true", help="Enable curriculum learning"
    )
    p.add_argument(
        "--no-mlflow", action="store_true", help="Disable MLflow tracking"
    )
    return p.parse_args()


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Load YAML config
    cfg = load_config(args.config)
    env_cfg = cfg.get("environment", {})
    agent_cfg = cfg.get("agent", {})
    train_cfg = cfg.get("training", {})
    reward_cfg = cfg.get("reward", {})
    paths_cfg = cfg.get("paths", {})

    # CLI overrides
    algorithm = args.algorithm or agent_cfg.get("algorithm", "dqn")
    attack_mode = args.attack_mode or env_cfg.get("attack_mode", "brute_force")
    intensity = args.intensity or env_cfg.get("intensity", "medium")
    timesteps = args.timesteps or train_cfg.get("total_timesteps", 50_000)
    output_dir = args.output_dir or paths_cfg.get("results_dir", "results")
    model_path = args.model_path or os.path.join(
        paths_cfg.get("model_dir", "models"), f"{algorithm}_agent"
    )
    n_envs = args.n_envs or train_cfg.get("n_envs", 1)
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    use_curriculum = args.curriculum or train_cfg.get("curriculum", {}).get(
        "enabled", False
    )
    use_mlflow = _HAS_MLFLOW and not args.no_mlflow

    # Reproducibility
    set_seed(seed)

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Environment kwargs from config
    env_kwargs = {
        "noisy_observations": env_cfg.get("noisy_observations", False),
        "noise_std": env_cfg.get("noise_std", 0.05),
        "partial_observability": env_cfg.get("partial_observability", False),
        "mask_probability": env_cfg.get("mask_probability", 0.1),
        "action_cooldown": env_cfg.get("action_cooldown", 0),
        "delayed_effect_steps": env_cfg.get("delayed_effect_steps", 0),
        "resource_budget": env_cfg.get("resource_budget", None),
        "temporal_window": env_cfg.get("temporal_window", 1),
        "reward_cfg": reward_cfg,
    }

    algo_kwargs = dict(agent_cfg.get(algorithm, {}))

    print(
        f"[AIRS] Training {algorithm.upper()} | "
        f"attack={attack_mode} | intensity={intensity} | "
        f"timesteps={timesteps:,} | n_envs={n_envs} | seed={seed}"
    )

    # ── MLflow tracking ──
    if use_mlflow:
        tracking_uri = paths_cfg.get("mlflow_uri", "experiments/training_runs")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("AIRS_Training")
        mlflow.start_run(run_name=f"{algorithm}_{attack_mode}_{intensity}_s{seed}")
        mlflow.log_params(
            {
                "algorithm": algorithm,
                "attack_mode": attack_mode,
                "intensity": intensity,
                "total_timesteps": timesteps,
                "n_envs": n_envs,
                "seed": seed,
                "curriculum": use_curriculum,
                **{f"env.{k}": str(v) for k, v in env_kwargs.items() if k != "reward_cfg"},
                **{f"reward.{k}": v for k, v in reward_cfg.items()},
                **{f"algo.{k}": v for k, v in algo_kwargs.items()},
            }
        )

    # ── Build agent ──
    agent = AIRSAgent(
        algorithm=algorithm,
        attack_mode=attack_mode,
        intensity=intensity,
        n_envs=n_envs,
        seed=seed,
        env_kwargs=env_kwargs,
        algo_kwargs=algo_kwargs,
    )

    train_kwargs = dict(
        eval_freq=train_cfg.get("eval_freq", 5000),
        eval_episodes=train_cfg.get("eval_episodes", 10),
        checkpoint_best=train_cfg.get("checkpoint_best", True),
        early_stopping_patience=train_cfg.get("early_stopping_patience", 5),
        model_save_path=model_path,
    )

    # ── Train ──
    t0 = time.time()
    if use_curriculum:
        stages = train_cfg.get("curriculum", {}).get(
            "stages",
            [
                {"intensity": "low", "timesteps": 15000},
                {"intensity": "medium", "timesteps": 20000},
                {"intensity": "high", "timesteps": 15000},
            ],
        )
        agent.train_curriculum(stages, **train_kwargs)
    else:
        agent.train(total_timesteps=timesteps, **train_kwargs)
    elapsed = time.time() - t0

    agent.save(model_path)
    print(f"[AIRS] Model saved → {model_path}")

    # ── Log results ──
    n_episodes = len(agent.episode_rewards)
    final_avg = (
        sum(agent.episode_rewards[-20:]) / max(len(agent.episode_rewards[-20:]), 1)
        if n_episodes > 0
        else 0.0
    )

    if use_mlflow:
        mlflow.log_metrics(
            {
                "total_episodes": n_episodes,
                "final_avg_reward_20": final_avg,
                "training_time_seconds": elapsed,
            }
        )
        # Log reward curve as individual steps
        for i, r in enumerate(agent.episode_rewards):
            mlflow.log_metric("episode_reward", r, step=i)
        mlflow.log_artifact(model_path + ".zip" if os.path.exists(model_path + ".zip") else model_path)

    # ── Visualisation ──
    if agent.episode_rewards:
        viz = AIRSVisualizer(output_dir=output_dir)
        path = viz.plot_reward_curve(
            agent.episode_rewards,
            title=f"Training Reward ({algorithm.upper()}, {attack_mode})",
            filename=f"training_reward_{algorithm}_{attack_mode}.png",
        )
        print(f"[AIRS] Reward curve saved → {path}")
        if use_mlflow:
            mlflow.log_artifact(path)

    if use_mlflow:
        mlflow.end_run()

    print(
        f"[AIRS] Training complete in {elapsed:.1f}s. "
        f"Episodes: {n_episodes}, "
        f"Final avg reward (last 20): {final_avg:.2f}"
    )


if __name__ == "__main__":
    main()
