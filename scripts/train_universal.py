#!/usr/bin/env python3
"""
train_universal.py – Train DQN and PPO agents across ALL attack modes and intensities.

Trains a single generalist model per algorithm using MultiScenarioEnv,
which randomly samples from all 12 (attack_mode × intensity) combinations
each episode. This produces agents that generalise across the full space.

Usage:
  python scripts/train_universal.py                    # Both algorithms
  python scripts/train_universal.py --algorithm dqn    # DQN only
  python scripts/train_universal.py --algorithm ppo    # PPO only
  python scripts/train_universal.py --timesteps 200000 # More training
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airs.agent.rl_agent import AIRSAgent
from airs.config import load_config
from airs.visualization.visualizer import AIRSVisualizer


def parse_args():
    p = argparse.ArgumentParser(description="Train universal AIRS agents")
    p.add_argument("--algorithm", default=None, choices=["dqn", "ppo"],
                   help="Train single algorithm (default: both)")
    p.add_argument("--timesteps", default=None, type=int,
                   help="Total timesteps per algorithm (default: from config)")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--seed", default=42, type=int)
    return p.parse_args()


def train_one(algorithm: str, timesteps: int, cfg: dict, seed: int):
    """Train one algorithm and save the model."""
    env_cfg = cfg.get("environment", {})
    agent_cfg = cfg.get("agent", {})
    train_cfg = cfg.get("training", {})
    reward_cfg = cfg.get("reward", {})
    paths_cfg = cfg.get("paths", {})

    model_path = os.path.join(paths_cfg.get("model_dir", "models"), f"{algorithm}_agent")
    output_dir = paths_cfg.get("results_dir", "results")
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

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

    print(f"\n{'='*60}")
    print(f" Training {algorithm.upper()} – Universal Agent")
    print(f" Timesteps: {timesteps:,} | Multi-scenario training")
    print(f"{'='*60}\n")

    t0 = time.time()

    agent = AIRSAgent(
        algorithm=algorithm,
        attack_mode="brute_force",  # fallback, multi_scenario overrides
        intensity="medium",
        n_envs=1,
        seed=seed,
        env_kwargs=env_kwargs,
        algo_kwargs=algo_kwargs,
        multi_scenario=True,
    )

    agent.train(
        total_timesteps=timesteps,
        eval_freq=train_cfg.get("eval_freq", 5000),
        eval_episodes=train_cfg.get("eval_episodes", 10),
        checkpoint_best=train_cfg.get("checkpoint_best", True),
        early_stopping_patience=max(train_cfg.get("early_stopping_patience", 5), 15),
        model_save_path=model_path,
    )

    agent.save(model_path)
    elapsed = time.time() - t0

    # Plot training reward curve
    if agent.episode_rewards:
        viz = AIRSVisualizer(output_dir=output_dir)
        viz.plot_reward_curve(
            agent.episode_rewards,
            title=f"Universal {algorithm.upper()} Training Reward",
            filename=f"training_reward_{algorithm}_universal.png",
        )

    n_eps = len(agent.episode_rewards)
    avg20 = sum(agent.episode_rewards[-20:]) / max(len(agent.episode_rewards[-20:]), 1)
    print(f"\n[AIRS] {algorithm.upper()} training complete.")
    print(f"  Model saved: {model_path}")
    print(f"  Episodes: {n_eps}")
    print(f"  Final avg reward (last 20): {avg20:.2f}")
    print(f"  Time: {elapsed:.0f}s")
    return agent


def main():
    args = parse_args()
    cfg = load_config(args.config)
    train_cfg = cfg.get("training", {})
    timesteps = args.timesteps or train_cfg.get("total_timesteps", 100_000)

    algorithms = [args.algorithm] if args.algorithm else ["dqn", "ppo"]

    for algo in algorithms:
        train_one(algo, timesteps, cfg, args.seed)

    print(f"\n{'='*60}")
    print(f" All training complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
