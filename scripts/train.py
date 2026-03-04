"""
train.py – Train an AIRS RL agent.

Usage
-----
python scripts/train.py [--algorithm dqn|ppo] [--attack_mode brute_force|flood|adaptive]
                        [--intensity low|medium|high] [--timesteps 50000]
                        [--output_dir results] [--model_path models/airs_agent]

Example
-------
python scripts/train.py --algorithm dqn --attack_mode adaptive --timesteps 50000
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airs.agent.rl_agent import AIRSAgent
from airs.visualization.visualizer import AIRSVisualizer


def parse_args():
    p = argparse.ArgumentParser(description="Train AIRS RL agent")
    p.add_argument("--algorithm",   default="dqn", choices=["dqn", "ppo"])
    p.add_argument("--attack_mode", default="brute_force",
                   choices=["brute_force", "flood", "adaptive"])
    p.add_argument("--intensity",   default="medium",
                   choices=["low", "medium", "high"])
    p.add_argument("--timesteps",   default=50_000, type=int)
    p.add_argument("--output_dir",  default="results")
    p.add_argument("--model_path",  default="models/airs_agent")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.model_path) or ".", exist_ok=True)

    print(
        f"[AIRS] Training {args.algorithm.upper()} | "
        f"attack={args.attack_mode} | intensity={args.intensity} | "
        f"timesteps={args.timesteps:,}"
    )

    agent = AIRSAgent(
        algorithm=args.algorithm,
        attack_mode=args.attack_mode,
        intensity=args.intensity,
    )
    agent.train(total_timesteps=args.timesteps)

    agent.save(args.model_path)
    print(f"[AIRS] Model saved → {args.model_path}")

    # Plot training reward curve
    if agent.episode_rewards:
        viz = AIRSVisualizer(output_dir=args.output_dir)
        path = viz.plot_reward_curve(
            agent.episode_rewards,
            title=f"Training Reward ({args.algorithm.upper()}, {args.attack_mode})",
            filename=f"training_reward_{args.algorithm}_{args.attack_mode}.png",
        )
        print(f"[AIRS] Reward curve saved → {path}")

    print(
        f"[AIRS] Training complete. "
        f"Episodes completed: {len(agent.episode_rewards)}, "
        f"Final avg reward (last 20): "
        f"{sum(agent.episode_rewards[-20:]) / max(len(agent.episode_rewards[-20:]), 1):.2f}"
    )


if __name__ == "__main__":
    main()
