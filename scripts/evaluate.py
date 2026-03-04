"""
evaluate.py – Evaluate a trained AIRS agent over multiple episodes.

Usage
-----
python scripts/evaluate.py [--algorithm dqn|ppo] [--attack_mode brute_force|flood|adaptive]
                           [--intensity low|medium|high] [--episodes 50]
                           [--model_path models/airs_agent] [--output_dir results]

Example
-------
python scripts/evaluate.py --algorithm dqn --attack_mode adaptive --episodes 20
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airs.agent.rl_agent import AIRSAgent
from airs.environment.network_env import NetworkSecurityEnv
from airs.visualization.visualizer import AIRSVisualizer


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate AIRS RL agent")
    p.add_argument("--algorithm",   default="dqn", choices=["dqn", "ppo"])
    p.add_argument("--attack_mode", default="brute_force",
                   choices=["brute_force", "flood", "adaptive"])
    p.add_argument("--intensity",   default="medium",
                   choices=["low", "medium", "high"])
    p.add_argument("--episodes",    default=50, type=int)
    p.add_argument("--model_path",  default="models/airs_agent")
    p.add_argument("--output_dir",  default="results")
    return p.parse_args()


def evaluate_agent(agent: AIRSAgent, attack_mode: str, intensity: str, n_episodes: int):
    """Run the agent in a fresh environment and collect metrics."""
    env = NetworkSecurityEnv(attack_mode=attack_mode, intensity=intensity)

    episode_rewards = []
    attack_success_rates = []
    action_counts = defaultdict(int)
    all_threat_levels = []
    all_actions = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        high_threat_steps = 0
        acted_on_high_threat = 0
        ep_threats = []
        ep_actions = []

        done = False
        while not done:
            action = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            ep_reward += reward
            action_counts[info["action_name"]] += 1
            ep_threats.append(info["threat_level"])
            ep_actions.append(action)

            if info["threat_level"] > NetworkSecurityEnv.HIGH_THREAT_THRESHOLD:
                high_threat_steps += 1
                if action > 0:  # any defensive action
                    acted_on_high_threat += 1

        episode_rewards.append(ep_reward)
        # Attack success = fraction of high-threat steps where no defense was taken
        if high_threat_steps > 0:
            attack_success = 1.0 - (acted_on_high_threat / high_threat_steps)
        else:
            attack_success = 0.0
        attack_success_rates.append(attack_success)
        all_threat_levels.extend(ep_threats)
        all_actions.extend(ep_actions)

    return {
        "episode_rewards": episode_rewards,
        "attack_success_rates": attack_success_rates,
        "action_counts": dict(action_counts),
        "all_threat_levels": all_threat_levels,
        "all_actions": all_actions,
    }


def main():
    args = parse_args()

    print(
        f"[AIRS] Evaluating {args.algorithm.upper()} | "
        f"attack={args.attack_mode} | intensity={args.intensity} | "
        f"episodes={args.episodes}"
    )

    agent = AIRSAgent(
        algorithm=args.algorithm,
        attack_mode=args.attack_mode,
        intensity=args.intensity,
        model_path=args.model_path if os.path.exists(args.model_path + ".zip") else None,
    )

    metrics = evaluate_agent(agent, args.attack_mode, args.intensity, args.episodes)

    # Print summary
    rewards = metrics["episode_rewards"]
    asr = metrics["attack_success_rates"]
    print(f"\n{'='*55}")
    print(f" EVALUATION RESULTS ({args.attack_mode}, {args.intensity})")
    print(f"{'='*55}")
    print(f"  Episodes:              {args.episodes}")
    print(f"  Mean episode reward:   {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Min / Max reward:      {np.min(rewards):.2f} / {np.max(rewards):.2f}")
    print(f"  Mean attack success:   {np.mean(asr)*100:.1f}%")
    print(f"  Action distribution:   {metrics['action_counts']}")
    print(f"{'='*55}\n")

    # Save plots
    viz = AIRSVisualizer(output_dir=args.output_dir)
    prefix = f"eval_{args.algorithm}_{args.attack_mode}"

    viz.plot_reward_curve(
        rewards,
        title=f"Evaluation Reward ({args.algorithm.upper()}, {args.attack_mode})",
        filename=f"{prefix}_reward.png",
    )
    viz.plot_action_distribution(
        metrics["action_counts"],
        title=f"Action Distribution ({args.attack_mode})",
        filename=f"{prefix}_actions.png",
    )
    viz.plot_attack_success_rate(
        asr,
        title=f"Attack Success Rate ({args.attack_mode}, {args.intensity})",
        filename=f"{prefix}_attack_success.png",
    )
    # Plot the first episode's threat timeline
    ep_len = NetworkSecurityEnv.MAX_STEPS
    viz.plot_threat_timeline(
        metrics["all_threat_levels"][:ep_len],
        metrics["all_actions"][:ep_len],
        title=f"Threat Timeline – Episode 1 ({args.attack_mode})",
        filename=f"{prefix}_threat_timeline.png",
    )

    print(f"[AIRS] Plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
