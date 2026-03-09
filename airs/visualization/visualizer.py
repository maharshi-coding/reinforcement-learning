"""
AIRSVisualizer – plotting utilities for training and evaluation results.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt


class AIRSVisualizer:
    """Generate and save AIRS evaluation plots."""

    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def plot_reward_curve(
        self,
        rewards: List[float],
        title: str = "Training Reward Curve",
        filename: str = "reward_curve.png",
        window: int = 20,
    ) -> str:
        """Plot cumulative episode reward with a moving-average overlay."""
        fig, ax = plt.subplots(figsize=(10, 4))
        episodes = np.arange(1, len(rewards) + 1)
        ax.plot(episodes, rewards, alpha=0.4, color="steelblue", label="Episode reward")

        if len(rewards) >= window:
            ma = np.convolve(rewards, np.ones(window) / window, mode="valid")
            ax.plot(
                np.arange(window, len(rewards) + 1),
                ma,
                color="navy",
                linewidth=2,
                label=f"{window}-ep moving avg",
            )

        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative Reward")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    def plot_action_distribution(
        self,
        action_counts: dict,
        title: str = "Action Distribution",
        filename: str = "action_distribution.png",
    ) -> str:
        """Bar chart of how often each action was taken."""
        names = list(action_counts.keys())
        counts = list(action_counts.values())
        colors = ["#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.bar(names, counts, color=colors[: len(names)])
        ax.bar_label(bars, padding=3)
        ax.set_xlabel("Action")
        ax.set_ylabel("Count")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    def plot_threat_timeline(
        self,
        threat_levels: List[float],
        actions: List[int],
        title: str = "Threat Level & Actions Over Time",
        filename: str = "threat_timeline.png",
    ) -> str:
        """Dual-axis plot: threat level line + action scatter."""
        steps = np.arange(len(threat_levels))
        action_names = ["no_op", "block_ip", "rate_limit", "isolate"]
        markers = ["o", "s", "^", "D"]
        action_colors = ["#2ecc71", "#e74c3c", "#f39c12", "#9b59b6"]

        fig, ax1 = plt.subplots(figsize=(12, 4))
        ax1.plot(steps, threat_levels, color="gray", linewidth=1.5, label="Threat level")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Threat Level", color="gray")
        ax1.set_ylim(0, 1.05)

        ax2 = ax1.twinx()
        for action_id in range(4):
            idx = [i for i, a in enumerate(actions) if a == action_id]
            if idx:
                ax2.scatter(
                    idx,
                    [action_id] * len(idx),
                    marker=markers[action_id],
                    color=action_colors[action_id],
                    label=action_names[action_id],
                    s=20,
                    alpha=0.7,
                )
        ax2.set_ylabel("Action Taken")
        ax2.set_yticks([0, 1, 2, 3])
        ax2.set_yticklabels(action_names)
        ax2.set_ylim(-0.5, 3.5)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

        ax1.set_title(title)
        ax1.grid(True, alpha=0.2)
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    def plot_attack_success_rate(
        self,
        attack_success_rates: List[float],
        title: str = "Attack Success Rate Over Episodes",
        filename: str = "attack_success_rate.png",
    ) -> str:
        """Line chart of attack success rate over evaluation episodes."""
        fig, ax = plt.subplots(figsize=(10, 4))
        episodes = np.arange(1, len(attack_success_rates) + 1)
        ax.plot(episodes, attack_success_rates, color="crimson", marker="o", markersize=4)
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="50% baseline")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Attack Success Rate")
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.grid(True, alpha=0.3)
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    # ------------------------------------------------------------------
    # NEW visualisation methods
    # ------------------------------------------------------------------

    def plot_policy_comparison(
        self,
        results: dict[str, dict],
        metric: str = "mean_reward",
        title: str = "Policy Comparison",
        filename: str = "policy_comparison.png",
    ) -> str:
        """Bar chart comparing a metric across multiple policies with error bars."""
        names = list(results.keys())
        means = [results[n].get(metric, 0.0) for n in names]
        stds = [results[n].get(f"std_{metric.replace('mean_', '')}", 0.0) for n in names]

        colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(names, means, yerr=stds, capsize=5, color=colors, edgecolor="gray")
        ax.bar_label(bars, fmt="%.1f", padding=5)
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        plt.xticks(rotation=15, ha="right")
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_detection_delay(
        self,
        delays: List[float],
        title: str = "Detection Delay Distribution",
        filename: str = "detection_delay.png",
    ) -> str:
        """Histogram of detection delays across episodes."""
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(delays, bins=20, color="steelblue", edgecolor="white", alpha=0.8)
        ax.axvline(np.mean(delays), color="red", linestyle="--",
                   label=f"Mean = {np.mean(delays):.1f}")
        ax.set_xlabel("Detection Delay (steps)")
        ax.set_ylabel("Frequency")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_multi_seed_rewards(
        self,
        seed_rewards: dict[int, List[float]],
        title: str = "Multi-Seed Reward Distribution",
        filename: str = "multi_seed_rewards.png",
    ) -> str:
        """Box plot of reward distributions across seeds."""
        fig, ax = plt.subplots(figsize=(8, 5))
        labels = [f"Seed {s}" for s in seed_rewards.keys()]
        data = list(seed_rewards.values())
        ax.boxplot(data, labels=labels, patch_artist=True,
                   boxprops=dict(facecolor="lightblue"))
        ax.set_ylabel("Episode Reward")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path
