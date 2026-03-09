#!/usr/bin/env python3
"""
evaluate_all.py – Evaluate DQN and PPO agents across ALL 12 scenario combinations.

Produces:
  - Per-scenario metrics (reward, FPR, detection delay, cost)
  - DQN vs PPO comparison tables
  - Charts: reward by intensity, reward by attack mode, algorithm comparison
  - CSV export of all results

Usage:
  python scripts/evaluate_all.py
  python scripts/evaluate_all.py --episodes 30
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airs.agent.rl_agent import AIRSAgent
from airs.agent.baselines import get_baseline
from airs.config import load_config
from airs.environment.network_env import NetworkSecurityEnv
from airs.evaluation import evaluate_policy

ATTACK_MODES = ["brute_force", "flood", "adaptive", "multi_stage"]
INTENSITIES = ["low", "medium", "high"]
ALGORITHMS = ["dqn", "ppo"]


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate all scenarios")
    p.add_argument("--episodes", default=30, type=int)
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--output_dir", default="results")
    p.add_argument("--seed", default=42, type=int)
    return p.parse_args()


def evaluate_agent(algo, model_path, episodes, seed, env_kwargs):
    """Evaluate one algorithm across all 12 scenarios."""
    rows = []
    for mode in ATTACK_MODES:
        for intensity in INTENSITIES:
            agent = AIRSAgent(
                algorithm=algo,
                attack_mode=mode,
                intensity=intensity,
                model_path=model_path,
                seed=seed,
                env_kwargs=env_kwargs,
            )
            result = evaluate_policy(
                agent, f"{algo}_agent", mode, intensity,
                n_episodes=episodes, seed=seed, env_kwargs=env_kwargs,
            )
            row = {
                "algorithm": algo.upper(),
                "attack_mode": mode,
                "intensity": intensity,
                "mean_reward": result.mean_reward,
                "std_reward": result.std_reward,
                "mean_fpr": result.mean_fpr,
                "mean_detection_delay": result.mean_detection_delay,
                "mean_cost": result.mean_cost,
                "mean_downtime": result.mean_downtime,
                "mean_attack_success": result.mean_attack_success,
                "action_counts": result.action_counts,
            }
            rows.append(row)
            print(f"  {algo.upper()} | {mode:15s} | {intensity:6s} | "
                  f"reward={result.mean_reward:+7.1f} ± {result.std_reward:5.1f} | "
                  f"FPR={result.mean_fpr*100:4.1f}% | "
                  f"delay={result.mean_detection_delay:4.1f}")
    return rows


def evaluate_baselines(episodes, seed, env_kwargs):
    """Evaluate baseline policies across all scenarios."""
    rows = []
    baseline_names = ["always_noop", "random_policy", "rule_based_threshold"]
    for bl_name in baseline_names:
        bl = get_baseline(bl_name)
        for mode in ATTACK_MODES:
            for intensity in INTENSITIES:
                result = evaluate_policy(
                    bl, bl_name, mode, intensity,
                    n_episodes=episodes, seed=seed, env_kwargs=env_kwargs,
                )
                rows.append({
                    "algorithm": bl_name,
                    "attack_mode": mode,
                    "intensity": intensity,
                    "mean_reward": result.mean_reward,
                    "std_reward": result.std_reward,
                    "mean_fpr": result.mean_fpr,
                    "mean_detection_delay": result.mean_detection_delay,
                    "mean_cost": result.mean_cost,
                    "mean_downtime": result.mean_downtime,
                    "mean_attack_success": result.mean_attack_success,
                    "action_counts": result.action_counts,
                })
    return rows


def save_csv(rows, path):
    """Save evaluation results to CSV."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df = pd.DataFrame(rows)
    cols = ["algorithm", "attack_mode", "intensity", "mean_reward", "std_reward",
            "mean_fpr", "mean_detection_delay", "mean_cost", "mean_downtime",
            "mean_attack_success"]
    df[cols].to_csv(path, index=False, float_format="%.4f")
    return df


def plot_reward_by_intensity(df, output_dir):
    """Bar chart: mean reward by intensity, grouped by algorithm."""
    fig, ax = plt.subplots(figsize=(10, 5))
    algos = df["algorithm"].unique()
    x = np.arange(len(INTENSITIES))
    width = 0.8 / len(algos)
    colours = {"DQN": "#4C9AFF", "PPO": "#F97316", "always_noop": "#888",
               "random_policy": "#aaa", "rule_based_threshold": "#6c6"}

    for i, algo in enumerate(algos):
        means = []
        stds = []
        for intensity in INTENSITIES:
            subset = df[(df["algorithm"] == algo) & (df["intensity"] == intensity)]
            means.append(subset["mean_reward"].mean())
            stds.append(subset["mean_reward"].std())
        ax.bar(x + i * width, means, width, yerr=stds, label=algo,
               color=colours.get(algo, f"C{i}"), capsize=3, alpha=0.85)

    ax.set_xticks(x + width * (len(algos) - 1) / 2)
    ax.set_xticklabels(INTENSITIES)
    ax.set_xlabel("Intensity")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Mean Reward by Intensity Level")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    path = os.path.join(output_dir, "reward_by_intensity.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_reward_by_attack_mode(df, output_dir):
    """Bar chart: mean reward by attack mode, grouped by algorithm."""
    fig, ax = plt.subplots(figsize=(12, 5))
    algos = df["algorithm"].unique()
    x = np.arange(len(ATTACK_MODES))
    width = 0.8 / len(algos)
    colours = {"DQN": "#4C9AFF", "PPO": "#F97316", "always_noop": "#888",
               "random_policy": "#aaa", "rule_based_threshold": "#6c6"}

    for i, algo in enumerate(algos):
        means = []
        for mode in ATTACK_MODES:
            subset = df[(df["algorithm"] == algo) & (df["attack_mode"] == mode)]
            means.append(subset["mean_reward"].mean())
        ax.bar(x + i * width, means, width, label=algo,
               color=colours.get(algo, f"C{i}"), alpha=0.85)

    ax.set_xticks(x + width * (len(algos) - 1) / 2)
    ax.set_xticklabels(ATTACK_MODES, rotation=15)
    ax.set_xlabel("Attack Mode")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Mean Reward by Attack Mode")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    path = os.path.join(output_dir, "reward_by_attack_mode.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_algorithm_comparison(df, output_dir):
    """Grouped bar: DQN vs PPO per scenario."""
    rl_df = df[df["algorithm"].isin(["DQN", "PPO"])]
    if len(rl_df["algorithm"].unique()) < 2:
        return

    scenarios = []
    dqn_rewards = []
    ppo_rewards = []
    for mode in ATTACK_MODES:
        for intensity in INTENSITIES:
            label = f"{mode}\n{intensity}"
            scenarios.append(label)
            d = rl_df[(rl_df["algorithm"] == "DQN") & (rl_df["attack_mode"] == mode) & (rl_df["intensity"] == intensity)]
            p = rl_df[(rl_df["algorithm"] == "PPO") & (rl_df["attack_mode"] == mode) & (rl_df["intensity"] == intensity)]
            dqn_rewards.append(d["mean_reward"].values[0] if len(d) > 0 else 0)
            ppo_rewards.append(p["mean_reward"].values[0] if len(p) > 0 else 0)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(scenarios))
    width = 0.35
    ax.bar(x - width/2, dqn_rewards, width, label="DQN", color="#4C9AFF", alpha=0.85)
    ax.bar(x + width/2, ppo_rewards, width, label="PPO", color="#F97316", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=8)
    ax.set_ylabel("Mean Reward")
    ax.set_title("DQN vs PPO – All 12 Scenarios")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    path = os.path.join(output_dir, "dqn_vs_ppo_comparison.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_heatmap(df, output_dir):
    """Heatmaps: reward per (attack_mode × intensity) for DQN and PPO."""
    for algo in ["DQN", "PPO"]:
        sub = df[df["algorithm"] == algo]
        if sub.empty:
            continue
        pivot = sub.pivot_table(values="mean_reward", index="attack_mode",
                                columns="intensity", aggfunc="mean")
        # Reorder
        pivot = pivot.reindex(index=ATTACK_MODES, columns=INTENSITIES)

        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(INTENSITIES)))
        ax.set_xticklabels(INTENSITIES)
        ax.set_yticks(range(len(ATTACK_MODES)))
        ax.set_yticklabels(ATTACK_MODES)
        ax.set_title(f"{algo} – Mean Reward Heatmap")

        # Annotate cells
        for i in range(len(ATTACK_MODES)):
            for j in range(len(INTENSITIES)):
                val = pivot.values[i, j]
                ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                        fontsize=12, fontweight="bold",
                        color="white" if val < pivot.values.mean() else "black")

        fig.colorbar(im, ax=ax, label="Mean Reward")
        path = os.path.join(output_dir, f"heatmap_{algo.lower()}.png")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved: {path}")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    env_cfg = cfg.get("environment", {})
    reward_cfg = cfg.get("reward", {})
    paths_cfg = cfg.get("paths", {})
    output_dir = args.output_dir

    env_kwargs = {
        "reward_cfg": reward_cfg,
        "temporal_window": env_cfg.get("temporal_window", 1),
    }

    all_rows = []

    # Evaluate RL agents
    for algo in ALGORITHMS:
        model_path = os.path.join(paths_cfg.get("model_dir", "models"), f"{algo}_agent")
        if not os.path.exists(model_path + ".zip") and not os.path.exists(model_path):
            print(f"[SKIP] No model found at {model_path}")
            continue
        print(f"\n{'='*60}")
        print(f" Evaluating {algo.upper()} across all scenarios")
        print(f"{'='*60}")
        rows = evaluate_agent(algo, model_path, args.episodes, args.seed, env_kwargs)
        all_rows.extend(rows)

    # Evaluate baselines
    print(f"\n{'='*60}")
    print(f" Evaluating baselines")
    print(f"{'='*60}")
    baseline_rows = evaluate_baselines(args.episodes, args.seed, env_kwargs)
    all_rows.extend(baseline_rows)

    # Save CSV
    csv_path = os.path.join(output_dir, "eval_all_scenarios.csv")
    df = save_csv(all_rows, csv_path)
    print(f"\n  CSV saved: {csv_path}")

    # Generate charts
    print(f"\nGenerating charts...")
    os.makedirs(output_dir, exist_ok=True)
    plot_reward_by_intensity(df, output_dir)
    plot_reward_by_attack_mode(df, output_dir)
    plot_algorithm_comparison(df, output_dir)
    plot_heatmap(df, output_dir)

    # Print summary table
    print(f"\n{'='*70}")
    print(f" SUMMARY: Mean Reward by Algorithm")
    print(f"{'='*70}")
    summary = df.groupby("algorithm")["mean_reward"].agg(["mean", "std", "min", "max"])
    print(summary.to_string())

    print(f"\n{'='*70}")
    print(f" DQN vs PPO by Intensity")
    print(f"{'='*70}")
    rl_df = df[df["algorithm"].isin(["DQN", "PPO"])]
    pivot = rl_df.pivot_table(values="mean_reward", index="intensity",
                               columns="algorithm", aggfunc="mean")
    print(pivot.to_string())

    print(f"\n{'='*70}")
    print(f" DQN vs PPO by Attack Mode")
    print(f"{'='*70}")
    pivot2 = rl_df.pivot_table(values="mean_reward", index="attack_mode",
                                columns="algorithm", aggfunc="mean")
    print(pivot2.to_string())


if __name__ == "__main__":
    main()
