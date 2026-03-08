"""
evaluate_agent.py – Evaluate a trained AIRS agent with comprehensive metrics.

Features:
  - Multi-seed evaluation with 95% confidence intervals
  - Baseline policy comparison (noop, random, rule-based)
  - Extended metrics (detection delay, FPR, downtime, cost)
  - Statistical tests (paired t-test, bootstrap CI)
  - Out-of-distribution scenario tests
  - CSV artifact export
  - MLflow logging

Usage
-----
python src/evaluation/evaluate_agent.py --config configs/training_config.yaml
python src/evaluation/evaluate_agent.py --algorithm dqn --multi_seed --baselines --ood
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml

_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.agent.rl_agent import AIRSAgent
from src.baselines import get_baseline
from src.environment.intrusion_env import IntrusionEnv
from src.evaluation.metrics import (
    EvalResult,
    compare_policies,
    evaluate_policy,
    multi_seed_evaluate,
    run_ood_tests,
    save_results_csv,
)
from src.visualization.visualizer import AIRSVisualizer

try:
    import mlflow
    _HAS_MLFLOW = True
except ImportError:
    _HAS_MLFLOW = False


def load_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate AIRS RL agent")
    p.add_argument("--algorithm", default=None, choices=["dqn", "ppo"])
    p.add_argument("--attack_mode", default=None,
                   choices=["brute_force", "flood", "adaptive", "multi_stage"])
    p.add_argument("--intensity", default=None, choices=["low", "medium", "high"])
    p.add_argument("--episodes", default=None, type=int)
    p.add_argument("--model_path", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--config", default="configs/training_config.yaml")
    p.add_argument("--seed", default=None, type=int)
    p.add_argument("--multi_seed", action="store_true",
                   help="Run multi-seed evaluation with 95%% CI")
    p.add_argument("--baselines", action="store_true",
                   help="Compare against baseline policies")
    p.add_argument("--ood", action="store_true",
                   help="Run out-of-distribution tests")
    p.add_argument("--no-mlflow", action="store_true",
                   help="Disable MLflow tracking")
    return p.parse_args()


def _print_result(r: EvalResult):
    print(f"\n{'='*60}")
    print(f" {r.policy_name} | {r.attack_mode} | {r.intensity}")
    print(f"{'='*60}")
    print(f"  Episodes:              {r.n_episodes}")
    print(f"  Mean reward:           {r.mean_reward:.2f} ± {r.std_reward:.2f}")
    print(f"  95% CI:                [{r.ci95_reward[0]:.2f}, {r.ci95_reward[1]:.2f}]")
    print(f"  Attack success rate:   {r.mean_attack_success*100:.1f}%")
    print(f"  Detection delay:       {r.mean_detection_delay:.1f} steps")
    print(f"  False positive rate:   {r.mean_fpr*100:.1f}%")
    print(f"  Service downtime:      {r.mean_downtime:.2f}")
    print(f"  Cost per episode:      {r.mean_cost:.2f}")
    print(f"  Actions:               {r.action_counts}")
    print(f"{'='*60}")


def main():
    args = parse_args()
    cfg = load_config(args.config)

    env_cfg = cfg.get("environment", {})
    agent_cfg = cfg.get("agent", {})
    eval_cfg = cfg.get("evaluation", {})
    reward_cfg = cfg.get("reward", {})
    paths_cfg = cfg.get("paths", {})

    algorithm = args.algorithm or agent_cfg.get("algorithm", "dqn")
    attack_mode = args.attack_mode or env_cfg.get("attack_mode", "brute_force")
    intensity = args.intensity or env_cfg.get("intensity", "medium")
    episodes = args.episodes or eval_cfg.get("episodes", 50)
    output_dir = args.output_dir or paths_cfg.get("results_dir", "results")
    model_path = args.model_path or os.path.join(
        paths_cfg.get("model_dir", "models"), f"{algorithm}_agent"
    )
    seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    use_mlflow = _HAS_MLFLOW and not args.no_mlflow

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

    print(
        f"[AIRS] Evaluating {algorithm.upper()} | "
        f"attack={attack_mode} | intensity={intensity} | "
        f"episodes={episodes}"
    )

    agent = AIRSAgent(
        algorithm=algorithm,
        attack_mode=attack_mode,
        intensity=intensity,
        model_path=model_path,
        seed=seed,
        env_kwargs=env_kwargs,
    )

    all_results: list[EvalResult] = []

    # ── MLflow ──
    if use_mlflow:
        tracking_uri = paths_cfg.get("mlflow_uri", "experiments/training_runs")
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment("AIRS_Evaluation")
        mlflow.start_run(run_name=f"eval_{algorithm}_{attack_mode}_{intensity}")

    # ── Standard evaluation ──
    result = evaluate_policy(
        agent, f"{algorithm}_agent", attack_mode, intensity, episodes,
        seed=seed, env_kwargs=env_kwargs,
    )
    _print_result(result)
    all_results.append(result)

    if use_mlflow:
        mlflow.log_metrics({
            "mean_reward": result.mean_reward,
            "attack_success_rate": result.mean_attack_success,
            "detection_delay": result.mean_detection_delay,
            "false_positive_rate": result.mean_fpr,
            "service_downtime": result.mean_downtime,
        })

    # ── Multi-seed evaluation ──
    if args.multi_seed:
        n_seeds = eval_cfg.get("n_seeds", 5)
        seeds = list(range(seed, seed + n_seeds))
        ms_result = multi_seed_evaluate(
            agent, f"{algorithm}_agent", attack_mode, intensity, episodes,
            seeds=seeds, env_kwargs=env_kwargs,
        )
        print(f"\n[AIRS] Multi-seed ({n_seeds} seeds):")
        print(f"  Mean reward:  {ms_result['mean_reward']:.2f} ± {ms_result['std_reward']:.2f}")
        print(f"  95% CI:       [{ms_result['ci95_reward'][0]:.2f}, {ms_result['ci95_reward'][1]:.2f}]")
        print(f"  Det. delay:   {ms_result['mean_detection_delay']:.1f}")
        print(f"  FPR:          {ms_result['mean_fpr']*100:.1f}%")

        if use_mlflow:
            mlflow.log_metrics({
                "ms_mean_reward": ms_result["mean_reward"],
                "ms_std_reward": ms_result["std_reward"],
                "ms_ci95_low": ms_result["ci95_reward"][0],
                "ms_ci95_high": ms_result["ci95_reward"][1],
            })

    # ── Baseline comparison ──
    if args.baselines:
        baseline_names = eval_cfg.get("baselines", ["always_noop", "random_policy", "rule_based_threshold"])
        print("\n[AIRS] Running baseline comparisons...")

        n_seeds = eval_cfg.get("n_seeds", 5)
        seeds = list(range(seed, seed + n_seeds))
        agent_ms = multi_seed_evaluate(
            agent, f"{algorithm}_agent", attack_mode, intensity, episodes,
            seeds=seeds, env_kwargs=env_kwargs,
        )

        for bl_name in baseline_names:
            bl_policy = get_baseline(bl_name)
            bl_ms = multi_seed_evaluate(
                bl_policy, bl_name, attack_mode, intensity, episodes,
                seeds=seeds, env_kwargs=env_kwargs,
            )
            for r in bl_ms["results"]:
                all_results.append(r)

            comp = compare_policies(agent_ms, bl_ms)
            print(f"\n  {algorithm}_agent vs {bl_name}:")
            print(f"    Mean diff: {comp['mean_diff']:.2f}")
            print(f"    t-stat:    {comp['t_stat']:.3f}, p={comp['p_value']:.4f}")
            print(f"    95% CI:    [{comp['bootstrap_ci_95'][0]:.2f}, {comp['bootstrap_ci_95'][1]:.2f}]")
            print(f"    Significant (p<0.05): {comp['significant_at_005']}")

            if use_mlflow:
                mlflow.log_metrics({
                    f"vs_{bl_name}_mean_diff": comp["mean_diff"],
                    f"vs_{bl_name}_p_value": comp["p_value"],
                })

    # ── OOD tests ──
    if args.ood:
        print("\n[AIRS] Running out-of-distribution tests...")
        ood_results = run_ood_tests(agent, f"{algorithm}_agent", n_episodes=20, seed=seed)
        for scenario, ood_r in ood_results.items():
            _print_result(ood_r)
            all_results.append(ood_r)
            if use_mlflow:
                mlflow.log_metric(f"ood_{scenario}_reward", ood_r.mean_reward)

    # ── Save artifacts ──
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "eval_summary.csv")
    save_results_csv(all_results, csv_path)

    # ── Visualisations ──
    viz = AIRSVisualizer(output_dir=output_dir)
    prefix = f"eval_{algorithm}_{attack_mode}"

    viz.plot_reward_curve(
        result.episode_rewards,
        title=f"Evaluation Reward ({algorithm.upper()}, {attack_mode})",
        filename=f"{prefix}_reward.png",
    )
    viz.plot_action_distribution(
        result.action_counts,
        title=f"Action Distribution ({attack_mode})",
        filename=f"{prefix}_actions.png",
    )
    viz.plot_attack_success_rate(
        [m.attack_success_rate for m in result.episode_metrics],
        title=f"Attack Success Rate ({attack_mode}, {intensity})",
        filename=f"{prefix}_attack_success.png",
    )
    if result.episode_metrics:
        first_ep = result.episode_metrics[0]
        viz.plot_threat_timeline(
            first_ep.threat_levels,
            first_ep.actions,
            title=f"Threat Timeline – Episode 1 ({attack_mode})",
            filename=f"{prefix}_threat_timeline.png",
        )
        viz.plot_detection_delay(
            [m.detection_delay for m in result.episode_metrics],
            title=f"Detection Delay ({attack_mode})",
            filename=f"{prefix}_detection_delay.png",
        )

    if use_mlflow:
        mlflow.log_artifact(csv_path)
        for f in os.listdir(output_dir):
            if f.endswith(".png") and f.startswith(prefix):
                mlflow.log_artifact(os.path.join(output_dir, f))
        mlflow.end_run()

    print(f"\n[AIRS] Plots and CSV saved to {output_dir}/")


if __name__ == "__main__":
    main()
