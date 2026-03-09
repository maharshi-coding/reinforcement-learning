"""
evaluate.py – Evaluate a trained AIRS agent with comprehensive metrics.

Features:
  - Multi-seed evaluation with 95% confidence intervals
  - Baseline policy comparison (noop, random, rule-based)
  - Extended metrics (detection delay, FPR, downtime, cost)
  - Statistical tests (paired t-test, bootstrap CI)
  - Out-of-distribution scenario tests
  - CSV artifact export

Usage
-----
python scripts/evaluate.py [--algorithm dqn|ppo] [--attack_mode brute_force|flood|adaptive|multi_stage]
                           [--intensity low|medium|high] [--episodes 50]
                           [--model_path models/dqn_agent] [--output_dir results]
                           [--multi_seed] [--baselines] [--ood] [--config configs/default.yaml]
"""

import argparse
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airs.agent.baselines import get_baseline
from airs.agent.rl_agent import AIRSAgent
from airs.config import load_config
from airs.evaluation import (
    EvalResult,
    compare_policies,
    evaluate_policy,
    multi_seed_evaluate,
    run_ood_tests,
    save_results_csv,
)
from airs.visualization.visualizer import AIRSVisualizer


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate AIRS RL agent")
    p.add_argument("--algorithm",   default=None, choices=["dqn", "ppo"])
    p.add_argument("--attack_mode", default=None,
                   choices=["brute_force", "flood", "adaptive", "multi_stage"])
    p.add_argument("--intensity",   default=None,
                   choices=["low", "medium", "high"])
    p.add_argument("--episodes",    default=None, type=int)
    p.add_argument("--model_path",  default=None)
    p.add_argument("--output_dir",  default=None)
    p.add_argument("--config",      default="configs/default.yaml")
    p.add_argument("--seed",        default=None, type=int)
    p.add_argument("--multi_seed",  action="store_true",
                   help="Run multi-seed evaluation with 95%% CI")
    p.add_argument("--baselines",   action="store_true",
                   help="Compare against baseline policies")
    p.add_argument("--ood",         action="store_true",
                   help="Run out-of-distribution tests")
    return p.parse_args()


def _print_result(r: EvalResult):
    """Pretty-print a single evaluation result."""
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

    # Load config
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

    # Create the trained agent
    agent = AIRSAgent(
        algorithm=algorithm,
        attack_mode=attack_mode,
        intensity=intensity,
        model_path=model_path,
        seed=seed,
        env_kwargs=env_kwargs,
    )

    all_results: list[EvalResult] = []

    # ─── Standard evaluation ───
    result = evaluate_policy(
        agent, f"{algorithm}_agent", attack_mode, intensity, episodes,
        seed=seed, env_kwargs=env_kwargs,
    )
    _print_result(result)
    all_results.append(result)

    # ─── Multi-seed evaluation ───
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

    # ─── Baseline comparison ───
    agent_ms = None
    if args.baselines:
        baseline_names = eval_cfg.get("baselines", ["always_noop", "random_policy", "rule_based_threshold"])
        print("\n[AIRS] Running baseline comparisons...")

        # Multi-seed for agent (for fair comparison)
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
            # Print baseline results
            for r in bl_ms["results"]:
                all_results.append(r)

            comp = compare_policies(agent_ms, bl_ms)
            print(f"\n  {algorithm}_agent vs {bl_name}:")
            print(f"    Mean diff: {comp['mean_diff']:.2f}")
            print(f"    t-stat:    {comp['t_stat']:.3f}, p={comp['p_value']:.4f}")
            print(f"    95% CI:    [{comp['bootstrap_ci_95'][0]:.2f}, {comp['bootstrap_ci_95'][1]:.2f}]")
            print(f"    Significant (p<0.05): {comp['significant_at_005']}")

    # ─── OOD tests ───
    if args.ood:
        print("\n[AIRS] Running out-of-distribution tests...")
        ood_results = run_ood_tests(agent, f"{algorithm}_agent", n_episodes=20, seed=seed)
        for scenario, ood_r in ood_results.items():
            _print_result(ood_r)
            all_results.append(ood_r)

    # ─── Save artifacts ───
    os.makedirs(output_dir, exist_ok=True)
    save_results_csv(all_results, os.path.join(output_dir, "eval_summary.csv"))

    # ─── Visualisations ───
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

    print(f"\n[AIRS] Plots and CSV saved to {output_dir}/")


if __name__ == "__main__":
    main()
