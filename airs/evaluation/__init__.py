"""
AIRS Evaluation Framework.

Provides:
  - Multi-seed evaluation with confidence intervals
  - Baseline policy comparison
  - Extended metrics (detection delay, FPR, service downtime, cost)
  - Statistical tests (paired t-test, bootstrap CI)
  - Out-of-distribution test scenarios
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

import numpy as np
from scipy import stats

from airs.environment.network_env import NetworkSecurityEnv


# ---------------------------------------------------------------------------
# Protocol for any "policy" that can predict actions
# ---------------------------------------------------------------------------

class PolicyLike(Protocol):
    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int: ...


# ---------------------------------------------------------------------------
# Metrics dataclass
# ---------------------------------------------------------------------------

@dataclass
class EpisodeMetrics:
    reward: float = 0.0
    attack_success_rate: float = 0.0
    detection_delay: int = 0
    false_positive_count: int = 0
    false_positive_rate: float = 0.0
    service_downtime: float = 0.0
    cost_per_episode: float = 0.0
    actions: list[int] = field(default_factory=list)
    threat_levels: list[float] = field(default_factory=list)


@dataclass
class EvalResult:
    """Aggregated evaluation results over multiple episodes."""
    policy_name: str
    attack_mode: str
    intensity: str
    n_episodes: int
    seed: int

    mean_reward: float = 0.0
    std_reward: float = 0.0
    ci95_reward: tuple[float, float] = (0.0, 0.0)

    mean_attack_success: float = 0.0
    mean_detection_delay: float = 0.0
    mean_fpr: float = 0.0
    mean_downtime: float = 0.0
    mean_cost: float = 0.0

    action_counts: dict[str, int] = field(default_factory=dict)
    episode_rewards: list[float] = field(default_factory=list)
    episode_metrics: list[EpisodeMetrics] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate_policy(
    policy: PolicyLike,
    policy_name: str,
    attack_mode: str = "brute_force",
    intensity: str = "medium",
    n_episodes: int = 50,
    seed: int = 42,
    env_kwargs: Optional[dict] = None,
) -> EvalResult:
    """Evaluate a policy over n_episodes and compute extended metrics."""

    env_kw = dict(env_kwargs or {})
    env = NetworkSecurityEnv(attack_mode=attack_mode, intensity=intensity, **env_kw)

    action_names = {0: "no_op", 1: "block_ip", 2: "rate_limit", 3: "isolate_service"}
    action_counts: dict[str, int] = defaultdict(int)
    episode_metrics_list: list[EpisodeMetrics] = []
    episode_rewards: list[float] = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        em = EpisodeMetrics()
        high_threat_steps = 0
        acted_on_high = 0
        first_high_step = -1
        first_action_step = -1
        total_steps = 0

        done = False
        while not done:
            action = policy.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_steps += 1

            em.reward += reward
            em.actions.append(action)
            em.threat_levels.append(info["threat_level"])
            action_counts[action_names.get(action, str(action))] += 1

            # Detection delay: steps between first high-threat and first defensive action
            if info["threat_level"] > NetworkSecurityEnv.HIGH_THREAT_THRESHOLD:
                high_threat_steps += 1
                if first_high_step < 0:
                    first_high_step = total_steps
                if action > 0:
                    acted_on_high += 1
                    if first_action_step < 0:
                        first_action_step = total_steps

            # False positive: defensive action when threat is low
            if info["threat_level"] < NetworkSecurityEnv.LOW_THREAT_THRESHOLD and action > 0:
                em.false_positive_count += 1

            em.cost_per_episode += info.get("service_cost", 0.0)
            em.service_downtime += info.get("service_cost", 0.0)

        # Attack success rate
        if high_threat_steps > 0:
            em.attack_success_rate = 1.0 - (acted_on_high / high_threat_steps)
        else:
            em.attack_success_rate = 0.0

        # Detection delay
        if first_high_step >= 0 and first_action_step >= 0:
            em.detection_delay = first_action_step - first_high_step
        elif first_high_step >= 0:
            em.detection_delay = total_steps - first_high_step  # never responded

        # FPR: false positives / total low-threat steps
        low_threat_steps = sum(
            1 for t in em.threat_levels if t < NetworkSecurityEnv.LOW_THREAT_THRESHOLD
        )
        em.false_positive_rate = (
            em.false_positive_count / max(low_threat_steps, 1)
        )

        episode_metrics_list.append(em)
        episode_rewards.append(em.reward)

    env.close()

    # Aggregate
    rewards = np.array(episode_rewards)
    mean_r = float(np.mean(rewards))
    std_r = float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0
    ci95 = _bootstrap_ci(rewards) if len(rewards) >= 5 else (mean_r, mean_r)

    return EvalResult(
        policy_name=policy_name,
        attack_mode=attack_mode,
        intensity=intensity,
        n_episodes=n_episodes,
        seed=seed,
        mean_reward=mean_r,
        std_reward=std_r,
        ci95_reward=ci95,
        mean_attack_success=float(np.mean([m.attack_success_rate for m in episode_metrics_list])),
        mean_detection_delay=float(np.mean([m.detection_delay for m in episode_metrics_list])),
        mean_fpr=float(np.mean([m.false_positive_rate for m in episode_metrics_list])),
        mean_downtime=float(np.mean([m.service_downtime for m in episode_metrics_list])),
        mean_cost=float(np.mean([m.cost_per_episode for m in episode_metrics_list])),
        action_counts=dict(action_counts),
        episode_rewards=episode_rewards,
        episode_metrics=episode_metrics_list,
    )


# ---------------------------------------------------------------------------
# Multi-seed evaluation
# ---------------------------------------------------------------------------

def multi_seed_evaluate(
    policy: PolicyLike,
    policy_name: str,
    attack_mode: str = "brute_force",
    intensity: str = "medium",
    n_episodes: int = 50,
    seeds: list[int] | None = None,
    env_kwargs: Optional[dict] = None,
) -> dict[str, Any]:
    """Run evaluation across multiple seeds and aggregate.

    Returns a dict with mean/std/CI across seeds.
    """
    if seeds is None:
        seeds = [42, 123, 256, 512, 1024]

    per_seed_means: list[float] = []
    all_results: list[EvalResult] = []

    for s in seeds:
        result = evaluate_policy(
            policy, policy_name, attack_mode, intensity, n_episodes, seed=s,
            env_kwargs=env_kwargs,
        )
        per_seed_means.append(result.mean_reward)
        all_results.append(result)

    arr = np.array(per_seed_means)
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    se = std / np.sqrt(n) if n > 1 else 0.0
    t_val = stats.t.ppf(0.975, df=max(n - 1, 1))
    ci95 = (mean - t_val * se, mean + t_val * se)

    return {
        "policy_name": policy_name,
        "seeds": seeds,
        "per_seed_means": per_seed_means,
        "mean_reward": mean,
        "std_reward": std,
        "ci95_reward": ci95,
        "mean_attack_success": float(np.mean([r.mean_attack_success for r in all_results])),
        "mean_detection_delay": float(np.mean([r.mean_detection_delay for r in all_results])),
        "mean_fpr": float(np.mean([r.mean_fpr for r in all_results])),
        "mean_downtime": float(np.mean([r.mean_downtime for r in all_results])),
        "mean_cost": float(np.mean([r.mean_cost for r in all_results])),
        "results": all_results,
    }


# ---------------------------------------------------------------------------
# Statistical comparison
# ---------------------------------------------------------------------------

def compare_policies(
    result_a: dict[str, Any],
    result_b: dict[str, Any],
) -> dict[str, Any]:
    """Paired t-test + bootstrap CI comparing two multi-seed evaluation results."""
    a = np.array(result_a["per_seed_means"])
    b = np.array(result_b["per_seed_means"])
    diff = a - b

    # Paired t-test
    if len(diff) > 1 and np.std(diff, ddof=1) > 0:
        t_stat, p_value = stats.ttest_rel(a, b)
    else:
        t_stat, p_value = 0.0, 1.0

    # Bootstrap CI on the mean difference
    boot_ci = _bootstrap_ci(diff)

    return {
        "policy_a": result_a["policy_name"],
        "policy_b": result_b["policy_name"],
        "mean_diff": float(np.mean(diff)),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "bootstrap_ci_95": boot_ci,
        "significant_at_005": p_value < 0.05,
    }


# ---------------------------------------------------------------------------
# Out-of-distribution test scenarios
# ---------------------------------------------------------------------------

OOD_SCENARIOS = {
    "bursty_traffic": {
        "attack_mode": "flood",
        "intensity": "high",
        "noisy_observations": True,
        "noise_std": 0.15,
    },
    "noisy_telemetry": {
        "attack_mode": "brute_force",
        "intensity": "medium",
        "noisy_observations": True,
        "noise_std": 0.20,
        "partial_observability": True,
        "mask_probability": 0.2,
    },
    "unseen_attack_combo": {
        "attack_mode": "multi_stage",
        "intensity": "high",
    },
}


def run_ood_tests(
    policy: PolicyLike,
    policy_name: str,
    n_episodes: int = 20,
    seed: int = 42,
) -> dict[str, EvalResult]:
    """Evaluate the policy on out-of-distribution scenarios."""
    results = {}
    for scenario_name, scenario_cfg in OOD_SCENARIOS.items():
        attack_mode = scenario_cfg.pop("attack_mode", "brute_force")
        intensity = scenario_cfg.pop("intensity", "medium")
        env_kwargs = {k: v for k, v in scenario_cfg.items()}
        # Restore popped keys for next call
        OOD_SCENARIOS[scenario_name]["attack_mode"] = attack_mode
        OOD_SCENARIOS[scenario_name]["intensity"] = intensity

        result = evaluate_policy(
            policy, f"{policy_name}_{scenario_name}",
            attack_mode=attack_mode,
            intensity=intensity,
            n_episodes=n_episodes,
            seed=seed,
            env_kwargs=env_kwargs,
        )
        results[scenario_name] = result
    return results


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

def save_results_csv(results: list[EvalResult], path: str):
    """Save evaluation results to CSV."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = [
        "policy_name", "attack_mode", "intensity", "n_episodes", "seed",
        "mean_reward", "std_reward", "ci95_low", "ci95_high",
        "mean_attack_success", "mean_detection_delay", "mean_fpr",
        "mean_downtime", "mean_cost",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "policy_name": r.policy_name,
                "attack_mode": r.attack_mode,
                "intensity": r.intensity,
                "n_episodes": r.n_episodes,
                "seed": r.seed,
                "mean_reward": f"{r.mean_reward:.4f}",
                "std_reward": f"{r.std_reward:.4f}",
                "ci95_low": f"{r.ci95_reward[0]:.4f}",
                "ci95_high": f"{r.ci95_reward[1]:.4f}",
                "mean_attack_success": f"{r.mean_attack_success:.4f}",
                "mean_detection_delay": f"{r.mean_detection_delay:.4f}",
                "mean_fpr": f"{r.mean_fpr:.4f}",
                "mean_downtime": f"{r.mean_downtime:.4f}",
                "mean_cost": f"{r.mean_cost:.4f}",
            })


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    data: np.ndarray,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap percentile confidence interval."""
    rng = np.random.default_rng(seed)
    n = len(data)
    if n < 2:
        m = float(data[0]) if n == 1 else 0.0
        return (m, m)
    boot_means = np.array([
        float(np.mean(rng.choice(data, size=n, replace=True)))
        for _ in range(n_bootstrap)
    ])
    lo = float(np.percentile(boot_means, 100 * alpha / 2))
    hi = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return (lo, hi)
