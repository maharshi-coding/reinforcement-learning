"""
train.py – Train an AIRS RL agent.

Usage
-----
python scripts/train.py [--algorithm dqn|ppo] [--attack_mode brute_force|flood|adaptive|multi_stage]
                        [--intensity low|medium|high] [--timesteps 50000]
                        [--output_dir results] [--model_path models/airs_agent]
                        [--n_envs 1] [--seed 42] [--config configs/default.yaml]
                        [--curriculum]

Example
-------
python scripts/train.py --algorithm dqn --attack_mode adaptive --timesteps 50000
python scripts/train.py --algorithm ppo --curriculum --config configs/default.yaml
"""

import argparse
import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airs.agent.rl_agent import AIRSAgent
from airs.config import load_config
from airs.visualization.visualizer import AIRSVisualizer


def parse_args():
    p = argparse.ArgumentParser(description="Train AIRS RL agent")
    p.add_argument("--algorithm",   default=None, choices=["dqn", "ppo"])
    p.add_argument("--attack_mode", default=None,
                   choices=["brute_force", "flood", "adaptive", "multi_stage"])
    p.add_argument("--intensity",   default=None,
                   choices=["low", "medium", "high"])
    p.add_argument("--timesteps",   default=None, type=int)
    p.add_argument("--output_dir",  default=None)
    p.add_argument("--model_path",  default=None)
    p.add_argument("--n_envs",      default=None, type=int)
    p.add_argument("--seed",        default=None, type=int)
    p.add_argument("--config",      default="configs/default.yaml",
                   help="Path to YAML config file")
    p.add_argument("--curriculum",  action="store_true",
                   help="Enable curriculum learning")
    return p.parse_args()


def main():
    args = parse_args()

    # Load YAML config
    cfg = load_config(args.config)
    env_cfg = cfg.get("environment", {})
    agent_cfg = cfg.get("agent", {})
    train_cfg = cfg.get("training", {})
    reward_cfg = cfg.get("reward", {})
    paths_cfg = cfg.get("paths", {})

    # CLI overrides take precedence
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

    use_curriculum = args.curriculum or train_cfg.get("curriculum", {}).get("enabled", False)

    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)

    # Build env_kwargs from config
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

    # Build algo kwargs from config
    algo_kwargs = dict(agent_cfg.get(algorithm, {}))

    print(
        f"[AIRS] Training {algorithm.upper()} | "
        f"attack={attack_mode} | intensity={intensity} | "
        f"timesteps={timesteps:,} | n_envs={n_envs} | seed={seed}"
    )

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

    if use_curriculum:
        stages = train_cfg.get("curriculum", {}).get("stages", [
            {"intensity": "low", "timesteps": 15000},
            {"intensity": "medium", "timesteps": 20000},
            {"intensity": "high", "timesteps": 15000},
        ])
        agent.train_curriculum(stages, **train_kwargs)
    else:
        agent.train(total_timesteps=timesteps, **train_kwargs)

    agent.save(model_path)
    print(f"[AIRS] Model saved → {model_path}")

    # Plot training reward curve
    if agent.episode_rewards:
        viz = AIRSVisualizer(output_dir=output_dir)
        path = viz.plot_reward_curve(
            agent.episode_rewards,
            title=f"Training Reward ({algorithm.upper()}, {attack_mode})",
            filename=f"training_reward_{algorithm}_{attack_mode}.png",
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
