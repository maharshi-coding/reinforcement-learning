#!/usr/bin/env python3
"""
watch_agent.py – Visualise a trained AIRS agent defending against attacks.

Opens a pygame window showing:
  - Network topology (server, firewall, database) with threat-coloured glow
  - Animated attacker with red particle beams
  - Defensive action indicators (block, rate-limit, isolate)
  - Real-time HUD: threat gauge, reward bar, metrics, action timeline

Controls:
  ESC / close window   – quit
  UP / DOWN            – speed up / slow down (FPS)
  SPACE                – pause / resume
  R                    – restart episode

Usage:
  python scripts/watch_agent.py
  python scripts/watch_agent.py --model_path models/dqn_agent --attack_mode adaptive --intensity high
  python scripts/watch_agent.py --episodes 5 --fps 6
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from airs.environment.network_env import NetworkSecurityEnv
from src.visualization.renderer import AIRSRenderer

# Map action int → name (must match env)
ACTION_NAMES = {0: "observe", 1: "block_ip", 2: "rate_limit", 3: "isolate_service"}


def _load_model(model_path: str, algorithm: str):
    """Load a Stable-Baselines3 model from disk."""
    if algorithm == "dqn":
        from stable_baselines3 import DQN as Algo
    elif algorithm == "a2c":
        from stable_baselines3 import A2C as Algo
    else:
        from stable_baselines3 import PPO as Algo

    for p in [model_path, model_path + ".zip"]:
        if os.path.exists(p):
            return Algo.load(p)
    raise FileNotFoundError(f"No model found at {model_path}(.zip)")


def run_episode(env, model, renderer, deterministic=True):
    """Run one episode with live pygame rendering. Returns total reward."""
    obs, info = env.reset()
    episode_reward = 0.0
    step = 0
    paused = False

    while True:
        # Handle pygame events (pause, speed, quit)
        import pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None  # signal quit
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return None
                if event.key == pygame.K_SPACE:
                    paused = not paused
                if event.key == pygame.K_UP:
                    renderer.set_fps(renderer.FPS + 2)
                if event.key == pygame.K_DOWN:
                    renderer.set_fps(renderer.FPS - 2)
                if event.key == pygame.K_r:
                    return episode_reward  # restart episode

        if paused:
            time.sleep(0.05)
            continue

        # Agent picks an action
        action, _ = model.predict(obs, deterministic=deterministic)
        action = int(action)

        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        step += 1

        # Build state dict for renderer
        # Extract raw telemetry from the flat obs (first 6 values of the latest frame)
        frame = obs[-6:]  # most recent temporal frame
        state = {
            "step": step,
            "action": action,
            "action_name": ACTION_NAMES.get(action, "?"),
            "reward": float(reward),
            "episode_reward": float(episode_reward),
            "threat_level": float(info.get("threat_level", frame[4])),
            "service_cost": float(info.get("service_cost", 0.0)),
            "phase": info.get("phase", ""),
            "traffic_rate": float(frame[0] * env.TRAFFIC_MAX),
            "failed_logins": float(frame[1] * env.LOGINS_MAX),
            "cpu": float(frame[2]),
            "memory": float(frame[3]),
        }

        alive = renderer.render_frame(state)
        if not alive:
            return None

        if terminated or truncated:
            # Pause briefly at the end so user sees final state
            time.sleep(1.0)
            break

    return episode_reward


def main():
    parser = argparse.ArgumentParser(description="Watch a trained AIRS agent in action")
    parser.add_argument("--model_path", default="models/dqn_agent",
                        help="Path to trained model (without .zip)")
    parser.add_argument("--algorithm", default="dqn", choices=["dqn", "ppo", "a2c"])
    parser.add_argument("--attack_mode", default="brute_force",
                        choices=["brute_force", "flood", "adaptive", "multi_stage"])
    parser.add_argument("--intensity", default="high",
                        choices=["low", "medium", "high"])
    parser.add_argument("--episodes", default=3, type=int,
                        help="Number of episodes to visualise")
    parser.add_argument("--fps", default=8, type=int,
                        help="Rendering speed (steps per second)")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic (non-deterministic) actions")
    args = parser.parse_args()

    print(f"Loading model from {args.model_path} ({args.algorithm.upper()})...")
    model = _load_model(args.model_path, args.algorithm)

    env = NetworkSecurityEnv(
        attack_mode=args.attack_mode,
        intensity=args.intensity,
    )

    renderer = AIRSRenderer()
    renderer.set_fps(args.fps)
    renderer.set_scenario(args.attack_mode, args.intensity)

    print(f"Running {args.episodes} episodes | {args.attack_mode} | {args.intensity}")
    print("Controls: SPACE=pause  UP/DOWN=speed  R=restart  ESC=quit")

    for ep in range(1, args.episodes + 1):
        print(f"\n── Episode {ep}/{args.episodes} ──")
        renderer._history.clear()
        renderer._particles.clear()
        total = run_episode(env, model, renderer, deterministic=not args.stochastic)
        if total is None:
            print("Quit by user.")
            break
        print(f"  Total reward: {total:+.1f}")

    renderer.close()
    env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
