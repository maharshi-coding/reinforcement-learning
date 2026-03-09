#!/usr/bin/env python3
"""
train_with_visualizer.py – Run RL training with a live pygame visualizer.

The visualizer runs in the main thread (required by pygame).
Training runs in a background thread, pushing step/episode data
through a thread-safe queue that the visualizer consumes.

Usage:
  python scripts/train_with_visualizer.py                        # Train PPO with visualizer
  python scripts/train_with_visualizer.py --algorithm dqn        # Train DQN
  python scripts/train_with_visualizer.py --algorithm a2c        # Train A2C
  python scripts/train_with_visualizer.py --timesteps 100000     # Shorter run
  python scripts/train_with_visualizer.py --curriculum           # Curriculum training

The visualizer window opens immediately. Training begins in the background.
Press ESC or close the window to stop both the visualizer and training.
"""

import argparse
import os
import sys
import threading

# Suppress ALSA warnings on WSL (no audio hardware)
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from airs.agent.rl_agent import AIRSAgent
from airs.config import load_config
from airs.environment.multi_scenario_env import MultiScenarioEnv
from src.visualization.training_state import TrainingState, StepData, EpisodeData
from src.visualization.training_visualizer import TrainingVisualizer


# ── Visualizer Callback ─────────────────────────────────────────────

class VisualizerCallback(BaseCallback):
    """SB3 callback that pushes every step's data to the TrainingState bridge.

    Extracts info from the training locals dict and pushes StepData /
    EpisodeData into the thread-safe queues.
    """

    def __init__(self, state: TrainingState, verbose: int = 0):
        super().__init__(verbose)
        self._state = state
        self._ep_reward = 0.0
        self._ep_steps = 0
        self._ep_count = 0
        self._current_mode = ""
        self._current_intensity = ""

    def _on_step(self) -> bool:
        # Check if visualizer requested stop
        if self._state.should_stop():
            return False  # stops training

        # Extract step data from SB3 locals
        infos = self.locals.get("infos", [{}])
        info = infos[0] if infos else {}
        rewards = self.locals.get("rewards", [0.0])
        reward = float(rewards[0]) if rewards is not None else 0.0

        actions = self.locals.get("actions", [0])
        action = int(actions[0]) if actions is not None else 0

        # Observation: the raw obs for the first env
        new_obs = self.locals.get("new_obs", None)
        obs = new_obs[0] if new_obs is not None and len(new_obs) > 0 else np.zeros(6)

        self._ep_reward += reward
        self._ep_steps += 1

        threat = info.get("threat_level", float(obs[4]) if len(obs) > 4 else 0.0)

        step = StepData(
            timestep=self.num_timesteps,
            episode=self._ep_count,
            step_in_episode=self._ep_steps,
            action=action,
            reward=reward,
            episode_reward=self._ep_reward,
            threat_level=threat,
            traffic=float(obs[0]) if len(obs) > 0 else 0.0,
            failed_logins=float(obs[1]) if len(obs) > 1 else 0.0,
            cpu=float(obs[2]) if len(obs) > 2 else 0.0,
            memory=float(obs[3]) if len(obs) > 3 else 0.0,
            service_cost=info.get("service_cost", 0.0),
            action_name=info.get("action_name", ""),
            phase=info.get("phase", ""),
            done=False,
            attack_mode=info.get("attack_mode", self._current_mode),
            intensity=info.get("intensity", self._current_intensity),
        )

        dones = self.locals.get("dones", [False])
        done = bool(dones[0]) if dones is not None else False

        if done:
            step.done = True
            self._state.push_episode(EpisodeData(
                episode=self._ep_count,
                total_reward=self._ep_reward,
                steps=self._ep_steps,
                attack_mode=self._current_mode,
                intensity=self._current_intensity,
            ))
            self._ep_count += 1
            self._ep_reward = 0.0
            self._ep_steps = 0

        self._state.push_step(step)
        return True


# ── Environment Factories ────────────────────────────────────────────

def _make_env(env_kwargs, seed=42):
    def _init():
        env = MultiScenarioEnv(env_kwargs=env_kwargs)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


def _make_curriculum_env(intensities, env_kwargs, seed=42):
    def _init():
        env = MultiScenarioEnv(intensities=intensities, env_kwargs=env_kwargs)
        env = Monitor(env)
        env.reset(seed=seed)
        return env
    return _init


# ── Training Thread ──────────────────────────────────────────────────

def _run_training(algorithm: str, timesteps: int, cfg: dict, seed: int,
                  state: TrainingState, curriculum: bool):
    """Runs in a background thread. Pushes updates to TrainingState."""
    try:
        env_cfg = cfg.get("environment", {})
        reward_cfg = cfg.get("reward", {})
        paths_cfg = cfg.get("paths", {})

        model_path = os.path.join(paths_cfg.get("model_dir", "models"), f"{algorithm}_agent")
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)

        env_kwargs = {
            "noisy_observations": env_cfg.get("noisy_observations", False),
            "noise_std": env_cfg.get("noise_std", 0.05),
            "temporal_window": env_cfg.get("temporal_window", 1),
            "reward_cfg": reward_cfg,
        }

        # Build agent — resume from existing model if available
        agent = AIRSAgent(
            algorithm=algorithm,
            multi_scenario=True,
            seed=seed,
            env_kwargs=env_kwargs,
            model_path=model_path,
        )
        model = agent._model

        resumed = os.path.exists(model_path) or os.path.exists(model_path + ".zip")
        if resumed:
            print(f"[AIRS] Resuming from existing model: {model_path}")
        else:
            print("[AIRS] No existing model found — starting fresh.")

        # Create visualizer callback
        vis_cb = VisualizerCallback(state)

        if curriculum:
            _train_curriculum(model, model_path, timesteps, env_kwargs, seed, state, vis_cb)
        else:
            _train_standard(model, model_path, timesteps, env_kwargs, seed, state, vis_cb)

        # Save model
        model.save(model_path)
        print(f"\n[AIRS] Training complete. Model saved: {model_path}")

    except Exception as e:
        print(f"\n[AIRS] Training error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        state.signal_done()


def _train_standard(model, model_path, timesteps, env_kwargs, seed, state, vis_cb):
    """Standard multi-scenario training."""
    env = DummyVecEnv([_make_env(env_kwargs, seed)])
    model.set_env(env)

    eval_env = DummyVecEnv([_make_env(env_kwargs, seed + 1000)])
    stop_cb = StopTrainingOnNoModelImprovement(max_no_improvement_evals=20, verbose=0)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.dirname(model_path) or "models",
        eval_freq=5000,
        n_eval_episodes=10,
        callback_after_eval=stop_cb,
        verbose=0,
    )

    print(f"[AIRS] Starting standard training ({timesteps:,} timesteps)...")
    model.learn(
        total_timesteps=timesteps,
        callback=CallbackList([vis_cb, eval_cb]),
    )


def _train_curriculum(model, model_path, timesteps, env_kwargs, seed, state, vis_cb):
    """Curriculum training: low → medium → high → mixed."""
    stage_steps = timesteps // 4
    stages = [
        ("Stage 1: LOW", ["low"], stage_steps),
        ("Stage 2: MEDIUM", ["medium"], stage_steps),
        ("Stage 3: HIGH", ["high"], stage_steps),
        ("Stage 4: MIXED", ["low", "medium", "high"], stage_steps),
    ]

    print(f"[AIRS] Starting curriculum training ({timesteps:,} timesteps, {len(stages)} stages)...")

    for name, intensities, steps in stages:
        if state.should_stop():
            break

        print(f"\n── {name} ({steps:,} steps) ──")
        env = DummyVecEnv([_make_curriculum_env(intensities, env_kwargs, seed)])
        model.set_env(env)

        eval_env = DummyVecEnv([_make_env(env_kwargs, seed + 1000)])
        stop_cb = StopTrainingOnNoModelImprovement(max_no_improvement_evals=20, verbose=0)
        eval_cb = EvalCallback(
            eval_env,
            best_model_save_path=os.path.dirname(model_path) or "models",
            eval_freq=5000,
            n_eval_episodes=10,
            callback_after_eval=stop_cb,
            verbose=0,
        )

        model.learn(
            total_timesteps=steps,
            callback=CallbackList([vis_cb, eval_cb]),
            reset_num_timesteps=False,
        )


# ── Argument Parsing ─────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train an AIRS RL agent with a live pygame visualizer"
    )
    p.add_argument("--algorithm", default="ppo", choices=["dqn", "ppo", "a2c"],
                   help="RL algorithm to train (default: ppo)")
    p.add_argument("--timesteps", default=None, type=int,
                   help="Total training timesteps (default: from config)")
    p.add_argument("--curriculum", action="store_true",
                   help="Use curriculum training (low→medium→high→mixed)")
    p.add_argument("--config", default="configs/default.yaml",
                   help="Config file path")
    p.add_argument("--seed", default=42, type=int, help="Random seed")
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = load_config(args.config)
    train_cfg = cfg.get("training", {})
    timesteps = args.timesteps or train_cfg.get("total_timesteps", 100_000)

    # Set up shared state
    state = TrainingState()
    state.algorithm = args.algorithm
    state.total_timesteps = timesteps

    print(f"{'='*60}")
    print("  AIRS Training Visualizer")
    print(f"  Algorithm:  {args.algorithm.upper()}")
    print(f"  Timesteps:  {timesteps:,}")
    print(f"  Curriculum: {'Yes' if args.curriculum else 'No'}")
    print(f"{'='*60}")
    print()
    print("Opening visualizer window...")
    print("Training starts automatically in background.")
    print("Press ESC or close window to stop.\n")

    # Start training in background thread
    train_thread = threading.Thread(
        target=_run_training,
        args=(args.algorithm, timesteps, cfg, args.seed, state, args.curriculum),
        daemon=True,
    )
    train_thread.start()

    # Run visualizer in main thread (required by pygame)
    viz = TrainingVisualizer(state)
    viz.run()

    # Wait for training thread to finish (it will stop via should_stop)
    train_thread.join(timeout=5)
    print("\nDone.")


if __name__ == "__main__":
    main()
