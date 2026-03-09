"""
AIRS Real-Time Inference Engine.

Runs a continuous loop: collect live metrics → build observation →
agent predicts action → responder executes → log & repeat.

Usage:
    python -m airs.realtime.engine --algorithm ppo --model models/ppo_agent
    python -m airs.realtime.engine --dry-run        # safe mode (default)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from airs.monitoring.monitor import SystemMonitor
from airs.realtime import RealTimeCollector
from airs.realtime.responder import RealTimeResponder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("airs.realtime.engine")


class RealTimeEngine:
    """Main real-time inference loop.

    Collect → Observe → Predict → Act → Log

    Args:
        algorithm:      Which trained model to load (dqn/ppo/a2c).
        model_path:     Path to the saved SB3 model.
        poll_interval:  Seconds between each observation cycle.
        dry_run:        If True, log actions but don't execute them.
        log_path:       Path to write JSON event log.
        threat_threshold: Minimum threat level to allow non-noop actions.
    """

    def __init__(
        self,
        algorithm: str = "ppo",
        model_path: str = "models/ppo_agent",
        poll_interval: float = 1.0,
        dry_run: bool = True,
        log_path: str = "results/realtime_log.jsonl",
        threat_threshold: float = 0.2,
    ):
        self.algorithm = algorithm
        self.poll_interval = poll_interval
        self.dry_run = dry_run
        self.log_path = log_path
        self.threat_threshold = threat_threshold
        self._running = False

        # Components
        self.collector = RealTimeCollector(poll_interval=poll_interval)
        self.monitor = SystemMonitor()
        self.responder = RealTimeResponder(dry_run=dry_run)

        # Load trained model
        logger.info("Loading %s model from %s ...", algorithm.upper(), model_path)
        self._model = self._load_model(algorithm, model_path)
        logger.info("Model loaded successfully")

        # State tracking
        self._last_action: int = 0
        self._step_count: int = 0
        self._total_threat: float = 0.0
        self._action_counts: dict[int, int] = {0: 0, 1: 0, 2: 0, 3: 0}

        # Ensure log directory exists
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    def _load_model(self, algorithm: str, model_path: str):
        """Load a trained SB3 model."""
        from stable_baselines3 import A2C, DQN, PPO

        algo_map = {"dqn": DQN, "ppo": PPO, "a2c": A2C}
        algo_cls = algo_map.get(algorithm)
        if algo_cls is None:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        # Try with and without .zip
        for p in [model_path, model_path + ".zip"]:
            if os.path.exists(p):
                return algo_cls.load(p)
        raise FileNotFoundError(f"Model not found at {model_path}[.zip]")

    def _build_observation(self, normalised: dict[str, float], threat: float) -> np.ndarray:
        """Build the 6-dim observation vector matching the training environment."""
        obs = np.array([
            normalised["traffic_rate"],
            normalised["failed_logins"],
            normalised["cpu_usage"],
            normalised["memory_usage"],
            threat,
            self._last_action / 3.0,  # normalised last action
        ], dtype=np.float32)
        return obs

    def _predict(self, obs: np.ndarray) -> int:
        """Get agent's action from observation."""
        action, _ = self._model.predict(obs, deterministic=True)
        return int(action)

    def _log_event(self, event: dict) -> None:
        """Append event to JSONL log file."""
        with open(self.log_path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def run(self) -> None:
        """Start the real-time inference loop (blocks until interrupted)."""
        self._running = True

        # Handle graceful shutdown
        def _shutdown(sig, frame):
            logger.info("Shutting down (signal %s)...", sig)
            self._running = False

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)

        mode = "DRY-RUN" if self.dry_run else "LIVE"
        logger.info("=" * 60)
        logger.info(" AIRS Real-Time Engine — %s mode", mode)
        logger.info(" Algorithm: %s | Poll: %.1fs | Threshold: %.2f",
                     self.algorithm.upper(), self.poll_interval, self.threat_threshold)
        logger.info(" Log: %s", self.log_path)
        logger.info("=" * 60)

        while self._running:
            try:
                self._step()
                time.sleep(self.poll_interval)
            except Exception as e:
                logger.error("Error in step %d: %s", self._step_count, e)
                time.sleep(self.poll_interval)

        self._shutdown_summary()

    def _step(self) -> None:
        """Execute one collect → predict → act cycle."""
        self._step_count += 1

        # 1. Collect live metrics
        snap = self.collector.collect()
        normalised = self.collector.normalise(snap)

        # 2. Compute threat level
        threat = self.monitor.compute_threat_level(
            traffic_rate=snap.traffic_rate,
            failed_logins=snap.failed_logins,
            cpu_usage=snap.cpu_usage,
            memory_usage=snap.memory_usage,
        )
        self._total_threat += threat

        # 3. Build observation and predict
        obs = self._build_observation(normalised, threat)
        action = self._predict(obs)

        # Safety: override to no-op if threat is below threshold
        original_action = action
        if threat < self.threat_threshold and action != 0:
            action = 0

        # 4. Execute response
        record = self.responder.act(action)

        # 5. Update state
        self._last_action = action
        self._action_counts[action] = self._action_counts.get(action, 0) + 1

        # 6. Log
        event = {
            "step": self._step_count,
            "timestamp": snap.timestamp,
            "metrics": {
                "traffic_rate": round(snap.traffic_rate, 1),
                "failed_logins": round(snap.failed_logins, 1),
                "cpu": round(snap.cpu_usage, 3),
                "memory": round(snap.memory_usage, 3),
                "connections": snap.connections,
            },
            "threat_level": round(threat, 4),
            "agent_action": original_action,
            "final_action": action,
            "action_name": record.action_name,
            "executed": record.executed,
            "detail": record.detail,
        }
        self._log_event(event)

        # Console output
        threat_bar = "█" * int(threat * 20) + "░" * (20 - int(threat * 20))
        action_sym = {0: "👁️ ", 1: "🛑", 2: "⚡", 3: "🔒"}
        sym = action_sym.get(action, "?")

        overridden = " (overridden→noop)" if original_action != action else ""
        logger.info(
            "Step %4d | Threat [%s] %.3f | %s %s%s | CPU %.0f%% MEM %.0f%% | Pkts/s %.0f",
            self._step_count, threat_bar, threat,
            sym, record.action_name, overridden,
            snap.cpu_usage * 100, snap.memory_usage * 100,
            snap.traffic_rate,
        )

    def _shutdown_summary(self) -> None:
        """Print summary on shutdown."""
        logger.info("=" * 60)
        logger.info(" Session Summary")
        logger.info("  Steps: %d", self._step_count)
        logger.info("  Avg Threat: %.3f",
                     self._total_threat / max(self._step_count, 1))
        logger.info("  Actions: %s", {
            RealTimeResponder.ACTION_NAMES[k]: v
            for k, v in self._action_counts.items()
        })
        logger.info("  Log saved to: %s", self.log_path)
        logger.info("=" * 60)


def parse_args():
    p = argparse.ArgumentParser(description="AIRS Real-Time Inference Engine")
    p.add_argument("--algorithm", default="ppo", choices=["dqn", "ppo", "a2c"],
                   help="Which trained model to use")
    p.add_argument("--model", default=None,
                   help="Model path (default: models/<algorithm>_agent)")
    p.add_argument("--interval", type=float, default=1.0,
                   help="Seconds between observation cycles")
    p.add_argument("--threshold", type=float, default=0.2,
                   help="Minimum threat level for non-noop actions")
    p.add_argument("--log", default="results/realtime_log.jsonl",
                   help="Path for event log output")
    p.add_argument("--live", action="store_true",
                   help="LIVE mode: actually execute iptables commands (requires root)")
    return p.parse_args()


def main():
    args = parse_args()
    model_path = args.model or f"models/{args.algorithm}_agent"

    engine = RealTimeEngine(
        algorithm=args.algorithm,
        model_path=model_path,
        poll_interval=args.interval,
        dry_run=not args.live,
        log_path=args.log,
        threat_threshold=args.threshold,
    )
    engine.run()


if __name__ == "__main__":
    main()
