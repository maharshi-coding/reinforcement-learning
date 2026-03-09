"""
Thread-safe bridge between the SB3 training loop and the pygame visualizer.

The training callback pushes step/episode data into a queue;
the visualizer pops items each frame. No locks needed — queue.Queue
is already thread-safe.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StepData:
    """One environment step worth of data."""

    timestep: int = 0
    episode: int = 0
    step_in_episode: int = 0
    action: int = 0
    reward: float = 0.0
    episode_reward: float = 0.0
    threat_level: float = 0.0
    traffic: float = 0.0
    cpu: float = 0.0
    memory: float = 0.0
    failed_logins: float = 0.0
    service_cost: float = 0.0
    action_name: str = "no_op"
    phase: str = ""
    done: bool = False
    attack_mode: str = ""
    intensity: str = ""


@dataclass
class EpisodeData:
    """Summary of a completed episode."""

    episode: int = 0
    total_reward: float = 0.0
    steps: int = 0
    attack_mode: str = ""
    intensity: str = ""


class TrainingState:
    """Thread-safe state bridge for training visualizer.

    Usage
    -----
    # In training callback:
        state.push_step(StepData(...))
        state.push_episode(EpisodeData(...))

    # In visualizer main loop:
        steps = state.get_pending_steps()
        episodes = state.get_pending_episodes()
    """

    def __init__(self, maxsize: int = 5000):
        self._step_queue: queue.Queue[StepData] = queue.Queue(maxsize=maxsize)
        self._episode_queue: queue.Queue[EpisodeData] = queue.Queue(maxsize=1000)
        self._stop_event = threading.Event()
        self.total_timesteps: int = 0
        self.algorithm: str = ""
        self.training_done = threading.Event()

    # ── Producer API (training thread) ───────────────────────────────

    def push_step(self, data: StepData) -> None:
        try:
            self._step_queue.put_nowait(data)
        except queue.Full:
            # Drop oldest to keep flowing
            try:
                self._step_queue.get_nowait()
            except queue.Empty:
                pass
            self._step_queue.put_nowait(data)

    def push_episode(self, data: EpisodeData) -> None:
        try:
            self._episode_queue.put_nowait(data)
        except queue.Full:
            try:
                self._episode_queue.get_nowait()
            except queue.Empty:
                pass
            self._episode_queue.put_nowait(data)

    def signal_done(self) -> None:
        self.training_done.set()

    def should_stop(self) -> bool:
        return self._stop_event.is_set()

    # ── Consumer API (visualizer thread / main thread) ───────────────

    def get_pending_steps(self, max_items: int = 50) -> list[StepData]:
        items = []
        for _ in range(max_items):
            try:
                items.append(self._step_queue.get_nowait())
            except queue.Empty:
                break
        return items

    def get_pending_episodes(self, max_items: int = 20) -> list[EpisodeData]:
        items = []
        for _ in range(max_items):
            try:
                items.append(self._episode_queue.get_nowait())
            except queue.Empty:
                break
        return items

    def request_stop(self) -> None:
        """Called by the visualizer to ask training to stop."""
        self._stop_event.set()

    def is_training_done(self) -> bool:
        return self.training_done.is_set()
