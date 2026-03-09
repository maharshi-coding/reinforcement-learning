"""
Real-time network data collector for AIRS.

Collects actual system & network metrics using psutil, replacing the
simulated AttackSimulator for live deployment.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import psutil


@dataclass
class NetworkSnapshot:
    """A single point-in-time measurement of system + network state."""

    timestamp: float
    traffic_rate: float       # packets/sec (normalised to 0–1 later)
    failed_logins: float      # approximated from connection failures
    cpu_usage: float          # 0–1
    memory_usage: float       # 0–1
    connections: int          # active TCP connections
    bytes_sent: int
    bytes_recv: int
    packets_sent: int
    packets_recv: int


class RealTimeCollector:
    """Collects live system and network metrics.

    Uses psutil to read actual CPU, memory, network counters, and
    connection states.  Maintains a rolling window for computing
    rates (packets/sec, connection-failure rates).

    Args:
        poll_interval: seconds between samples (default 1.0)
        window_size:   number of past snapshots to retain
        traffic_max:   expected max packets/sec for normalisation
        logins_max:    expected max "failed connections"/sec for normalisation
    """

    def __init__(
        self,
        poll_interval: float = 1.0,
        window_size: int = 60,
        traffic_max: float = 10000.0,
        logins_max: float = 500.0,
    ):
        self.poll_interval = poll_interval
        self.window_size = window_size
        self.traffic_max = traffic_max
        self.logins_max = logins_max

        self._history: deque[NetworkSnapshot] = deque(maxlen=window_size)
        self._prev_net = psutil.net_io_counters()
        self._prev_time = time.monotonic()

        # Take an initial sample so rates are available immediately
        psutil.cpu_percent(interval=None)  # prime the CPU counter

    def collect(self) -> NetworkSnapshot:
        """Take a single measurement and return a NetworkSnapshot."""
        now = time.monotonic()
        dt = max(now - self._prev_time, 0.001)

        # Network I/O counters
        net = psutil.net_io_counters()
        packets_in = net.packets_recv - self._prev_net.packets_recv
        packets_out = net.packets_sent - self._prev_net.packets_sent
        bytes_in = net.bytes_recv - self._prev_net.bytes_recv
        bytes_out = net.bytes_sent - self._prev_net.bytes_sent
        self._prev_net = net
        self._prev_time = now

        traffic_rate = (packets_in + packets_out) / dt

        # Approximate "failed logins" from connection failures / resets
        failed_logins = self._estimate_failed_connections()

        cpu = psutil.cpu_percent(interval=None) / 100.0
        mem = psutil.virtual_memory().percent / 100.0
        connections = len(psutil.net_connections(kind="tcp"))

        snap = NetworkSnapshot(
            timestamp=time.time(),
            traffic_rate=traffic_rate,
            failed_logins=failed_logins,
            cpu_usage=cpu,
            memory_usage=mem,
            connections=connections,
            bytes_sent=bytes_out,
            bytes_recv=bytes_in,
            packets_sent=packets_out,
            packets_recv=packets_in,
        )
        self._history.append(snap)
        return snap

    def _estimate_failed_connections(self) -> float:
        """Count TCP connections in suspicious states as a proxy for attacks."""
        suspicious_states = {"SYN_SENT", "SYN_RECV", "CLOSE_WAIT", "TIME_WAIT"}
        count = 0
        try:
            for conn in psutil.net_connections(kind="tcp"):
                if conn.status in suspicious_states:
                    count += 1
        except (psutil.AccessDenied, PermissionError):
            # Fallback: use net_io error counters
            net = psutil.net_io_counters()
            count = getattr(net, "errin", 0) + getattr(net, "errout", 0)
        return float(count)

    def normalise(self, snap: NetworkSnapshot) -> dict[str, float]:
        """Normalise raw snapshot to 0–1 range matching the agent's input."""
        return {
            "traffic_rate": min(snap.traffic_rate / self.traffic_max, 1.0),
            "failed_logins": min(snap.failed_logins / self.logins_max, 1.0),
            "cpu_usage": snap.cpu_usage,
            "memory_usage": snap.memory_usage,
        }

    @property
    def history(self) -> list[NetworkSnapshot]:
        return list(self._history)
