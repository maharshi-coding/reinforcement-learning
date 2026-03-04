"""
System Monitor for AIRS.

Collects real-time CPU / memory usage via psutil and computes a
scalar threat-level score from the full observation vector.
"""

import psutil
import numpy as np


class SystemMonitor:
    """Reads live system metrics and computes a threat-level score."""

    # Weights used when computing the scalar threat level (must sum to 1)
    _THREAT_WEIGHTS = np.array([0.30, 0.30, 0.15, 0.10, 0.15], dtype=np.float32)

    def get_system_metrics(self) -> dict:
        """Return current CPU and memory utilisation (0–1 normalised)."""
        cpu = psutil.cpu_percent(interval=None) / 100.0
        mem = psutil.virtual_memory().percent / 100.0
        return {"cpu_usage": float(cpu), "memory_usage": float(mem)}

    def compute_threat_level(
        self,
        traffic_rate: float,
        failed_logins: float,
        cpu_usage: float,
        memory_usage: float,
        traffic_max: float = 1000.0,
        logins_max: float = 300.0,
    ) -> float:
        """Compute a normalised threat-level score in [0, 1].

        The score is a weighted sum of five normalised indicators:
            1. traffic_rate   / traffic_max
            2. failed_logins  / logins_max
            3. cpu_usage      (already 0–1)
            4. memory_usage   (already 0–1)
            5. combination spike indicator

        Args:
            traffic_rate:  Raw packet/request rate.
            failed_logins: Number of failed login attempts.
            cpu_usage:     CPU utilisation in [0, 1].
            memory_usage:  Memory utilisation in [0, 1].
            traffic_max:   Expected maximum traffic rate (for normalisation).
            logins_max:    Expected maximum failed-login count (for normalisation).

        Returns:
            Threat level in [0, 1].
        """
        t = min(traffic_rate / traffic_max, 1.0)
        f = min(failed_logins / logins_max, 1.0)
        c = float(np.clip(cpu_usage, 0.0, 1.0))
        m = float(np.clip(memory_usage, 0.0, 1.0))
        spike = min(t * f + 0.5 * c, 1.0)  # joint indicator

        features = np.array([t, f, c, m, spike], dtype=np.float32)
        threat = float(np.dot(self._THREAT_WEIGHTS, features))
        return float(np.clip(threat, 0.0, 1.0))
