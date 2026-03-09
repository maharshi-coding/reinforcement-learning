"""
Real-time response executor for AIRS.

Translates the agent's discrete actions into real firewall / system commands.
Supports a dry-run mode (default) for safe testing without actually modifying
iptables rules.
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass

logger = logging.getLogger("airs.realtime.responder")


@dataclass
class ResponseRecord:
    """Record of a response action taken."""

    timestamp: float
    action_id: int
    action_name: str
    target: str          # e.g. IP address or "system"
    executed: bool       # False if dry-run
    detail: str          # command or description


class RealTimeResponder:
    """Executes defensive responses on the real system.

    Actions:
        0 - no_op:          Do nothing (observe only)
        1 - block_ip:       Block top suspicious IPs via iptables
        2 - rate_limit:     Apply rate limiting via iptables
        3 - isolate_service: Reject all incoming traffic (emergency)

    Args:
        dry_run:  If True (default), log the commands but don't execute them.
        cooldown: Minimum seconds between consecutive non-noop actions.
    """

    ACTION_NAMES = {0: "no_op", 1: "block_ip", 2: "rate_limit", 3: "isolate_service"}

    def __init__(self, dry_run: bool = True, cooldown: float = 5.0):
        self.dry_run = dry_run
        self.cooldown = cooldown
        self._last_action_time: float = 0
        self._history: list[ResponseRecord] = []
        self._blocked_ips: set[str] = set()

        if dry_run:
            logger.info("RealTimeResponder running in DRY-RUN mode (no actual changes)")
        else:
            logger.warning("RealTimeResponder running in LIVE mode — will modify iptables!")

    def act(self, action_id: int, suspicious_ips: list[str] | None = None) -> ResponseRecord:
        """Execute the defensive action.

        Args:
            action_id:      0–3 matching the agent's action space.
            suspicious_ips: Optional list of IPs to target (for block/rate-limit).
        """
        now = time.time()
        name = self.ACTION_NAMES.get(action_id, "unknown")

        # Cooldown check — skip non-noop if too soon
        if action_id != 0 and (now - self._last_action_time) < self.cooldown:
            record = ResponseRecord(
                timestamp=now, action_id=action_id, action_name=name,
                target="", executed=False, detail="Skipped (cooldown)",
            )
            self._history.append(record)
            return record

        if action_id == 0:
            record = self._do_noop(now)
        elif action_id == 1:
            record = self._do_block(now, suspicious_ips or [])
        elif action_id == 2:
            record = self._do_rate_limit(now)
        elif action_id == 3:
            record = self._do_isolate(now)
        else:
            record = ResponseRecord(
                timestamp=now, action_id=action_id, action_name="unknown",
                target="", executed=False, detail=f"Unknown action {action_id}",
            )

        if action_id != 0:
            self._last_action_time = now

        self._history.append(record)
        return record

    # ── Action implementations ────────────────────────────────────

    def _do_noop(self, ts: float) -> ResponseRecord:
        return ResponseRecord(
            timestamp=ts, action_id=0, action_name="no_op",
            target="system", executed=True, detail="Observing — no action taken",
        )

    def _do_block(self, ts: float, ips: list[str]) -> ResponseRecord:
        if not ips:
            return ResponseRecord(
                timestamp=ts, action_id=1, action_name="block_ip",
                target="none", executed=False, detail="No suspicious IPs to block",
            )

        cmds = []
        for ip in ips[:5]:  # cap at 5 IPs per action
            if ip in self._blocked_ips:
                continue
            cmd = ["iptables", "-A", "INPUT", "-s", ip, "-j", "DROP"]
            cmds.append((ip, cmd))

        detail_parts = []
        for ip, cmd in cmds:
            cmd_str = " ".join(cmd)
            if self.dry_run:
                logger.info("[DRY-RUN] Would execute: %s", cmd_str)
                detail_parts.append(f"[DRY-RUN] {cmd_str}")
            else:
                self._run_cmd(cmd)
                self._blocked_ips.add(ip)
                detail_parts.append(f"[EXECUTED] {cmd_str}")

        return ResponseRecord(
            timestamp=ts, action_id=1, action_name="block_ip",
            target=",".join(ip for ip, _ in cmds),
            executed=not self.dry_run,
            detail="; ".join(detail_parts) if detail_parts else "No new IPs to block",
        )

    def _do_rate_limit(self, ts: float) -> ResponseRecord:
        cmd = [
            "iptables", "-A", "INPUT", "-p", "tcp",
            "-m", "limit", "--limit", "25/minute", "--limit-burst", "100",
            "-j", "ACCEPT",
        ]
        cmd_str = " ".join(cmd)

        if self.dry_run:
            logger.info("[DRY-RUN] Would execute: %s", cmd_str)
            detail = f"[DRY-RUN] {cmd_str}"
        else:
            self._run_cmd(cmd)
            detail = f"[EXECUTED] {cmd_str}"

        return ResponseRecord(
            timestamp=ts, action_id=2, action_name="rate_limit",
            target="system", executed=not self.dry_run, detail=detail,
        )

    def _do_isolate(self, ts: float) -> ResponseRecord:
        cmd = ["iptables", "-P", "INPUT", "DROP"]
        cmd_str = " ".join(cmd)

        if self.dry_run:
            logger.info("[DRY-RUN] Would execute: %s", cmd_str)
            detail = f"[DRY-RUN] {cmd_str}"
        else:
            self._run_cmd(cmd)
            detail = f"[EXECUTED] {cmd_str}"

        return ResponseRecord(
            timestamp=ts, action_id=3, action_name="isolate_service",
            target="system", executed=not self.dry_run, detail=detail,
        )

    def _run_cmd(self, cmd: list[str]) -> None:
        """Execute a shell command. Requires root for iptables."""
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=5)
        except subprocess.CalledProcessError as e:
            logger.error("Command failed: %s — %s", " ".join(cmd), e.stderr)
        except FileNotFoundError:
            logger.error("Command not found: %s", cmd[0])

    def clear_blocks(self) -> None:
        """Remove all AIRS-added iptables rules (flush INPUT chain)."""
        cmd = ["iptables", "-F", "INPUT"]
        if self.dry_run:
            logger.info("[DRY-RUN] Would flush iptables INPUT chain")
        else:
            self._run_cmd(cmd)
            self._blocked_ips.clear()
            logger.info("Flushed all iptables INPUT rules")

    @property
    def history(self) -> list[ResponseRecord]:
        return list(self._history)
