"""
Tests for AIRS – covers environment, attack simulator, monitor, response engine,
baselines, and evaluation framework.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from airs.environment.attack_simulator import AttackSimulator
from airs.environment.network_env import NetworkSecurityEnv
from airs.monitoring.monitor import SystemMonitor
from airs.response.response_engine import ResponseEngine
from airs.agent.baselines import AlwaysNoopPolicy, RandomPolicy, RuleBasedThresholdPolicy, get_baseline


# ---------------------------------------------------------------------------
# AttackSimulator
# ---------------------------------------------------------------------------

class TestAttackSimulator:
    def test_valid_modes(self):
        for mode in AttackSimulator.MODES:
            sim = AttackSimulator(mode=mode)
            assert sim.mode == mode

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            AttackSimulator(mode="unknown_mode")

    def test_step_returns_expected_keys(self):
        sim = AttackSimulator(mode="brute_force", intensity="medium")
        result = sim.step(last_action=0)
        assert "traffic_rate" in result
        assert "failed_logins" in result
        assert "is_attacking" in result
        assert "phase" in result

    def test_step_values_non_negative(self):
        for mode in AttackSimulator.MODES:
            sim = AttackSimulator(mode=mode, intensity="high")
            for _ in range(20):
                result = sim.step(last_action=0)
                assert result["traffic_rate"] >= 0.0
                assert result["failed_logins"] >= 0.0

    def test_block_action_reduces_logins(self):
        """Blocking should reduce failed logins (averaged over multiple steps)."""
        np.random.seed(42)
        sim_base = AttackSimulator(mode="brute_force", intensity="high")
        sim_block = AttackSimulator(mode="brute_force", intensity="high")

        logins_base = [sim_base.step(0)["failed_logins"] for _ in range(50)]
        logins_blocked = [sim_block.step(1)["failed_logins"] for _ in range(50)]
        assert np.mean(logins_blocked) < np.mean(logins_base)

    def test_reset_clears_state(self):
        sim = AttackSimulator(mode="adaptive")
        for _ in range(50):
            sim.step(0)
        sim.reset()
        assert sim._step_count == 0

    def test_adaptive_mode_switches(self):
        """Adaptive attacker should switch strategy."""
        sim = AttackSimulator(mode="adaptive", intensity="medium")
        for _ in range(200):
            sim.step(last_action=1)
        assert sim._step_count == 200

    def test_multi_stage_mode(self):
        """Multi-stage attack should cycle through phases."""
        sim = AttackSimulator(mode="multi_stage", intensity="medium")
        phases_seen = set()
        for _ in range(200):
            result = sim.step(last_action=0)
            phases_seen.add(result["phase"])
        assert len(phases_seen) >= 2  # should see at least 2 phases

    def test_defender_history_tracked(self):
        sim = AttackSimulator(mode="adaptive", intensity="medium", defender_history_len=5)
        for _ in range(10):
            sim.step(last_action=1)
        assert len(sim._defender_history) == 5


# ---------------------------------------------------------------------------
# SystemMonitor
# ---------------------------------------------------------------------------

class TestSystemMonitor:
    def setup_method(self):
        self.monitor = SystemMonitor()

    def test_get_system_metrics_keys(self):
        metrics = self.monitor.get_system_metrics()
        assert "cpu_usage" in metrics
        assert "memory_usage" in metrics

    def test_get_system_metrics_range(self):
        metrics = self.monitor.get_system_metrics()
        assert 0.0 <= metrics["cpu_usage"] <= 1.0
        assert 0.0 <= metrics["memory_usage"] <= 1.0

    def test_threat_level_zero_attack(self):
        threat = self.monitor.compute_threat_level(
            traffic_rate=0.0, failed_logins=0.0, cpu_usage=0.0, memory_usage=0.0
        )
        assert threat == pytest.approx(0.0, abs=1e-6)

    def test_threat_level_max_attack(self):
        threat = self.monitor.compute_threat_level(
            traffic_rate=1000.0, failed_logins=300.0, cpu_usage=1.0, memory_usage=1.0
        )
        assert threat == pytest.approx(1.0, abs=1e-6)

    def test_threat_level_bounded(self):
        """Threat level should always be in [0, 1]."""
        for _ in range(100):
            t = np.random.uniform(0, 2000)
            f = np.random.uniform(0, 600)
            c = np.random.uniform(0, 2)
            m = np.random.uniform(0, 2)
            threat = self.monitor.compute_threat_level(t, f, c, m)
            assert 0.0 <= threat <= 1.0

    def test_threat_increases_with_attack_intensity(self):
        low = self.monitor.compute_threat_level(10, 5, 0.1, 0.2)
        high = self.monitor.compute_threat_level(900, 280, 0.9, 0.8)
        assert high > low


# ---------------------------------------------------------------------------
# ResponseEngine
# ---------------------------------------------------------------------------

class TestResponseEngine:
    def setup_method(self):
        self.engine = ResponseEngine()

    def test_num_actions(self):
        assert self.engine.num_actions == 4

    def test_action_names(self):
        assert self.engine.get_action_name(0) == "no_op"
        assert self.engine.get_action_name(1) == "block_ip"
        assert self.engine.get_action_name(2) == "rate_limit"
        assert self.engine.get_action_name(3) == "isolate_service"

    def test_invalid_action_raises(self):
        with pytest.raises(ValueError):
            self.engine.apply(action_id=99, threat_level=0.5)

    def test_no_op_zero_reduction(self):
        outcome = self.engine.apply(action_id=0, threat_level=0.8)
        assert outcome.threat_reduction == pytest.approx(0.0)
        assert outcome.service_cost == pytest.approx(0.0)

    def test_isolate_highest_reduction(self):
        outcomes = [self.engine.apply(i, 0.8) for i in range(4)]
        reductions = [o.threat_reduction for o in outcomes]
        assert reductions[3] == max(reductions)

    def test_isolate_highest_cost(self):
        outcomes = [self.engine.apply(i, 0.8) for i in range(4)]
        costs = [o.service_cost for o in outcomes]
        assert costs[3] == max(costs)

    def test_reduction_bounded(self):
        for action_id in range(4):
            for threat in [0.0, 0.3, 0.6, 1.0]:
                outcome = self.engine.apply(action_id, threat)
                assert 0.0 <= outcome.threat_reduction <= 1.0
                assert 0.0 <= outcome.service_cost <= 1.0


# ---------------------------------------------------------------------------
# NetworkSecurityEnv
# ---------------------------------------------------------------------------

class TestNetworkSecurityEnv:
    def setup_method(self):
        self.env = NetworkSecurityEnv(attack_mode="brute_force", intensity="medium")

    def teardown_method(self):
        self.env.close()

    def test_observation_space_shape(self):
        obs, _ = self.env.reset()
        assert obs.shape == (6,)

    def test_observation_in_range(self):
        obs, _ = self.env.reset()
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)

    def test_action_space_size(self):
        assert self.env.action_space.n == 4

    def test_step_returns_correct_types(self):
        self.env.reset()
        obs, reward, terminated, truncated, info = self.env.step(0)
        assert obs.shape == (6,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_obs_in_range(self):
        self.env.reset()
        for _ in range(10):
            obs, _, _, _, _ = self.env.step(self.env.action_space.sample())
            assert np.all(obs >= 0.0)
            assert np.all(obs <= 1.0)

    def test_episode_terminates_at_max_steps(self):
        self.env.reset()
        done = False
        steps = 0
        while not done:
            _, _, terminated, truncated, _ = self.env.step(0)
            done = terminated or truncated
            steps += 1
        assert steps == NetworkSecurityEnv.MAX_STEPS

    def test_reset_resets_step_count(self):
        self.env.reset()
        for _ in range(10):
            self.env.step(0)
        self.env.reset()
        assert self.env._step_count == 0

    def test_info_contains_expected_keys(self):
        self.env.reset()
        _, _, _, _, info = self.env.step(1)
        for key in ("action_name", "threat_level", "threat_reduction",
                    "service_cost", "episode_reward", "step", "phase"):
            assert key in info

    def test_reward_no_op_under_high_threat_penalised(self):
        np.random.seed(0)
        env_noop = NetworkSecurityEnv(attack_mode="flood", intensity="high")
        env_block = NetworkSecurityEnv(attack_mode="flood", intensity="high")

        rewards_noop = []
        rewards_block = []
        for _ in range(3):
            obs, _ = env_noop.reset()
            env_block.reset()
            for _ in range(10):
                obs, r, _, _, _ = env_noop.step(0)
                rewards_noop.append(r)
            for _ in range(10):
                obs, r, _, _, _ = env_block.step(2)
                rewards_block.append(r)

        env_noop.close()
        env_block.close()
        assert np.mean(rewards_block) > np.mean(rewards_noop)

    def test_all_attack_modes(self):
        for mode in AttackSimulator.MODES:
            env = NetworkSecurityEnv(attack_mode=mode, intensity="medium")
            obs, _ = env.reset()
            assert obs.shape == (6,)
            for _ in range(5):
                obs, _, _, _, _ = env.step(env.action_space.sample())
                assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
            env.close()

    def test_temporal_window_stacking(self):
        env = NetworkSecurityEnv(
            attack_mode="brute_force", intensity="medium", temporal_window=3
        )
        obs, _ = env.reset()
        assert obs.shape == (18,)  # 6 * 3
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
        env.close()

    def test_noisy_observations(self):
        env = NetworkSecurityEnv(
            attack_mode="brute_force", intensity="medium",
            noisy_observations=True, noise_std=0.1,
        )
        obs, _ = env.reset()
        assert obs.shape == (6,)
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
        env.close()

    def test_resource_budget(self):
        env = NetworkSecurityEnv(
            attack_mode="brute_force", intensity="medium", resource_budget=3,
        )
        env.reset()
        # Use 3 actions → budget exhausted → forced noop
        for _ in range(3):
            env.step(1)
        _, _, _, _, info = env.step(1)  # should be forced to noop
        assert info["action_name"] == "no_op"
        env.close()


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class TestBaselines:
    def test_always_noop(self):
        p = AlwaysNoopPolicy()
        assert p.predict(np.zeros(6)) == 0

    def test_random_policy(self):
        p = RandomPolicy(seed=0)
        actions = [p.predict(np.zeros(6)) for _ in range(100)]
        assert len(set(actions)) > 1  # should pick more than one action

    def test_rule_based_threshold(self):
        p = RuleBasedThresholdPolicy()
        # High threat → isolate
        obs = np.array([0.5, 0.5, 0.5, 0.5, 0.8, 0.0])
        assert p.predict(obs) == 3
        # Low threat → noop
        obs_low = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.0])
        assert p.predict(obs_low) == 0

    def test_get_baseline_registry(self):
        for name in ["always_noop", "random_policy", "rule_based_threshold"]:
            bl = get_baseline(name)
            assert hasattr(bl, "predict")
