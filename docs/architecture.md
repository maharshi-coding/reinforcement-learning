# AIRS – Architecture Documentation

## Overview

The Autonomous Intrusion Response System (AIRS) is a production-grade,
research-level Reinforcement Learning system that learns to defend a simulated
network against dynamically changing cyber-attacks.

---

## System Layers

```
┌─────────────────────────────────────────────────────┐
│                     Layer 3: Agent                  │
│   RL Agent (DQN / PPO) – learns the defense policy  │
└───────────────────────┬─────────────────────────────┘
                        │ obs / reward / action
┌───────────────────────▼─────────────────────────────┐
│                Layer 2: Environment                 │
│  NetworkSecurityEnv (Gymnasium)                     │
│    ├── AttackSimulator  (generates threats)         │
│    ├── SystemMonitor    (collects metrics)          │
│    └── ResponseEngine   (executes actions)          │
└───────────────────────┬─────────────────────────────┘
                        │ metrics / outcomes
┌───────────────────────▼─────────────────────────────┐
│             Layer 1: Simulation / Data              │
│  psutil metrics + synthetic attack traffic          │
│  (replaceable with real network tap)                │
└─────────────────────────────────────────────────────┘
```

---

## Module Breakdown

### 1. Attack Simulator (`airs/environment/attack_simulator.py`)
Generates synthetic attack traffic for three attack modes:

| Mode         | Description                                                  |
|--------------|--------------------------------------------------------------|
| brute_force  | Spikes in failed login attempts; moderate traffic            |
| flood        | High traffic rate (DDoS-style); few login failures           |
| adaptive     | Dynamically switches between brute_force and flood to evade  |

Configurable intensity levels: `low`, `medium`, `high`.

The adaptive attacker switches strategy either periodically or when a strong
defensive action (block / isolate) is detected, modelling a realistic adversary.

---

### 2. Network Environment (`airs/environment/network_env.py`)
Gymnasium-compatible environment implementing the full RL loop.

- **Observation space**: `Box(6,)` – all features normalised to [0, 1]
- **Action space**: `Discrete(4)` – 0=no_op, 1=block_ip, 2=rate_limit, 3=isolate
- **Reward function**: composite (see `reward_design.md`)
- **Episode length**: 200 steps (truncated, not terminated)

---

### 3. RL Agent (`airs/agent/rl_agent.py`)
Thin wrapper around Stable-Baselines3.

- **Primary algorithm**: DQN (justified by discrete action space + value-based objective)
- **Secondary algorithm**: PPO (available for comparison experiments)
- Exposes `train()`, `predict()`, `save()` methods
- `RewardLoggerCallback` records per-episode cumulative rewards for analysis

---

### 4. Monitoring Layer (`airs/monitoring/monitor.py`)
- Reads live CPU / memory utilisation via `psutil`
- Computes a scalar threat-level score from the raw observation vector
- Threat level is a weighted dot-product of five normalised indicators

---

### 5. Response Engine (`airs/response/response_engine.py`)
Maps the discrete action integer to a simulated `ActionOutcome`:

| Action | Name            | Base Threat Reduction | Base Service Cost |
|--------|-----------------|-----------------------|-------------------|
| 0      | no_op           | 0%                    | 0%                |
| 1      | block_ip        | 55%                   | 10%               |
| 2      | rate_limit      | 40%                   | 15%               |
| 3      | isolate_service | 80%                   | 40%               |

Effective values are scaled by the current threat level.

---

### 6. Visualizer (`airs/visualization/visualizer.py`)
Generates four types of plots (saved as PNG):

1. `reward_curve.png` – cumulative episode reward with moving average
2. `action_distribution.png` – bar chart of action frequency
3. `threat_timeline.png` – threat level + action scatter per timestep
4. `attack_success_rate.png` – attack success ratio over evaluation episodes

---

## Data Flow

```
Episode start
    │
    ▼
AttackSimulator.step(last_action)   → raw traffic metrics
    │
    ▼
SystemMonitor.compute_threat_level  → threat ∈ [0,1]
    │
    ▼
Build observation vector (6-dim)
    │
    ▼
RL Agent.predict(obs)               → action ∈ {0,1,2,3}
    │
    ▼
ResponseEngine.apply(action, threat) → ActionOutcome
    │
    ▼
Compute reward                      → scalar r
    │
    ▼
Return (obs, r, terminated, truncated, info)
```

---

## File Structure

```
reinforcement-learning/
├── airs/
│   ├── environment/
│   │   ├── attack_simulator.py
│   │   └── network_env.py
│   ├── agent/
│   │   └── rl_agent.py
│   ├── monitoring/
│   │   └── monitor.py
│   ├── response/
│   │   └── response_engine.py
│   └── visualization/
│       └── visualizer.py
├── scripts/
│   ├── train.py
│   └── evaluate.py
├── tests/
│   └── test_environment.py
├── docs/
│   ├── architecture.md         ← this file
│   ├── environment_design.md
│   ├── reward_design.md
│   ├── training_strategy.md
│   ├── experiments_log.md
│   └── lessons_learned.md
└── requirements.txt
```
