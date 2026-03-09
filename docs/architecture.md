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
│   RL Agent (DQN/PPO/A2C/RecurrentPPO)               │
│   + Adversarial Attacker (self-play)                │
│   + Explainability (XAI)                            │
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
Generates synthetic attack traffic for four attack modes:

| Mode         | Description                                                  |
|--------------|--------------------------------------------------------------|
| brute_force  | Spikes in failed login attempts; moderate traffic            |
| flood        | High traffic rate (DDoS-style); few login failures           |
| adaptive     | Dynamically switches between brute_force and flood to evade  |
| multi_stage  | Three-phase attack: reconnaissance → exploitation → persistence |

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

- **Algorithms**: DQN (primary), PPO, A2C, RecurrentPPO (LSTM, via sb3-contrib)
- Exposes `train()`, `train_curriculum()`, `predict()`, `save()` methods
- `RewardLoggerCallback` records per-episode cumulative rewards for analysis
- RecurrentPPO uses `MlpLstmPolicy` with 128-unit LSTM for temporal reasoning

---

### 3b. Adversarial Attacker (`airs/agent/adversarial_attacker.py`)
Second RL agent trained as the attacker via **self-play**.

- `AttackerEnv` — gym env where the agent controls 6 attack strategies
  (stealth, brute force, flood, full assault, balanced, evasion)
- `SelfPlayTrainer` — alternating training loop: freeze attacker → train defender
  → freeze defender → train attacker → repeat for N rounds
- Attacker observes defender's last action and adapts strategy
- Both agents improve simultaneously, producing a more robust defender

---

### 4. Monitoring Layer (`airs/monitoring/monitor.py`)
- Reads live CPU / memory utilisation via `psutil`
- Computes a scalar threat-level score from the raw observation vector
- Threat level is a weighted dot-product of five normalised indicators

---

### 5. Response Engine (`airs/response/response_engine.py`)
Maps the discrete action integer to a simulated `ActionOutcome`.

**Stochastic outcomes**: Each action now has a configurable success probability.
On failure, threat reduction is scaled to a residual fraction (default 10%),
but service cost is still incurred — modelling real-world unreliability.

| Action | Name            | Base Reduction | Base Cost | Success Prob |
|--------|-----------------|----------------|-----------|-------------|
| 0      | no_op           | 0%             | 0%        | 100%         |
| 1      | block_ip        | 55%            | 10%       | 90%          |
| 2      | rate_limit      | 40%            | 15%       | 80%          |
| 3      | isolate_service | 80%            | 40%       | 85%          |

Effective values are scaled by the current threat level.
Configurable via `stochastic`, `success_probs`, `failure_residual`, `seed`.

---

### 6. Explainability (`airs/explainability/__init__.py`)
Module to understand **why** the agent chose a particular action.

- **Perturbation importance** (always available): perturbs each observation
  feature and measures how the chosen-action score shifts
- **SHAP integration** (optional, requires `shap` package): KernelSHAP on
  the model's predict function
- Outputs: `Explanation` dataclass with feature importance, action values,
  and a human-readable summary
- Works with DQN (Q-values), PPO/A2C (logits), and RecurrentPPO

---

### 7. Visualizer (`airs/visualization/`)
Visualisation suite:

- `visualizer.py` — static plots: reward curve, action distribution, threat timeline
- `renderer.py` — real-time pygame network visualiser with particle effects
- `training_visualizer.py` — pixel-art training dashboard (live during training)
- `dashboard.py` — Streamlit interactive dashboard

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
│   │   ├── network_env.py
│   │   └── multi_scenario_env.py
│   ├── agent/
│   │   ├── rl_agent.py            ← DQN / PPO / A2C / RecurrentPPO
│   │   ├── adversarial_attacker.py ← self-play attacker + SelfPlayTrainer
│   │   └── baselines.py           ← noop / random / rule-based
│   ├── monitoring/
│   │   └── monitor.py
│   ├── response/
│   │   └── response_engine.py     ← stochastic action outcomes
│   ├── explainability/
│   │   └── __init__.py            ← perturbation importance + SHAP
│   └── visualization/
│       ├── visualizer.py
│       ├── renderer.py            ← pygame real-time visualiser
│       ├── training_visualizer.py ← pixel-art training dashboard
│       ├── training_state.py
│       └── dashboard.py           ← Streamlit dashboard
├── scripts/
│   ├── train.py
│   ├── train_universal.py
│   ├── train_self_play.py         ← adversarial self-play training
│   ├── train_with_visualizer.py
│   ├── evaluate.py
│   ├── evaluate_all.py
│   └── watch_agent.py
├── tests/
│   └── test_environment.py        ← 55 tests
├── configs/
│   └── default.yaml
├── docs/
│   ├── architecture.md            ← this file
│   ├── environment_design.md
│   ├── reward_design.md
│   ├── training_strategy.md
│   ├── experiments_log.md
│   └── lessons_learned.md
├── pyproject.toml                 ← ruff config
└── requirements.txt
```
