# AIRS – System Architecture

## Overview

The Autonomous Intrusion Response System (AIRS) is a reinforcement learning–based cybersecurity defense system that learns to respond to network intrusions in real time. It trains across all combinations of attack modes and intensities to produce generalist defense agents.

## Architecture Diagram

```
┌───────────────────────────────────────────────────────────────────┐
│                         AIRS System                               │
│                                                                   │
│  ┌──────────────┐    ┌──────────────────┐    ┌────────────────┐  │
│  │    Attack     │    │  MultiScenario   │    │   RL Agent     │  │
│  │  Simulator    │───▶│  Env (Gymnasium) │◀──▶│ (DQN/PPO/A2C) │  │
│  │  (4 modes)   │    │  12 scenarios    │    │  SB3 wrapper   │  │
│  └──────────────┘    └────────┬─────────┘    └────────────────┘  │
│                               │                                   │
│  ┌──────────────┐    ┌───────▼──────────┐    ┌────────────────┐  │
│  │   System     │    │    Response      │    │  Evaluation    │  │
│  │   Monitor    │───▶│    Engine        │    │  Framework     │  │
│  │  (threat     │    │  (4 actions)     │    │  (12 scenarios │  │
│  │   scoring)   │    │                  │    │   + baselines) │  │
│  └──────────────┘    └──────────────────┘    └────────────────┘  │
│                                                                   │
│  ┌──────────────┐    ┌──────────────────┐    ┌────────────────┐  │
│  │  Curriculum  │    │   Dashboard      │    │   Pygame       │  │
│  │  Trainer     │    │   (Streamlit)    │    │   Visualizer   │  │
│  │  (L→M→H)    │    │   6 tabs         │    │   (real-time)  │  │
│  └──────────────┘    └──────────────────┘    └────────────────┘  │
└───────────────────────────────────────────────────────────────────┘
```

## MultiScenarioEnv Architecture

The `MultiScenarioEnv` is a Gymnasium wrapper that enables generalist agent training:

```
MultiScenarioEnv
├── Randomly samples (attack_mode, intensity) each episode reset
├── attack_modes: [brute_force, flood, adaptive, multi_stage]
├── intensities: [low, medium, high]
├── 4 × 3 = 12 unique scenario combinations
└── Creates fresh NetworkSecurityEnv per episode
```

**Curriculum Training** mode progressively increases difficulty:
1. **Stage 1** (25%): Low intensity only
2. **Stage 2** (25%): Medium intensity only
3. **Stage 3** (25%): High intensity only
4. **Stage 4** (25%): Mixed (all intensities)

## Module Descriptions

### Environment (`airs/environment/`)
- **NetworkSecurityEnv**: Gymnasium-compatible environment implementing the MDP
- **MultiScenarioEnv**: Wrapper that randomizes attack mode + intensity per episode
- **AttackSimulator**: Generates synthetic attack traffic (brute force, flood, adaptive, multi-stage)

### Monitoring (`airs/monitoring/`)
- **SystemMonitor**: Computes composite threat level from system telemetry
  - Nonlinear threat scaling: `sqrt(threat_raw)` for stronger signal
  - Dominant signal amplification: `min(dominant^0.5 + 0.3*cpu, 1.0)`

### Response (`airs/response/`)
- **ResponseEngine**: Maps defensive actions to outcomes with threat reduction and service cost

### Agent (`airs/agent/`)
- **AIRSAgent**: Wraps Stable-Baselines3 DQN/PPO/A2C with:
  - Vectorised environments (DummyVecEnv)
  - Multi-scenario training support
  - Best-model checkpointing and early stopping
  - Tuned hyperparameters per algorithm

### Training (`scripts/train_universal.py`)
- Multi-algorithm training pipeline (DQN, PPO, A2C)
- Standard multi-scenario or curriculum training modes
- Learning curve logging (reward vs timesteps)
- JSON export of training data

### Evaluation (`scripts/evaluate_all.py`)
- Tests all algorithms across all 12 scenarios + 3 baselines
- Generates charts: heatmaps, bar charts, algorithm comparison
- CSV export of all results

### Visualization
- **Streamlit Dashboard** (`scripts/dashboard.py`): 6 tabs, dark theme, interactive
- **Pygame Renderer** (`src/visualization/renderer.py`): Real-time visualization with sparklines

## Reward Function

```
reward = threat_reduction × threat_weight
       - service_cost × service_cost_weight
       - false_positive_penalty        (action during low threat)
       - unnecessary_action_penalty    (action during ~zero threat)
       - breach_penalty                (accumulated threat from inaction)
       - ineffective_penalty           (no-op during high threat)
       + survival_bonus × (1 - threat)
```

Key design: **unnecessary_action_penalty** penalizes defensive actions when threat < 0.1, encouraging selective behavior. Heavier actions (isolate > rate_limit > block) incur larger penalties.

## Multi-Scenario Training Results (500k steps)

| Algorithm | Avg Reward | Best Scenario | Worst Scenario | FPR |
|-----------|-----------|---------------|----------------|-----|
| **PPO** | **617.3** | flood/high (835.1) | brute_force/low (472.8) | 0% |
| **DQN** | **604.6** | flood/high (859.4) | brute_force/low (484.6) | 0% |
| **A2C** | **583.0** | flood/high (700.2) | brute_force/low (484.5) | 0% |

### By Intensity
| Intensity | DQN | PPO | A2C |
|-----------|-----|-----|-----|
| Low | 512.6 | 517.0 | 508.5 |
| Medium | 600.8 | 619.7 | 581.5 |
| High | 700.5 | 715.3 | 658.9 |

### By Attack Mode
| Mode | DQN | PPO | A2C |
|------|-----|-----|-----|
| brute_force | 570.4 | 557.4 | 579.5 |
| flood | 718.7 | 711.2 | 618.1 |
| adaptive | 572.7 | 642.5 | 598.7 |
| multi_stage | 556.7 | 558.1 | 535.5 |

### Baselines
| Baseline | Avg Reward |
|----------|-----------|
| always_noop | -849.8 |
| random_policy | 168.1 |
| rule_based_threshold | 392.1 |

## Technology Stack

| Component | Technology |
|-----------|-----------|
| RL Framework | Stable-Baselines3 (DQN, PPO, A2C) |
| Environment | Gymnasium |
| Deep Learning | PyTorch |
| Experiment Tracking | MLflow |
| Visualisation | Matplotlib, Streamlit, Pygame |
| Configuration | PyYAML |
| Statistics | SciPy |
