# AIRS – System Architecture

## Overview

The Autonomous Intrusion Response System (AIRS) is a reinforcement learning–based cybersecurity defense system that learns to respond to network intrusions in real time.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                    AIRS System                          │
│                                                         │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │   Attack      │    │  Intrusion   │    │    RL     │ │
│  │  Simulator    │───▶│    Env       │◀──▶│   Agent   │ │
│  │              │    │  (MDP)       │    │ (DQN/PPO) │ │
│  └──────────────┘    └──────┬───────┘    └───────────┘ │
│                             │                           │
│  ┌──────────────┐    ┌──────▼───────┐    ┌───────────┐ │
│  │   System     │    │   Response   │    │ Evaluation│ │
│  │   Monitor    │───▶│   Engine     │    │ Framework │ │
│  │              │    │              │    │           │ │
│  └──────────────┘    └──────────────┘    └───────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Module Descriptions

### Environment (`src/environment/intrusion_env.py`)
- **IntrusionEnv**: Gymnasium-compatible environment implementing the MDP
- **AttackSimulator**: Generates synthetic attack traffic (brute force, flood, adaptive, multi-stage)
- **SystemMonitor**: Computes composite threat level from system telemetry
- **ResponseEngine**: Maps defensive actions to outcomes with threat reduction and service cost

### Agent (`src/agent/rl_agent.py`)
- **AIRSAgent**: Wraps Stable-Baselines3 DQN/PPO with:
  - Vectorised environments (VecEnv)
  - Curriculum learning support
  - Best-model checkpointing
  - Early stopping

### Training (`src/training/train_agent.py`)
- YAML-configurable training pipeline
- MLflow experiment tracking
- Multi-seed reproducibility
- Curriculum learning support

### Evaluation (`src/evaluation/`)
- Multi-seed evaluation with 95% confidence intervals
- Baseline comparison (no-op, random, rule-based)
- Statistical tests (paired t-test, bootstrap CI)
- Out-of-distribution test scenarios
- CSV export

### Baselines (`src/baselines/`)
- **AlwaysNoopPolicy**: Never acts (lower bound)
- **RandomPolicy**: Uniform random actions
- **RuleBasedThresholdPolicy**: Threshold-based heuristic

### Visualization (`src/visualization/`)
- Reward curves with moving average
- Action distribution charts
- Threat level timelines
- Policy comparison plots
- Streamlit dashboard

## Data Flow

1. **AttackSimulator** generates attack traffic each timestep
2. **SystemMonitor** computes threat level from telemetry
3. **IntrusionEnv** presents state (obs) to the **AIRSAgent**
4. Agent selects defensive action via learned policy
5. **ResponseEngine** computes action outcome (threat reduction, service cost)
6. Environment computes multi-objective reward
7. Agent updates policy using reward signal

## Technology Stack

| Component | Technology |
|-----------|-----------|
| RL Framework | Stable-Baselines3 |
| Environment | Gymnasium |
| Deep Learning | PyTorch |
| Experiment Tracking | MLflow |
| Visualisation | Matplotlib, Streamlit |
| Configuration | PyYAML |
| Statistics | SciPy |
