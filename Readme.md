# AIRS – Autonomous Intrusion Response System

A **reinforcement learning–based cybersecurity defense system** that learns to respond to network intrusions in real time using DQN and PPO algorithms.

## Project Structure

```
├── src/                              # Main source code (modular)
│   ├── environment/
│   │   └── intrusion_env.py          # Gymnasium MDP environment
│   ├── agent/
│   │   └── rl_agent.py               # DQN/PPO agent wrapper
│   ├── training/
│   │   └── train_agent.py            # Training pipeline + MLflow
│   ├── evaluation/
│   │   ├── evaluate_agent.py         # Evaluation script
│   │   └── metrics.py                # Metrics, stats, CSV export
│   ├── baselines/
│   │   ├── no_op_policy.py           # Always no-op baseline
│   │   ├── random_policy.py          # Random action baseline
│   │   └── rule_based_defender.py    # Threshold heuristic baseline
│   └── visualization/
│       ├── visualizer.py             # Plotting utilities
│       └── dashboard.py              # Streamlit dashboard
├── configs/
│   ├── default.yaml                  # Legacy config
│   └── training_config.yaml          # Main training config
├── experiments/
│   ├── training_runs/                # MLflow tracking
│   └── evaluation_results/           # Evaluation artifacts
├── models/                           # Saved model weights
├── plots/                            # Generated visualizations
├── tasks/                            # Task tracking
├── obsidian/                         # Research documentation vault
├── tests/                            # Test suite
└── docs/                             # Design documentation
```

## MDP Formulation

| Component | Definition |
|-----------|-----------|
| **State** | `[failed_logins, request_rate, cpu_usage, memory_usage, alert_level, last_action]` |
| **Actions** | `{do_nothing, block_ip, rate_limit, isolate_service}` |
| **Reward** | `security_gain - service_cost - false_positive_penalty + survival_bonus` |
| **Transitions** | Stochastic (attacker + system dynamics) |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train a DQN agent
python src/training/train_agent.py --config configs/training_config.yaml

# Train with PPO + curriculum learning
python src/training/train_agent.py --algorithm ppo --curriculum

# Evaluate with baselines and OOD tests
python src/evaluation/evaluate_agent.py --multi_seed --baselines --ood

# Launch dashboard
streamlit run src/visualization/dashboard.py
```

## Features

- **Algorithms**: DQN, PPO (via Stable-Baselines3)
- **Attack modes**: brute force, flood, adaptive, multi-stage
- **Training**: curriculum learning, early stopping, parallel envs
- **Evaluation**: multi-seed (5+), 95% CI, paired t-test, bootstrap
- **Baselines**: no-op, random, rule-based threshold
- **OOD testing**: bursty traffic, noisy telemetry, unseen attacks
- **Tracking**: MLflow experiment logging
- **Visualization**: reward curves, action distributions, threat timelines
- **Dashboard**: Streamlit interactive monitoring
- **Reproducibility**: YAML configs, fixed seeds, deterministic training

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| Mean Episode Reward | Average cumulative reward |
| Attack Success Rate | % of attacks succeeding |
| False Positive Rate | Legitimate actions incorrectly blocked |
| Detection Delay | Steps to first defensive response |
| Service Downtime | Disruption caused by defense |

## Configuration

All parameters are configured via `configs/training_config.yaml`:

```yaml
seed: 42
environment:
  attack_mode: brute_force
  intensity: medium
agent:
  algorithm: dqn
training:
  total_timesteps: 50000
  curriculum:
    enabled: true
```

## License

Research project – all rights reserved.
