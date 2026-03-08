# AIRS – Training Log

## Experiment Protocol

All experiments follow the multi-seed protocol:
- **Seeds**: minimum 5 per configuration
- **Reporting**: mean ± 95% confidence interval
- **Statistical tests**: paired t-test, bootstrap CI

## Experiment 001: DQN Baseline Training
- **Date**: 2026-03-05
- **Algorithm**: DQN
- **Attack mode**: brute_force, medium intensity
- **Timesteps**: 5,000 (smoke test)
- **Result**: 25 episodes, avg reward 0.72
- **Status**: PASS — training pipeline functional

## Experiment 002: PPO Baseline Training
- **Date**: 2026-03-05
- **Algorithm**: PPO
- **Attack mode**: brute_force, medium intensity
- **Timesteps**: 5,000 (smoke test)
- **Result**: 25 episodes, avg reward -82.67
- **Note**: PPO needs more timesteps to converge for this env (small discrete action space favours DQN)

## Experiment 003: Curriculum Learning (DQN)
- **Date**: 2026-03-05
- **Algorithm**: DQN
- **Attack mode**: brute_force
- **Stages**: low → medium → high (15k/20k/15k timesteps)
- **Result**: 250 episodes, avg reward 133.61
- **Status**: Curriculum learning working, shows progressive learning

## Experiment 004: Multi-Seed Evaluation (DQN)
- **Date**: 2026-03-05
- **Algorithm**: DQN
- **Attack mode**: brute_force
- **Seeds**: 5
- **Result**: Mean 18.06 ± 0.24, CI [17.76, 18.36]
- **Baseline comparison**: Agent outperforms random by +235; on par with rule-based and noop

## Experiment 005: OOD Tests (DQN)
- **Date**: 2026-03-05
- **Scenarios**: bursty_traffic, noisy_telemetry, unseen_attack_combo
- **Results**:
  - bursty_traffic: 561.64 (strong generalisation to flood)
  - noisy_telemetry: 17.84 (robust to noise)
  - unseen_attack_combo (multi_stage): 200.03 (handles novel attacks)

## Experiment 006: Full Pipeline Validation
- **Date**: 2026-03-08
- **Algorithm**: DQN
- **Config**: configs/training_config.yaml
- **Status**: Restructured src/ pipeline validation
- **Notes**: Validated training, evaluation, baselines, MLflow tracking, plot generation
