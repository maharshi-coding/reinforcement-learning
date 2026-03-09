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

## Experiment 007: Reward Function / Threat Level Fix
- **Date**: 2026-03-08
- **Problem**: Agent scored 17.99 (below "do nothing" at 20.0), 100% FPR, spammed single action
- **Root cause analysis**:
  1. **Threat level too low**: brute_force medium produced threat ≈ 0.12. The `HIGH_THREAT_THRESHOLD=0.6` never triggered, so the ineffective penalty (for ignoring attacks) never fired.
  2. **"Do nothing" was free**: survival bonus gave +0.1/step × 200 = 20.0 with zero risk. No penalty for ignoring attacks.
  3. **False positive penalty fired during real attacks**: because threat < 0.2 even during genuine brute-force, any defensive action was counted as a false positive.
- **Fixes applied** (in `airs/monitoring/monitor.py`, `airs/environment/network_env.py`, `src/environment/intrusion_env.py`):
  1. **Threat computation**: replaced `spike = t*f + 0.5*c` with `dominant = max(t,f); spike = dominant^0.5 + 0.3*c`, plus sqrt scaling. Threat now ≈ 0.5–0.6 for medium attacks.
  2. **Breach damage**: new accumulating penalty when agent does nothing during attacks. Capped at 3.0, multiplied by 1.0. Makes passivity non-viable.
  3. **Survival bonus**: scaled by `(1 - threat)` so calm periods reward patience, but active attacks don't.
- **Validation** (hand-coded policies):
  - Do nothing: -598 (was +20)
  - Smart (block when threat > 0.3): +513
  - Always block: +512
  - Random: +226

## Experiment 008: DQN Retrained (Post-Fix)
- **Date**: 2026-03-08
- **Algorithm**: DQN
- **Attack mode**: brute_force, medium intensity
- **Timesteps**: 100,000 (early-stopped at 300 episodes)
- **Final avg reward (last 20)**: 553.22
- **Evaluation (50 episodes, 5 seeds)**:
  - Mean reward: 571.17 ± 1.85, CI [570.68, 571.69]
  - Attack success: 0.0%
  - False positive rate: **0.0%** (was 100%)
  - Detection delay: 0.0 steps
  - Service downtime: 85.86
- **Baseline comparison**:
  - vs always_noop: **+1171.53** (p < 0.0001)
  - vs random: **+352.44** (p < 0.0001)
  - vs rule_based: **+187.24** (p < 0.0001)
- **OOD results**:
  - bursty_traffic (flood, high): 700.71 — strong generalisation
  - noisy_telemetry (brute_force, medium): 569.59 — robust to noise
  - unseen_attack_combo (multi_stage, high): 587.43 — handles novel attacks
- **Status**: PASS — agent decisively outperforms all baselines
- **Note**: Agent favours `isolate_service` (highest reduction). Future work: penalise over-isolation to encourage lighter actions.
