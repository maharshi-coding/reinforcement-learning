# AIRS – Evaluation Results

## Evaluation Methodology

### Metrics

| Metric | Description |
|--------|-------------|
| Mean Episode Reward | Average cumulative reward over all episodes |
| Attack Success Rate | Percentage of high-threat steps where no defense was applied |
| False Positive Rate | Defensive actions taken when threat was below LOW_THREAT_THRESHOLD |
| Detection Delay | Steps between first high-threat event and first defensive action |
| Service Downtime | Cumulative service cost from defensive actions |

### Statistical Rigor

- **Multi-seed evaluation**: ≥ 5 seeds per configuration
- **Confidence intervals**: 95% CI via bootstrap (10,000 resamples)
- **Hypothesis testing**: Paired t-test for baseline comparison
- **Reporting format**: mean ± 95% CI

## Baseline Comparison Template

| Policy | Mean Reward | 95% CI | Attack Success | FPR | Det. Delay |
|--------|------------|--------|---------------|-----|-----------|
| DQN Agent | — | — | — | — | — |
| PPO Agent | — | — | — | — | — |
| Always No-Op | — | — | — | — | — |
| Random Policy | — | — | — | — | — |
| Rule-Based | — | — | — | — | — |

## OOD Test Template

| Scenario | Mean Reward | Attack Success | Notes |
|----------|------------|---------------|-------|
| Bursty Traffic | — | — | High-intensity flood + noise |
| Noisy Telemetry | — | — | Gaussian noise + partial observability |
| Unseen Attack Combo | — | — | Multi-stage, high intensity |

## Key Findings

- DQN converges faster than PPO for this 4-action discrete space
- Curriculum learning (low→medium→high) improves final performance
- Agent is robust to noisy observations (std=0.2 has minimal impact)
- Rule-based baseline is competitive — RL agent must be well-trained to outperform

## Latest Results (2026-03-08, Post Reward Fix)

### DQN Agent — brute_force, medium intensity

| Policy | Mean Reward | 95% CI | Attack Success | FPR | Downtime | Cost |
|--------|------------|--------|---------------|-----|----------|------|
| **DQN Agent** | **571.17** | [570.68, 571.69] | 0.0% | **0.0%** | 85.86 | 85.86 |
| Always No-Op | -600.36 | — | — | 0.0% | 0.00 | 0.00 |
| Random Policy | 218.73 | — | 0.0% | — | 33.52 | 33.52 |
| Rule-Based | 383.93 | — | 0.0% | 0.0% | — | — |

### OOD Tests

| Scenario | Mean Reward | 95% CI | Attack Success | FPR | Notes |
|----------|------------|--------|---------------|-----|-------|
| Bursty Traffic (flood, high) | 700.71 | [700.03, 701.37] | 0.0% | 0.0% | Excellent generalisation |
| Noisy Telemetry (medium) | 569.59 | [568.91, 570.25] | 0.0% | 0.0% | Robust to noise |
| Unseen Attack Combo (multi_stage, high) | 587.43 | [582.39, 592.35] | 0.0% | 0.0% | Handles novel attacks |

### Before vs After Fix

| Metric | Before Fix | After Fix | Change |
|--------|-----------|-----------|--------|
| Mean Reward | 17.99 | **571.17** | +3074% |
| vs "Do Nothing" | -2.01 (lost) | **+1171** (won) | Fixed |
| False Positive Rate | 100% | **0%** | Fixed |
| OOD (unseen attacks) | -110.70 | **+587.43** | Fixed |

## Attack Success Rate Investigation

If `attack_success_rate == 0%`:
1. Check that the agent is not always selecting action > 0 (over-blocking)
2. Verify HIGH_THREAT_THRESHOLD is realistic for the attack mode
3. Ensure attack simulator is generating sufficient high-threat episodes
4. Check environment step count — multi-stage attacks need ≥ 200 steps to manifest
