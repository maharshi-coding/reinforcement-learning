# AIRS – Experiments Log

## Experiment Format

Each entry records: algorithm, attack_mode, intensity, timesteps, hyperparameters,
random seed, key results, and observations.

---

## Experiment 001 – DQN Baseline (Brute Force, Medium)

| Field            | Value                      |
|------------------|----------------------------|
| Date             | 2026-03-04                 |
| Algorithm        | DQN                        |
| Attack mode      | brute_force                |
| Intensity        | medium                     |
| Timesteps        | 50,000                     |
| Random seed      | default (SB3)              |
| Learning rate    | 1e-3                       |
| Gamma            | 0.99                       |
| Buffer size      | 50,000                     |
| Exploration frac | 0.30                       |

**Results**:
- Environment and all modules verified (30 unit tests passed)
- Simulation: agent correctly receives 6-dim observations in [0,1]
- Reward signal: verified positive correlation between defensive action and reward
  under high-threat conditions
- No-op under high threat correctly penalised (test passed)
- All attack modes (brute_force, flood, adaptive) produce valid observations

**Observations**:
- The environment correctly models attacker back-off after strong defensive action
- Threat level computation is deterministic and bounded
- Adaptive attacker switches strategy after block/isolate actions as designed

---

## Experiment 002 – Planned: DQN vs PPO Comparison

| Field         | DQN (planned) | PPO (planned) |
|---------------|---------------|---------------|
| Attack mode   | adaptive      | adaptive      |
| Intensity     | medium        | medium        |
| Timesteps     | 100,000       | 100,000       |
| Expected      | faster conv.  | more stable?  |

---

## Experiment 003 – Planned: Sparse vs Dense Reward

Test whether removing R_survival (sparse reward) affects convergence speed
and final policy quality.

---

## Experiment 004 – Planned: Static vs Adaptive Attacker

Compare final policy performance when trained only on static attacker vs
adaptive attacker. Hypothesis: adaptive-trained policy generalises better.
