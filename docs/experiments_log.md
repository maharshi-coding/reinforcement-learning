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

---

## Experiment 005 – Stochastic Action Outcomes

| Field            | Value                      |
|------------------|----------------------------|
| Date             | 2026-03-09                 |
| Change           | Stochastic ResponseEngine  |
| Success probs    | block=0.90, limit=0.80, isolate=0.85 |
| Failure residual | 10% of normal reduction    |

**Rationale**: Real defensive actions don't always succeed. IP blocks can be
evaded, rate limiting can be circumvented, service isolation may partially fail.
Stochastic outcomes force the agent to learn robust policies rather than
memorising a deterministic environment.

**Impact**: Reward variance increases, which may slow early training convergence
but should produce a more robust final policy.

---

## Experiment 006 – Adversarial Self-Play

| Field            | Value                      |
|------------------|----------------------------|
| Date             | 2026-03-09                 |
| Method           | SelfPlayTrainer (PPO vs PPO) |
| Attacker actions | 6 strategies (stealth, BF, flood, all-in, balanced, evasion) |
| Rounds           | 10 (configurable)          |
| Steps per round  | 20K defender + 20K attacker |

**Rationale**: Scripted attackers have fixed behaviour. A learned attacker
continuously discovers new exploits, forcing the defender to generalise.

**Expected**: Defender policy from self-play should outperform scripted-attacker
trained policy on unseen attack patterns.

---

## Experiment 007 – Planned: RecurrentPPO on Multi-Stage Attacks

| Field            | Value                      |
|------------------|----------------------------|
| Algorithm        | RecurrentPPO (MlpLstmPolicy) |
| Attack mode      | multi_stage                |
| Hypothesis       | LSTM memory allows the agent to track attack phase transitions |
| Comparison       | MlpPolicy PPO vs MlpLstmPolicy RecurrentPPO |

---

## Experiment 008 – Planned: Explainability Analysis

| Field            | Value                      |
|------------------|----------------------------|
| Method           | Perturbation importance + SHAP |
| Goal             | Understand which features drive defensive decisions |
| Expected         | threat_level and traffic_rate should dominate |
