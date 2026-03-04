# AIRS – Reward Function Design

## Overview

The reward function is the most critical component of the RL formulation.
A poorly designed reward leads to:
- Reward hacking (e.g., always isolate → high reward but zero availability)
- Policy collapse (e.g., always no-op)
- False positive exploitation (e.g., blocking everything even normal traffic)

---

## Formal Definition

At each step t, the scalar reward r_t is:

```
r_t = R_threat  −  R_service  −  R_false_positive  −  R_ineffective  +  R_survival
```

### Components

#### R_threat (Positive – Attack Mitigation)
```
R_threat = outcome.threat_reduction × 10
```
- Range: [0, 10]
- `threat_reduction` ∈ [0, 1] comes from `ResponseEngine.apply(action, threat_level)`
- Scales with both the action strength and the current threat level
- **Design choice**: weight = 10 makes this the dominant signal

#### R_service (Negative – Service Disruption Penalty)
```
R_service = outcome.service_cost × 5
```
- Range: [0, 5]
- Penalises disrupting legitimate users
- **Design choice**: weight = 5 (half of threat reward) balances security vs availability

#### R_false_positive (Negative – False Positive Penalty)
```
if threat_level < 0.2 and action > 0:
    R_false_positive = (action / 3) × 3
else:
    R_false_positive = 0
```
- Range: [0, 3]
- Penalises aggressive actions when threat is actually low
- Scales with action severity (isolate penalised more than block)
- **Design choice**: weight = 3 prevents the agent from defending when nothing is wrong

#### R_ineffective (Negative – Inaction Under High Threat)
```
if threat_level > 0.6 and action == 0:
    R_ineffective = threat_level × 5
else:
    R_ineffective = 0
```
- Range: [0, 5]
- Penalises doing nothing while under serious attack
- **Design choice**: proportional to threat level – worse to ignore severe attacks

#### R_survival (Positive – Step Bonus)
```
R_survival = 0.1
```
- Fixed small bonus for each completed step
- Encourages the agent to stay alive (maintain service) rather than collapsing
- **Design choice**: small enough not to dominate (0.1 vs 10 for threat reduction)

---

## Example Scenarios

| Scenario                           | Expected Reward Range |
|------------------------------------|-----------------------|
| High threat + block IP             | ~+3 to +5             |
| High threat + isolate service      | ~+4 to +6             |
| High threat + no-op                | ~-4 to -3             |
| Low threat + no-op                 | ~+0.1 (survival)      |
| Low threat + isolate (overkill)    | ~-1 to 0              |
| Medium threat + rate limit         | ~+2 to +4             |

---

## Discount Factor

**γ = 0.99** is used by both DQN and PPO.

**Justification**:
- Cyber-attacks unfold over many steps; the agent must plan ahead
- A high γ (close to 1) ensures the Q-function accounts for future threat propagation
- γ = 0.99 with episode length 200 means the effective planning horizon is
  approximately 1 / (1 - 0.99) = 100 steps

---

## Reward Scale Analysis

Total per-episode reward (200 steps) under optimal policy ≈ +300 to +700
Total per-episode reward under random policy ≈ -200 to +100
Total per-episode reward under always-isolate ≈ +100 to +200 (penalised by service cost)

This spread ensures the reward signal is informative and not degenerate.

---

## Immediate vs Delayed Rewards

- R_threat and R_service are immediate (computed at each step)
- The cumulative effect of defending early (preventing escalation) is captured by γ
- No explicit delayed reward shaping is needed due to the always-attacking simulator

---

## Avoiding Reward Hacking

Potential hacking modes and mitigations:

| Hacking Mode             | Mitigation                                          |
|--------------------------|-----------------------------------------------------|
| Always isolate           | R_service penalty (−40% × 5 = −2) per step         |
| Always no-op             | R_ineffective penalty under high threat             |
| Oscillate block/unblock  | last_action in state + service cost per block       |
| Block low-threat traffic | R_false_positive penalty                            |
