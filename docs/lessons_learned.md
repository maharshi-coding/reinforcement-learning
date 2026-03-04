# AIRS – Lessons Learned

## Architecture Lessons

### L1: Observation Normalisation is Non-Negotiable
All features must be in [0, 1] before entering the neural network.
Raw traffic rates (0–1000) or login counts (0–300) cause gradient instability
in DQN. Clipping after normalisation prevents edge-case NaN values.

### L2: Include Last Action in State
Without `last_action` in the observation, the agent cannot distinguish between
"I just blocked this IP" and "I haven't acted yet", leading to redundant
re-blocking and oscillating policies. Adding it as obs[5] gave temporal context.

### L3: Separate Simulation from Real Metrics
The `use_real_system_metrics` flag in `NetworkSecurityEnv` allows switching
between psutil (real) and simulated CPU/memory. This is essential because
the sandbox training environment's CPU load doesn't reflect a real attacked server.

---

## Reward Design Lessons

### L4: Service Cost Weight Must Be Below Threat Reward Weight
Initial design had service_cost × 10 (equal to threat_reduction × 10).
This caused the agent to learn "never act" because the service disruption
matched the threat mitigation benefit. Reducing to × 5 fixed this.

### L5: False Positive Penalty Must Scale with Action Severity
A flat false-positive penalty caused the agent to prefer `block_ip` (action 1)
over `isolate` (action 3) regardless of threat, because the isolate penalty was
the same as block despite being far more disruptive. Scaling by action_id / 3
resolved this.

### L6: Inaction Penalty Is Crucial for Dense Threat Environments
Without R_ineffective, the agent learned to do nothing (no-op) under high
flood attacks because the service cost of rate-limiting outweighed the
threat reduction in the early training phase. Adding the inaction penalty
pushed the agent to act under high-threat conditions.

---

## Training Lessons

### L7: Warm-Up Buffer Prevents Early Convergence to Bad Policies
Setting `learning_starts=500` ensures the replay buffer has diverse
transitions before the first gradient update. Without this, early
updates overfit to the (randomly-dominated) initial experience.

### L8: Target Network Update Interval of 500 Steps
A shorter interval (50 steps) caused the Q-function to chase its own
target, leading to divergence. 500 steps provides stable Bellman targets
for the observation/reward scale in this environment.

---

## Debugging Lessons

### L9: Always Verify with Unit Tests Before Training
Running `pytest tests/` takes <1 second and catches environment bugs
that would otherwise only surface after 30+ minutes of training.

### L10: Plot Threat Timeline to Diagnose Policy Collapse
The `plot_threat_timeline()` visualisation clearly shows when the agent
oscillates between actions (saw-tooth pattern in action scatter) vs when
it makes stable, threat-appropriate choices.
