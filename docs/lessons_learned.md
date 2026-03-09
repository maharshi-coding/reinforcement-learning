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

---

## Stochastic Environment Lessons

### L11: Stochastic Actions Increase Reward Variance
With actions that can fail (90%/80%/85% success rates), the reward signal
becomes noisier. The agent must learn to account for expected value rather
than deterministic outcomes. This initially slows convergence but produces
policies that are robust to real-world unreliability.

### L12: Failed Actions Still Incur Cost — Intentional Design
We chose to always charge service cost even on failure (a firewall rule
attempt still disrupts traffic briefly, even if the attacker evades it).
This teaches the agent that actions have guaranteed downsides but uncertain
upsides, matching real cybersecurity operations.

---

## Self-Play Lessons

### L13: Alternating Freeze is Simpler Than Joint Training
Joint training (both agents learning simultaneously) is unstable—the loss
landscape shifts for both agents every step. Alternating freeze (train one
while the other is fixed) converges more reliably.

### L14: Attacker Entropy Bonus Should Be Higher
The attacker benefits from higher exploration (`ent_coef=0.02` vs 0.01 for
defender) because it needs to discover diverse attack strategies to challenge
the defender effectively.

---

## Explainability Lessons

### L15: Perturbation Importance Is Model-Agnostic
Unlike gradient-based saliency, perturbation importance works with any SB3
model (DQN, PPO, A2C, RecurrentPPO) without accessing internal gradients.
This makes it the default XAI method.

### L16: Always Move Obs Tensor to Model Device
When extracting Q-values or logits directly from the model's network, the
observation tensor must be on the same device as the model weights. Missing
this caused CUDA/CPU mismatch errors in the initial implementation.

---

## Architecture Lessons (Consolidation)

### L17: Consolidate Early — Don't Maintain Two Module Trees
The project initially had both `src/` and `airs/` with overlapping code.
This caused import confusion and duplicated maintenance. Consolidating into
a single `airs/` module tree immediately improved clarity and reduced bugs.

### L18: Conditional Imports for Optional Dependencies
RecurrentPPO depends on `sb3-contrib`, which may not be installed.
Using `try/except ImportError` with a `_HAS_RECURRENT` flag keeps the
main agent module functional even without the optional dependency.
