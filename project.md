# AIRS Project Alignment Prompt

### RL-Based Autonomous Intrusion Response System (AIRS)

### Alignment With Course Project Requirements

---

# Objective

Upgrade and restructure the existing **AIRS (RL-Based Autonomous Intrusion Response System)** project so that it satisfies the **formal project structure, evaluation rigor, and experimental methodology required in the course project specification**.

The system must remain a **reinforcement learning–based cybersecurity defense system**, but its implementation, experiments, and documentation must be **structured to match the formal academic requirements of the project specification**.

The upgraded system must demonstrate:

* Correct **MDP formulation**
* Well-defined **policy network architecture**
* Proper **training methodology**
* **Rigorous evaluation**
* **Baseline comparisons**
* **Reproducible experiments**
* **Clear experimental analysis**

---

# 1. Project Restructuring Requirements

Restructure the repository so that the system clearly separates:

```
environment
agent
training
evaluation
experiments
documentation
```

Target structure:

```
airs-project/
│
├── src/
│   ├── environment/
│   │   └── intrusion_env.py
│   │
│   ├── agent/
│   │   └── rl_agent.py
│   │
│   ├── training/
│   │   └── train_agent.py
│   │
│   ├── evaluation/
│   │   ├── evaluate_agent.py
│   │   └── metrics.py
│   │
│   └── baselines/
│       ├── random_policy.py
│       ├── rule_based_defender.py
│       └── no_op_policy.py
│
├── experiments/
│   ├── training_runs
│   └── evaluation_results
│
├── plots/
│
├── configs/
│   └── training_config.yaml
│
├── tasks/
│   ├── todo.md
│   ├── experiments.md
│   └── lessons.md
│
├── README.md
└── report/
```

Each module must remain **independent and modular**.

---

# 2. Formal MDP Definition

Rewrite the AIRS system description to explicitly define the cybersecurity defense problem as an **MDP (S, A, P, R)**.

### State Space

The system state must contain system telemetry:

```
s_t = [
    failed_logins,
    request_rate,
    cpu_usage,
    memory_usage,
    alert_level
]
```

Optionally include temporal context:

```
s_t = [current_state, previous_k_states]
```

---

### Action Space

Actions available to the defender:

```
0 = do_nothing
1 = block_ip
2 = rate_limit
3 = isolate_service
```

---

### Transition Function

State transitions are influenced by:

* attacker behavior
* defender action
* stochastic system dynamics

Transitions must be **explicitly implemented in the environment simulation**.

---

### Reward Function

Define reward using a **multi-objective formulation**:

```
reward = security_gain 
         - service_disruption_cost
         - response_latency_penalty
```

Example rewards:

| Event                 | Reward |
| --------------------- | ------ |
| attack blocked        | +10    |
| successful mitigation | +15    |
| false positive        | -5     |
| service downtime      | -10    |
| attack success        | -15    |

---

### Objective

The RL policy must maximize expected cumulative reward:

```
maximize  E [ Σ reward_t ]
```

---

# 3. Model Architecture

Define the RL policy architecture clearly.

Initial baseline model:

```
Input: system state vector

MLP Policy Network:
Layer 1: 128 neurons
Layer 2: 128 neurons
Output: action probabilities
```

Policy output:

```
πθ(a | s)
```

Architectures to evaluate:

* MLP
* LSTM
* GRU

Document architecture choices.

---

# 4. Training Methodology

Training must follow **policy optimization methods**.

Recommended algorithms:

* PPO
* DQN

Training loop must include:

```
collect experience
update policy
evaluate policy
log metrics
checkpoint model
```

---

# 5. Multi-Seed Experiment Protocol

Every experiment must be executed with **multiple seeds**.

Minimum requirement:

```
seeds = 5
```

Report:

```
mean ± 95% confidence interval
```

Metrics must be aggregated across seeds.

---

# 6. Dataset and Simulation Environment

The project must implement a **synthetic attack generator**.

Attack types:

* brute force authentication attack
* request flooding
* scanning behavior
* burst traffic

The training environment must generate **at least 50,000 interaction steps**.

Testing must use **separate attack sequences** with fixed seeds.

---

# 7. Baseline Comparisons

The RL defender must be compared with classical baseline policies.

Required baselines:

### Always No-Op

```
defender never acts
```

### Random Policy

```
actions sampled uniformly
```

### Rule-Based Defender

Example:

```
if failed_logins > threshold:
    block_ip
```

All baselines must be evaluated using the same environment.

---

# 8. Evaluation Metrics

The system must evaluate:

| Metric              | Description                            |
| ------------------- | -------------------------------------- |
| Mean Episode Reward | average cumulative reward              |
| Attack Success Rate | percentage of attacks succeeding       |
| False Positive Rate | legitimate actions incorrectly blocked |
| Defense Delay       | time to respond to attack              |
| Service Downtime    | disruption caused by defense           |

Statistical tests required:

* paired t-test
* bootstrap confidence intervals

---

# 9. Out-of-Distribution Testing

Evaluate robustness under unseen scenarios:

* new attack combinations
* burst traffic
* noisy telemetry

Record performance degradation.

---

# 10. Visualization

Add training and evaluation visualizations.

Required plots:

* reward vs training steps
* attack success rate vs training
* defense action distribution
* comparison between RL and baselines

Optional dashboard:

* Streamlit
* Gradio

---

# 11. Experiment Tracking

Add experiment tracking using:

```
MLflow
or
Weights & Biases
```

Log:

* hyperparameters
* seeds
* metrics
* model checkpoints

---

# 12. Reproducibility Requirements

Ensure experiments are reproducible.

Required controls:

```
fixed random seeds
deterministic training
configuration files
environment snapshots
```

Use YAML config system.

Example:

```
configs/training_config.yaml
```

---

# 13. Documentation Requirements

Prepare documentation required for final submission.

Report must include:

1. Introduction and motivation
2. MDP formulation
3. Model architecture
4. Training methodology
5. Experimental setup
6. Baseline comparisons
7. Ablation study
8. Failure analysis
9. Conclusion

Report length:

```
8–12 pages
```

---

# 14. Presentation Preparation

Prepare a presentation including:

* problem motivation
* RL formulation
* system architecture
* experimental results
* baseline comparisons

Recommended structure:

```
20 slides
20 minutes
```

Include plots and visualizations.

---

# 15. Task Tracking

Use task files:

```
tasks/todo.md
tasks/experiments.md
tasks/lessons.md
```

Update after every experiment.

---

# 16. Final Deliverables

Submission must contain:

```
README.md
source code
trained models
evaluation results
screenshots
plots
presentation slides
final report
```

---

# Core Principles

### Scientific Rigor

All claims must be supported by experiments.

### Modularity

Environment, agent, training, and evaluation must remain independent.

### Reproducibility

Experiments must run with configuration files and fixed seeds.

### Simplicity

Prefer minimal implementations that produce measurable improvements.

---

# Expected Outcome

The upgraded AIRS project will become a **scientifically rigorous reinforcement learning system** with:

* proper MDP formulation
* modular architecture
* strong evaluation methodology
* baseline comparisons
* reproducible experiments

This structure will satisfy the **course project requirements and grading criteria** while preserving the originality of the AIRS cybersecurity defense system.
