# AIRS Upgrade – Task Tracker

## Phase 1: Foundation & Fixes
- [x] Create YAML config system (`configs/default.yaml`)
- [x] Fix model loading (.zip handling)
- [x] Separate model directories per algorithm
- [x] Update requirements.txt with new dependencies (mlflow added)

## Phase 2: Environment Realism
- [x] Multi-stage attack pipeline (recon → exploit → persist)
- [x] Adaptive attacker (uses defender's last K actions)
- [x] Noisy / partial observations
- [x] Operational constraints (budget, cooldown, delayed effects)
- [x] Temporal observation stacking

## Phase 3: Reward Engineering
- [x] Multi-objective reward (security_gain − service_cost − latency_penalty)
- [x] Constrained RL elements (downtime threshold penalty)
- [x] Configurable reward weights via YAML

## Phase 4: Evaluation Upgrades
- [x] Baseline policies (noop, random, rule-based)
- [x] Multi-seed evaluation with 95% CI
- [x] New metrics (detection delay, FPR, downtime, cost)
- [x] Statistical comparisons (t-test, bootstrap CI)
- [x] Out-of-distribution tests

## Phase 5: Training Improvements
- [x] VecEnv parallel training
- [x] Curriculum learning
- [x] Best-model checkpointing
- [x] Early stopping

## Phase 6: MLOps & Reproducibility
- [x] YAML config loader
- [x] Seed management & deterministic flags
- [x] Experiment logging to CSV
- [x] MLflow experiment tracking integrated
- [x] Dockerfile

## Phase 7: Visualization
- [x] Extended visualizer (detection delay, comparison plots)
- [x] Streamlit dashboard

## Phase 8: Repository Restructuring
- [x] Create `src/` modular architecture
- [x] `src/environment/intrusion_env.py` — integrated environment module
- [x] `src/agent/rl_agent.py` — AIRSAgent wrapper
- [x] `src/training/train_agent.py` — training pipeline with MLflow
- [x] `src/evaluation/evaluate_agent.py` — evaluation script
- [x] `src/evaluation/metrics.py` — metrics and statistical tests
- [x] `src/baselines/` — separate baseline policy files
- [x] `src/visualization/` — visualizer and Streamlit dashboard
- [x] `configs/training_config.yaml` — centralized config
- [x] Obsidian vault documentation

## Phase 9: Validation
- [x] Run full training pipeline
- [x] Run evaluation
- [x] Verify no runtime errors
