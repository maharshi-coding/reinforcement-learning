# AIRS – Training Strategy

## Algorithm Selection: DQN vs PPO vs A2C vs RecurrentPPO

| Criterion             | DQN                              | PPO                              | A2C                  | RecurrentPPO                    |
|-----------------------|----------------------------------|----------------------------------|----------------------|---------------------------------|
| Action space          | Discrete ✓                      | Continuous or Discrete ✓        | Discrete ✓          | Discrete ✓                     |
| Sample efficiency     | Higher (off-policy replay)       | Lower (on-policy)                | Lower (on-policy)    | Lower (on-policy + LSTM)        |
| Stability             | Stable with target network       | Stable with clipping             | Moderate             | Stable with clipping + memory   |
| Temporal reasoning    | Single observation               | Single observation               | Single observation   | **LSTM hidden state** ✓        |
| Cybersecurity fit     | Value-based: Q(s,a) per action ✓ | Policy-gradient: good for stoch  | Fast convergence     | Memory across attack phases    |
| Recommended           | **Primary (DQN)**                | Secondary (comparison)           | Quick experiments    | Multi-stage attack reasoning   |

**Primary choice: DQN** – the discrete action space and value-based objective
make DQN the natural fit. **RecurrentPPO** is recommended for multi-stage
attacks where temporal memory helps track attack phase transitions.

---

## DQN Hyperparameters

| Parameter                  | Value      | Justification                                   |
|----------------------------|------------|-------------------------------------------------|
| learning_rate              | 1e-3       | Standard for Adam optimizer in SB3              |
| buffer_size                | 50,000     | Sufficient for 200-step episodes × 250 episodes|
| learning_starts            | 500        | Avoid learning from empty/noisy buffer          |
| batch_size                 | 64         | Standard mini-batch                             |
| gamma (discount)           | 0.99       | Long planning horizon (see reward_design.md)    |
| train_freq                 | 4          | Update every 4 env steps                        |
| target_update_interval     | 500        | Stabilise Bellman targets                       |
| exploration_fraction       | 0.30       | 30% of timesteps for epsilon-greedy exploration |
| exploration_final_eps      | 0.05       | 5% final exploration floor                      |
| policy architecture        | MlpPolicy  | 64×64 hidden layers (SB3 default)               |

---

## PPO Hyperparameters (Tuned)

| Parameter      | Value  |
|----------------|--------|
| learning_rate  | 2.5e-4 |
| n_steps        | 1024   |
| batch_size     | 128    |
| n_epochs       | 15     |
| gamma          | 0.99   |
| gae_lambda     | 0.95   |
| clip_range     | 0.2    |
| ent_coef       | 0.01   |
| vf_coef        | 0.5    |
| net_arch       | pi: [256,256], vf: [256,256] |

---

## A2C Hyperparameters

| Parameter            | Value  |
|----------------------|--------|
| learning_rate        | 7e-4   |
| n_steps              | 256    |
| gamma                | 0.99   |
| gae_lambda           | 0.95   |
| ent_coef             | 0.01   |
| normalize_advantage  | true   |
| net_arch             | pi: [256,256], vf: [256,256] |

---

## RecurrentPPO Hyperparameters (LSTM Policy)

Requires `sb3-contrib>=2.0.0`.
Uses `MlpLstmPolicy` — maintains hidden state across steps within an episode.

| Parameter          | Value  |
|--------------------|--------|
| learning_rate      | 2.5e-4 |
| n_steps            | 1024   |
| batch_size         | 128    |
| n_epochs           | 10     |
| gamma              | 0.99   |
| gae_lambda         | 0.95   |
| clip_range         | 0.2    |
| lstm_hidden_size   | 128    |
| n_lstm_layers      | 1      |
| net_arch           | pi: [256], vf: [256] |

---

## Training Phases

### Phase 1 – Static Attacker (brute_force, medium)
- Timesteps: 50,000
- Goal: agent learns basic threat response
- Expected convergence: within 200–300 episodes

### Phase 2 – Flood Attacker (flood, medium)
- Timesteps: 50,000
- Goal: agent learns differentiated response to flood vs brute-force

### Phase 3 – Adaptive Attacker (adaptive, medium)
- Timesteps: 100,000
- Goal: agent generalises across switching attack strategies
- Expected: slower convergence; more exploration needed

### Phase 4 – High Intensity (adaptive, high)
- Timesteps: 100,000
- Goal: stress-test the policy under maximum attack pressure

---

## Exploration Strategy

DQN uses epsilon-greedy:
- ε starts at 1.0 (pure random)
- Decays linearly over 30% of total_timesteps
- Floor at ε = 0.05 (5% random actions maintained)

This ensures the agent sees a wide variety of states before committing to a policy.

---

## Curriculum Learning (Implemented)

1. Start with `brute_force, low`
2. Advance to `brute_force, medium` → `brute_force, high`
3. Then `flood, low` → `flood, high`
4. Finally `adaptive, medium` → `adaptive, high`

Configurable in `configs/default.yaml` under `training.curriculum`.
Method: `AIRSAgent.train_curriculum(stages)`.

---

## Self-Play Training (New)

Alternates between training a PPO defender and a PPO attacker:

1. **Round N**: Freeze attacker → train defender for `defender_steps`
2. **Round N**: Freeze defender → train attacker for `attacker_steps`
3. Repeat for `rounds` iterations

The attacker has 6 strategies (stealth, brute force, flood, full assault,
balanced, evasion) and learns to exploit defender weaknesses. The defender
becomes progressively more robust.

```bash
# Self-play training
python scripts/train_self_play.py --rounds 10 --defender_steps 20000 --attacker_steps 20000
```

---

## Training Commands

```bash
# Train DQN on adaptive attacker
python scripts/train.py --algorithm dqn --attack_mode adaptive --timesteps 100000

# Train PPO for comparison
python scripts/train.py --algorithm ppo --attack_mode adaptive --timesteps 100000

# Train with curriculum learning
python scripts/train.py --algorithm ppo --curriculum --config configs/default.yaml

# Self-play adversarial training
python scripts/train_self_play.py --rounds 10 --defender_steps 20000 --attacker_steps 20000

# Train RecurrentPPO (requires sb3-contrib)
python scripts/train.py --algorithm recurrent_ppo --attack_mode multi_stage --timesteps 100000
```

---

## Convergence Criteria

Training is considered converged when:
1. 20-episode moving average reward increases monotonically for 50 episodes
2. Attack success rate drops below 30%
3. False positive rate (defensive actions under low threat) is < 15%
4. Action distribution is non-degenerate (no single action > 80% frequency)

---

## Debugging Checklist

If training is unstable:
- [ ] Check reward scale – is R_threat dominating all other terms?
- [ ] Inspect action distribution – is the agent stuck on one action?
- [ ] Verify observation normalisation – are all values in [0,1]?
- [ ] Lower learning_rate (try 3e-4 or 1e-4)
- [ ] Increase buffer_size or learning_starts
- [ ] Check for reward hacking (always-isolate or always-no-op)
- [ ] Visualise threat_timeline to confirm environment is generating varied states
