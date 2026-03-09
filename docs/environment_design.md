# AIRS – Environment Design

## Gymnasium Compliance

`NetworkSecurityEnv` inherits from `gymnasium.Env` and implements:
- `reset(seed, options)` → `(obs, info)`
- `step(action)` → `(obs, reward, terminated, truncated, info)`
- `render()` → no-op (headless training)

---

## State Space

The 6-dimensional observation vector is fully normalised to [0, 1]:

| Index | Feature        | Raw Source                      | Normalisation        | Justification                                  |
|-------|----------------|---------------------------------|----------------------|------------------------------------------------|
| 0     | traffic_rate   | AttackSimulator → packets/sec   | ÷ 1000 (max rate)    | Primary flood indicator                        |
| 1     | failed_logins  | AttackSimulator → count/step    | ÷ 300 (max logins)   | Primary brute-force indicator                  |
| 2     | cpu_usage      | psutil or simulated             | already [0,1]        | Correlates with attack load on server          |
| 3     | memory_usage   | psutil or simulated             | already [0,1]        | Memory exhaustion risk under sustained attack  |
| 4     | threat_level   | SystemMonitor.compute_threat    | already [0,1]        | Compact composite summary for agent            |
| 5     | last_action    | Previous step's action          | ÷ 3 (num_actions-1)  | Temporal context; discourages oscillation      |

### Feature Scaling Rationale
- All features are clipped to [0, 1] to prevent gradient explosion in the DQN network.
- The normalisation constants (1000 for traffic, 300 for logins) reflect the upper bound
  of the `high` intensity attacker, ensuring full coverage of the observable range.
- Including `last_action` as a feature gives the agent one step of memory, helping it
  avoid flip-flopping between actions.

---

## Action Space

`Discrete(4)`:

| ID | Name            | Real-World Meaning                                          | Risk Trade-off                                       |
|----|-----------------|-------------------------------------------------------------|------------------------------------------------------|
| 0  | no_op           | Observe only; no defensive action                           | Zero cost but threat accumulates                     |
| 1  | block_ip        | Firewall rule to drop traffic from suspicious source        | May block legitimate users (false positive risk)     |
| 2  | rate_limit      | Throttle inbound connection rate from flagged sources       | Moderate disruption; softer than full block          |
| 3  | isolate_service | Take the targeted service offline or into maintenance mode  | Guaranteed threat stop; maximum service disruption   |

### Cost of Each Action
| Action | Threat Reduction (base) | Service Disruption (base) | Success Prob |
|--------|-------------------------|---------------------------|--------------|
| no_op  | 0%                      | 0%                        | 100%         |
| block  | 55%                     | 10%                       | 90%          |
| limit  | 40%                     | 15%                       | 80%          |
| isolate| 80%                     | 40%                       | 85%          |

**Stochastic Outcomes (new)**: Actions may fail based on their success probability.
On failure, threat reduction is reduced to 10% of normal (configurable via
`failure_residual`), but service cost is still incurred. This models real-world
unreliability of defensive measures. The `ActionOutcome.success` field and
`info["action_success"]` expose the result.

---

## Episode Dynamics

- **Episode length**: 200 steps (truncated, never terminated mid-episode)
- **Reset**: fresh `AttackSimulator` internal state; last_action reset to 0
- **Attacker response to defense**: the simulator reduces attack intensity
  proportionally to the action strength, modelling an adversary that backs off
  temporarily when strongly blocked

---

## Termination Conditions

| Condition               | Triggered By | Notes                          |
|-------------------------|--------------|--------------------------------|
| `truncated = True`      | step ≥ 200   | Normal end of episode          |
| `terminated = False`    | never        | No catastrophic failure state  |

This design keeps episodes fixed-length, which stabilises DQN training by
ensuring consistent return horizons.

---

## Observation Normalization Details

```python
obs[0] = traffic_rate  / 1000.0          # clip to [0,1]
obs[1] = failed_logins / 300.0           # clip to [0,1]
obs[2] = cpu_usage                       # [0,1] from psutil
obs[3] = memory_usage                    # [0,1] from psutil
obs[4] = threat_level                    # [0,1] from SystemMonitor
obs[5] = last_action / (num_actions - 1) # [0,1] i.e. {0, 0.33, 0.67, 1.0}
```

All values are additionally `np.clip`-ped to [0,1] before being returned.

---

## Dimensionality

- 6 features × 1 float32 each = 24 bytes per observation
- Minimal memory footprint; compatible with any MlpPolicy architecture
- With `temporal_window=N`, observation is (6×N,) — stacking last N timesteps
- Can be extended to 10+ features (e.g., port scan count, geo-IP flag) without
  architectural changes

---

## Additional Realism Knobs

| Feature               | Config Key              | Default | Effect                              |
|-----------------------|-------------------------|---------|-------------------------------------|
| Noisy observations    | `noisy_observations`    | false   | Gaussian sensor noise on obs        |
| Partial observability | `partial_observability`  | false   | Randomly mask features              |
| Action cooldown       | `action_cooldown`       | 0       | Min steps between non-noop actions  |
| Delayed effects       | `delayed_effect_steps`  | 0       | Action effect delay                 |
| Resource budget       | `resource_budget`       | null    | Max non-noop actions per episode    |
| Temporal window       | `temporal_window`       | 1       | Stack N past observations           |
| Stochastic actions    | `stochastic_actions`    | true    | Actions may fail (see above)        |
