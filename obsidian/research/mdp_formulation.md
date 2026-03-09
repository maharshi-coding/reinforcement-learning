# AIRS – MDP Formulation

## Problem Definition

The AIRS cybersecurity defense problem is formulated as a **Markov Decision Process (MDP)** defined by the tuple **(S, A, P, R)**.

## State Space S

The system state at time $t$ encodes network telemetry:

$$s_t = [\text{failed\_logins}, \text{request\_rate}, \text{cpu\_usage}, \text{memory\_usage}, \text{alert\_level}, \text{last\_action}]$$

All features are normalised to $[0, 1]$.

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | failed_logins | Normalised count of failed login attempts |
| 1 | request_rate | Normalised packet/request rate |
| 2 | cpu_usage | CPU utilisation (0–1) |
| 3 | memory_usage | Memory utilisation (0–1) |
| 4 | alert_level | Composite threat score (weighted sum of indicators) |
| 5 | last_action | Previous defensive action (normalised) |

### Temporal Extension

With `temporal_window=N`, the observation becomes a stacked vector of dimension $6N$:

$$s_t = [s_{t-N+1}, s_{t-N+2}, \ldots, s_t]$$

## Action Space A

| Action ID | Name | Description |
|-----------|------|-------------|
| 0 | do_nothing | Observe only; zero cost |
| 1 | block_ip | Block source IP; effective vs brute force |
| 2 | rate_limit | Rate-limit traffic; effective vs floods |
| 3 | isolate_service | Isolate service; max protection, max cost |

## Transition Function P(s' | s, a)

The transition function is stochastic and depends on:

1. **Attacker behaviour** — generates traffic/login patterns based on attack mode
2. **Defender action** — strong actions (block, isolate) reduce attack metrics
3. **System dynamics** — CPU/memory respond to traffic load

Attack modes:
- **brute_force**: High failed logins, moderate traffic
- **flood**: Very high traffic, low logins
- **adaptive**: Switches strategy based on defender history
- **multi_stage**: Reconnaissance → Exploitation → Persistence phases

## Reward Function R(s, a)

Multi-objective reward combining security gain and operational costs:

$$R(s, a) = \underbrace{r_{\text{threat}} \cdot w_{\text{threat}}}_{\text{security gain}} - \underbrace{c_{\text{service}} \cdot w_{\text{service}}}_{\text{disruption cost}} - \underbrace{p_{\text{fp}}}_{\text{false positive}} - \underbrace{p_{\text{ineff}}}_{\text{inaction penalty}} - \underbrace{p_{\text{breach}}}_{\text{breach damage}} - \underbrace{p_{\text{latency}}}_{\text{response delay}} - \underbrace{p_{\text{downtime}}}_{\text{cumulative cost}} + b_{\text{survival}} \cdot (1 - \text{threat})$$

### Breach Damage (added 2026-03-08)

Accumulating penalty when threat is present but agent does nothing:

$$\text{breach\_progress}_{t} = \begin{cases} \min(\text{breach\_progress}_{t-1} + \text{threat}_t, 3.0) & \text{if } a = 0 \\ \max(0, \text{breach\_progress}_{t-1} - r_{\text{reduction}}) & \text{if } a > 0 \end{cases}$$

$$p_{\text{breach}} = \text{breach\_progress}_t \times 1.0$$

This makes passivity non-viable: a "do nothing" policy accumulates escalating penalties.

### Threat Level Computation (updated 2026-03-08)

The threat level uses a nonlinear formula that amplifies the dominant attack signal:

$$\text{dominant} = \max(t, f)$$
$$\text{spike} = \min(\sqrt{\text{dominant}} + 0.3c, 1.0)$$
$$\text{threat}_{\text{raw}} = w^T \cdot [t, f, c, m, \text{spike}]$$
$$\text{threat} = \sqrt{\text{threat}_{\text{raw}}}$$

The sqrt scaling ensures even moderate attacks (medium intensity brute force) produce threat levels ≈ 0.5, making threshold-based penalties actually trigger.

### Reward Table

| Event | Reward |
|-------|--------|
| Attack blocked (threat reduction × weight) | +10 |
| Successful mitigation (high threat, strong action) | +15 |
| False positive (action on low threat) | −5 |
| Service downtime (cumulative cost exceeds threshold) | −10 |
| Attack success (no action at high threat) | −15 |

## Optimisation Objective

The RL policy $\pi_\theta(a|s)$ must maximise:

$$\max_\theta \mathbb{E}_{\pi_\theta} \left[ \sum_{t=0}^{T} \gamma^t R(s_t, a_t) \right]$$

where $\gamma = 0.99$ is the discount factor.

## Policy Architecture

### MLP Policy Network

```
Input:  state vector (6-d or 6N-d with temporal window)
Hidden: 2 × 64 neurons (ReLU activation)
Output: action probabilities π_θ(a|s)
```

### Algorithms

| Algorithm | Type | Key Properties |
|-----------|------|---------------|
| DQN | Value-based | Experience replay, target network, ε-greedy exploration |
| PPO | Policy gradient | Clipped surrogate objective, GAE advantages |
