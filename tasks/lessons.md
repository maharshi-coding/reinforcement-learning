# AIRS – Lessons Learned

## Training
- PPO requires significantly more timesteps than DQN for this small discrete action space
- Curriculum learning (low→medium→high) produces better final reward than flat training
- DQN with 5k timesteps already converges to a consistent block_ip strategy

## Environment
- Multi-stage attack mode needs longer episodes (200 steps) to see all 3 phases
- Noisy observations with std=0.2 don't significantly degrade learned policy performance
- Resource budget constraint forces interesting tradeoffs when budget < episode_length

## Evaluation
- Always compare against baselines — even a simple rule-based policy can be competitive
- Multi-seed evaluation reveals variance invisible in single-seed runs
- Bootstrap CI is more robust than t-test CI when seed count is small (n=5)

## Reward Design
- High false positive rate suggests reward balance may need tuning for specific attack modes
- With default weights, the agent strongly prefers block_ip — may need service cost weight increase
