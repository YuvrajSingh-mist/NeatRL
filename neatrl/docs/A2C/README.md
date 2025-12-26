# A2C (Advantage Actor-Critic)

A2C is an on-policy actor-critic algorithm that uses advantage estimates to reduce variance in policy gradient updates. This implementation includes both standard A2C for vector observations and CNN-based A2C for image observations.

**Note**: A2C works with both discrete and continuous action spaces, making it versatile for various environments.

## Features

- **Standard A2C**: For environments with vector observations (e.g., Acrobot, CartPole)
- **CNN A2C**: For environments with image observations (e.g., CarRacing, Atari games)
- **Advantage Estimation**: Uses Monte Carlo returns for advantage calculation
- **Separate Optimizers**: Independent optimization for actor and critic networks
- **W&B Integration**: Built-in logging and video recording
- **Episode-based Updates**: Pure on-policy learning with full episode rollouts

## Usage

### Standard A2C (Vector Observations)

```python
from neatrl.a2c import train_a2c

train_a2c(
    env_id="Acrobot-v1",
    total_timesteps=500000,
    lr=3e-4,
    use_wandb=True
)
```

### CNN A2C (Image Observations)

```python
from neatrl.a2c import train_a2c_cnn

train_a2c_cnn(
    env_id="CarRacing-v3",
    total_timesteps=500000,
    lr=3e-4,
    use_wandb=True,
    env_wrapper=car_racing_wrapper,
    actor_class=ActorNet,
    critic_class=CriticNet
)
```

## Network Architectures

### Actor Network (Standard)
- Input: State vector
- Hidden: 128 → 128 → 128 units with Tanh activation
- Output: Action probabilities (softmax for discrete) or action values (for continuous)

### Critic Network (Standard)
- Input: State vector
- Hidden: 128 → 128 → 128 units with Tanh activation
- Output: Single value estimate

### Actor Network (CNN)
- Input: Stacked frames (4, 84, 84)
- Conv layers: 32 → 64 → 64 filters
- FC: 512 units
- Output: Action probabilities/values

### Critic Network (CNN)
- Input: Stacked frames (4, 84, 84)
- Conv layers: 32 → 64 → 64 filters
- FC: 512 units
- Output: Single value estimate

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lr` | 3e-4 | Learning rate for both actor and critic |
| `gamma` | 0.99 | Discount factor for rewards |
| `max_steps` | 2048 | Maximum steps per episode rollout |
| `max_grad_norm` | 0.5 | Gradient clipping threshold |
| `update_epochs` | 1 | Number of update epochs (A2C uses 1) |

## Examples

### Acrobot Environment
```bash
cd docs/A2C
python run_a2c_acrobot.py
```

### CarRacing with CNN
```bash
cd docs/A2C
python run_a2c_cnn_car_racing.py
```

## Algorithm Details

A2C uses the following update rule:

**Policy Loss**:
```
L_policy = -mean(log_prob * advantage)
```

**Value Loss**:
```
L_value = MSE(value_estimate, monte_carlo_return)
```

**Advantages**:
```
A(s,a) = Q(s,a) - V(s) = Return - V(s)
```

Where returns are computed using Monte Carlo estimation from full episode rollouts.

## Performance Tips

1. **Learning Rate**: Start with 3e-4 and adjust based on convergence
2. **Gradient Clipping**: Use 0.5 for stable training
3. **Environment-Specific Tuning**: 
   - Acrobot: No observation normalization needed
   - CarRacing: Use frame stacking (4 frames) and CNN architecture
4. **Batch Size**: A2C is on-policy, so it processes full episodes
5. **Exploration**: A2C naturally explores through its stochastic policy

## Differences from PPO

Unlike PPO, A2C:
- Does NOT use clipped surrogate objective
- Does NOT use entropy bonus
- Uses simpler advantage estimation (Monte Carlo returns)
- Updates after each episode (on-policy)
- Has separate optimizers for actor and critic

This makes A2C simpler and faster per update, but potentially less sample-efficient than PPO.

## Common Issues

1. **Unstable Training**: Reduce learning rate or increase gradient clipping
2. **Poor Performance on CarRacing**: Ensure frame stacking is working correctly and CNN architecture matches input shape
3. **Slow Convergence**: Try increasing learning rate or adjusting gamma

## References

- [Mnih et al., 2016: Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
- [OpenAI Spinning Up: A2C](https://spinningup.openai.com/en/latest/algorithms/vpg.html)
