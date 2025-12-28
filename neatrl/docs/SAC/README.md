# SAC (Soft Actor-Critic)

SAC is an off-policy actor-critic algorithm that maximizes both expected return and entropy, leading to more stochastic and exploratory policies. This implementation includes both standard SAC for vector observations and CNN-based SAC for image observations.

**Note**: SAC works with continuous action spaces and uses entropy regularization for better exploration.

## Features

- **Standard SAC**: For environments with vector observations (e.g., Pendulum, HalfCheetah)
- **CNN SAC**: For environments with image observations (e.g., CarRacing)
- **Entropy Regularization**: Automatic temperature tuning for optimal exploration
- **Twin Q-Networks**: Two Q-networks to reduce overestimation bias
- **Reparameterization Trick**: Stable policy updates with reparameterized sampling
- **W&B Integration**: Built-in logging and video recording
- **Off-policy Learning**: Experience replay for sample efficiency

## Usage

### Standard SAC (Vector Observations)

```python
from neatrl.sac import train_sac

train_sac(
    env_id="HalfCheetah-v5",
    total_timesteps=1000000,
    alpha=0.2,
    autotune_alpha=True,
    use_wandb=True
)
```

### CNN SAC (Image Observations)

```python
from neatrl.sac import train_sac_cnn

train_sac_cnn(
    env_id="CarRacing-v3",
    total_timesteps=1000000,
    alpha=0.2,
    autotune_alpha=True,
    use_wandb=True,
    env_wrapper=car_racing_wrapper,
    actor_class=ActorNetCNN,
    q_network_class=QNetCNN
)
```

## Network Architectures

### Actor Network (Standard)
- Input: State vector
- Hidden: 256 → 256 units with ReLU
- Output: Mean and log standard deviation for Gaussian policy
- Action: Squashed with tanh for bounded actions

### Q-Network (Standard)
- Input: State + Action vectors
- Hidden: 256 → 256 → 256 units with Mish activation
- Output: Single Q-value

### Actor Network (CNN)
- Convolutional layers: 32@8x8, 64@4x4, 64@3x3
- Fully connected: 512 → action_dim (mean/log_std)

### Q-Network (CNN)
- Same convolutional layers as actor
- State features: 512 units
- Action features: action_dim → 512 units
- Combined: 1024 → 512 → 1

## Key Hyperparameters

- `alpha`: Entropy regularization coefficient (0.2 default)
- `autotune_alpha`: Whether to automatically tune alpha (True recommended)
- `target_entropy_scale`: Target entropy multiplier (-1.0 for continuous actions)
- `tau`: Soft update parameter (0.005)
- `policy_frequency`: How often to update policy (1 = every step)
- `learning_starts`: Steps before training begins (5000)

## Algorithm Overview

1. **Collect Experience**: Sample actions from stochastic policy
2. **Update Q-Networks**: Minimize Bellman error with entropy bonus
3. **Update Policy**: Maximize expected Q-value + entropy
4. **Update Alpha**: Tune temperature parameter (if autotune enabled)
5. **Soft Updates**: Gradually update target networks

## Mathematical Formulation

SAC maximizes the expected return while encouraging exploration through entropy:

```
J(π) = Σ E[(r + γ(r' + ...)) + α * H(π(·|s))]
```

Where:
- `H(π(·|s))` is the entropy of the policy
- `α` controls the exploration-exploitation trade-off

## Advantages

- **Sample Efficient**: Off-policy learning with replay buffer
- **Stable Training**: Twin Q-networks prevent overestimation
- **Automatic Exploration**: Entropy regularization adapts to task complexity
- **Continuous Actions**: Naturally handles continuous action spaces

## Environment Compatibility

### Recommended Environments
- **Continuous Control**: HalfCheetah, Ant, Walker2d, Humanoid
- **Classic Control**: Pendulum, MountainCarContinuous
- **Image-based**: CarRacing, Atari games (with CNN variant)

### Environment Wrappers
```python
def car_racing_wrapper(env):
    # Preprocessing for CarRacing
    return env
```
