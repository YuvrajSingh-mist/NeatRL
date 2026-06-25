# TD3 (Twin Delayed Deep Deterministic Policy Gradient)

TD3 is an improved version of DDPG that addresses function approximation error in actor-critic methods using twin Q-networks, delayed policy updates, and target policy smoothing.

Note: TD3 is designed for continuous action spaces. For discrete action environments, consider using DQN instead.

## Features

- **Standard TD3**: For environments with vector observations (e.g., HalfCheetah, BipedalWalker)
- **CNN TD3**: For environments with image observations (e.g., Atari games)
- **Twin Q-Networks**: Reduces overestimation bias using clipped double Q-learning
- **Delayed Policy Updates**: Updates actor less frequently than critics for stability
- **Target Policy Smoothing**: Adds noise to target actions to prevent exploitation
- **Experience Replay**: Stabilizes training with off-policy learning
- **W&B Integration**: Built-in logging and video recording

## Key Improvements over DDPG

1. **Twin Q-Networks**: Uses two critic networks and takes the minimum Q-value to reduce overestimation
2. **Delayed Policy Updates**: Updates the actor and target networks less frequently than the critic networks
3. **Target Policy Smoothing**: Adds noise to target actions when computing target Q-values

## Usage

### Standard TD3 (Vector Observations)

```python
from neatrl.td3_mlp import train_td3

train_td3(
    env_id="HalfCheetah-v5",
    total_timesteps=1000000,
    use_wandb=True
)
```

### CNN TD3 (Image Observations)

```python
from neatrl.td3_cnn import train_td3_cnn

train_td3_cnn(
    env_id="CarRacing-v2",
    total_timesteps=100000,
    use_wandb=True
)
```

## Network Architectures

### Actor Network (Standard)
- Input: State vector
- Hidden: 256 -> 256 units with Mish activation
- Output: Continuous actions (tanh activation)

### Twin Q-Networks (Standard)
Both Q1 and Q2 share the same architecture:
- State path: 256 units
- Action path: 256 units
- Combined: 512 -> 512 -> 256 -> 1 units with Mish activation

### Actor Network (CNN)
- Conv layers: 32@8x8 -> 64@4x4 -> 64@3x3
- FC layers: 3136 -> 512 -> action_dim
- Output: Continuous actions (tanh activation)

### Twin Q-Networks (CNN)
Both Q1 and Q2 share the same architecture:
- State conv: 32@8x8 -> 64@4x4 -> 64@3x3
- State FC: 3136 -> 512
- Action FC: action_dim -> 512
- Combined: 1024 -> 512 -> 1

## Configuration Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `env_id` | str | `"HalfCheetah-v5"` | Gymnasium environment ID |
| `env` | gym.Env | `None` | Optional pre-created environment instance |
| `grid_env` | bool | `False` | Whether to apply one-hot encoding for discrete state spaces |
| `atari_wrapper` | bool | `False` | Whether to apply Atari preprocessing |
| `env_wrapper` | Callable | `None` | Optional custom environment wrapper |
| `n_envs` | int | `1` | Number of parallel environments |
| `total_timesteps` | int | `1000000` | Total number of environment steps |
| `learning_rate` | float | `3e-4` | Learning rate for actor and critic optimizers |
| `buffer_size` | int | `100000` | Maximum replay buffer size |
| `gamma` | float | `0.99` | Discount factor |
| `tau` | float | `0.005` | Soft update coefficient for target networks |
| `batch_size` | int | `256` | Batch size for training |
| `learning_starts` | int | `25000` | Steps before training begins |
| `train_frequency` | int | `2` | Delayed policy update frequency |
| `target_network_frequency` | int | `1` | How often to perform soft updates on target networks |
| `policy_noise` | float | `0.2` | Std of Gaussian noise added to target policy |
| `exploration_noise` | float | `0.1` | Std of Gaussian noise added during exploration |
| `noise_clip` | float | `0.5` | Maximum absolute value for clipping noise |
| `low` | float | `-1.0` | Lower bound for action space |
| `high` | float | `1.0` | Upper bound for action space |
| `normalize_obs` | bool | `False` | Whether to normalize observations |
| `normalize_reward` | bool | `False` | Whether to normalize rewards |
| `seed` | int | `42` | Random seed |
| `exp_name` | str | `"TD3-Experiment"` | Experiment name |
| `use_wandb` | bool | `True` | Whether to log to Weights & Biases |
| `wandb_project` | str | `"cleanRL"` | W&B project name |
| `wandb_entity` | str | `""` | W&B team/username |
| `capture_video` | bool | `True` | Whether to capture evaluation videos |
| `eval_every` | int | `500` | Evaluation frequency (steps) |
| `save_every` | int | `10000` | Model save frequency (steps) |
| `num_eval_episodes` | int | `10` | Number of evaluation episodes |
| `max_grad_norm` | float | `0.0` | Maximum gradient norm for clipping |
| `log_gradients` | bool | `True` | Whether to log gradient norms |
| `device` | str | `"cpu"` | Training device ("auto", "cpu", "cuda", etc.) |
| `actor_class` | nn.Module | `ActorNet` | Custom actor network class |
| `q_network_class` | nn.Module | `QNet` | Custom Q-network class |

## Example Configurations

### Quick Test (Pendulum)

```python
train_td3(
    env_id="Pendulum-v1",
    total_timesteps=50000,
    learning_starts=1000,
    eval_every=5000
)
```

### MuJoCo Training

```python
train_td3(
    env_id="HalfCheetah-v5",
    total_timesteps=1000000,
    learning_rate=3e-4,
    batch_size=256,
    buffer_size=1000000,
    policy_noise=0.2,
    noise_clip=0.5,
    exploration_noise=0.1,
    train_frequency=2,
    target_network_frequency=1,
    use_wandb=True
)
```

### CNN-based Training (Car Racing)

```python
train_td3_cnn(
    env_id="CarRacing-v2",
    total_timesteps=100000,
    learning_rate=1e-4,
    batch_size=32,
    learning_starts=5000,
    atari_wrapper=False
)
```

## Custom Networks

```python
import torch.nn as nn

class CustomActor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, action_dim),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x)

train_td3(
    env_id="HalfCheetah-v5",
    actor_class=CustomActor,
    q_network_class=CustomCritic
)
```

## Tips

1. Start with `exploration_noise=0.1` and adjust based on your environment
2. The default `policy_noise=0.2` works well for most MuJoCo tasks
3. Keep `train_frequency=2` for the delayed policy update
4. Larger buffers (1M+) help for complex tasks but require more memory
5. Collect enough random samples before training (typically 10k-25k)

## Example Scripts

- `run_td3_pendulum.py` - TD3 training on Pendulum
- `run_td3_half_cheetah.py` - TD3 training on HalfCheetah
- `run_td3_bipedal_walker.py` - TD3 training on BipedalWalker
- `run_td3_cnn_car_racing.py` - TD3 CNN training on CarRacing

## Installation

```bash
pip install neatrl

pip install neatrl[atari]      # For CarRacing
pip install neatrl[box2d]      # For BipedalWalker
pip install neatrl[classic]    # For Pendulum
pip install neatrl[mujoco]     # For HalfCheetah
```

## PyPI

For installation and more information, visit [NeatRL on PyPI](https://pypi.org/project/neatrl/)
