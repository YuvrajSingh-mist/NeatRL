# NeatRL Documentation - DQN

This guide shows you how to use NeatRL's DQN implementation.

## Quick Start with DQN

### Basic Training

```python
from neatrl import train_dqn

model = train_dqn(
    env_id="CartPole-v1",
    total_timesteps=10000,
    seed=42
)
```

### Training with Experiment Tracking

```python
from neatrl import train_dqn

model = train_dqn(
    env_id="CartPole-v1",
    total_timesteps=50000,
    seed=42,
    capture_video=True,
    use_wandb=True,
    wandb_project="my-rl-experiments",
    exp_name="dqn-cartpole-v1"
)
```

### Atari Games Training

```python
import torch.nn as nn
from neatrl import train_dqn

class AtariQNet(nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.conv1 = nn.Conv2d(state_space, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, action_space)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = train_dqn(
    env_id="BreakoutNoFrameskip-v4",
    total_timesteps=1000000,
    seed=42,
    learning_rate=2.5e-4,
    buffer_size=100000,
    gamma=0.99,
    batch_size=32,
    atari_wrapper=True,
    custom_agent=AtariQNet(4, 4),
    use_wandb=True,
    wandb_project="atari-experiments",
    exp_name="dqn-breakout"
)
```

## Function Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `env_id` | str | `"BreakoutNoFrameskip-v4"` | Gymnasium environment ID |
| `total_timesteps` | int | `20000` | Total number of environment steps |
| `seed` | int | `42` | Random seed |
| `learning_rate` | float | `2.5e-4` | Learning rate for the Adam optimizer |
| `buffer_size` | int | `10000` | Size of the replay buffer |
| `gamma` | float | `0.99` | Discount factor |
| `tau` | float | `1.0` | Target network update rate (1.0 = hard update) |
| `target_network_frequency` | int | `50` | How often to update the target network |
| `batch_size` | int | `128` | Batch size for training |
| `start_e` | float | `1.0` | Initial epsilon for exploration |
| `end_e` | float | `0.05` | Final epsilon for exploration |
| `exploration_fraction` | float | `0.5` | Fraction of training steps for epsilon decay |
| `learning_starts` | int | `1000` | Steps before training begins |
| `train_frequency` | int | `10` | How often to train the network |
| `max_grad_norm` | float | `0.0` | Maximum gradient norm for clipping (0.0 to disable) |
| `capture_video` | bool | `False` | Whether to record training videos |
| `use_wandb` | bool | `False` | Whether to log to Weights & Biases |
| `wandb_project` | str | `"cleanRL"` | W&B project name |
| `wandb_entity` | str | `""` | W&B username/entity |
| `exp_name` | str | `"DQN"` | Experiment name |
| `eval_every` | int | `1000` | Evaluation frequency (steps) |
| `save_every` | int | `1000` | Model save frequency (steps) |
| `atari_wrapper` | bool | `False` | Whether to apply Atari preprocessing |
| `custom_agent` | nn.Module | `None` | Custom Q-network class/instance |
| `num_eval_eps` | int | `10` | Number of evaluation episodes |
| `n_envs` | int | `4` | Number of parallel environments |
| `device` | str | `"cpu"` | Training device ("cpu", "cuda", "mps") |
| `grid_env` | bool | `False` | Enable one-hot encoding for discrete states |

## Supported Environments

### Classic Control
- `CartPole-v1`
- `MountainCar-v0`
- `Acrobot-v1`

### Grid Environments

```python
from neatrl import train_dqn

model = train_dqn(
    env_id="FrozenLake-v1",
    total_timesteps=50000,
    seed=42,
    grid_env=True,
    use_wandb=True,
    exp_name="dqn-frozenlake"
)
```

### Box2D
- `LunarLander-v2`
- `LunarLander-v3`

### Toy Text
- `FrozenLake-v1`
- `Taxi-v3`
- `CliffWalking-v0`

### Atari Games
- `BreakoutNoFrameskip-v4`
- `ALE/Pong-v5`
- `ALE/SpaceInvaders-v5`

## Experiment Tracking with Weights & Biases

```python
model = train_dqn(
    env_id="CartPole-v1",
    total_timesteps=50000,
    use_wandb=True,
    wandb_project="my-rl-project",
    wandb_entity="your-username",
    exp_name="cartpole-experiment"
)
```

What gets logged:
- Episode returns, episode lengths, epsilon values, training loss
- Training progress videos (if `capture_video=True`)
- All training configuration

## Video Recording

```python
model = train_dqn(
    env_id="CartPole-v1",
    total_timesteps=50000,
    capture_video=True,
    use_wandb=True,
    upload_every=100
)
```

## Installation

```bash
pip install neatrl

pip install neatrl[atari]      # For Atari games
pip install neatrl[box2d]      # For LunarLander
pip install neatrl[classic]    # For CartPole, FrozenLake, MountainCar, Acrobot
```

## PyPI

For installation and more information, visit [NeatRL on PyPI](https://pypi.org/project/neatrl/)
