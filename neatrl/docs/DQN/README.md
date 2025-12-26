# 🎯 NeatRL Documentation

Welcome to the NeatRL documentation! This guide shows you how to use NeatRL's reinforcement learning algorithms, with a focus on practical examples and best practices.

## 🚀 Quick Start with DQN

### Basic Training

Train a DQN agent on CartPole in just a few lines:

```python
from neatrl import train_dqn

# Train DQN on CartPole
model = train_dqn(
    env_id="CartPole-v1",
    total_timesteps=10000,
    seed=42
)

print("Training completed! 🎉")
```

### Training with Experiment Tracking

Enable Weights & Biases for experiment tracking and video recording:

```python
from neatrl import train_dqn

model = train_dqn(
    env_id="CartPole-v1",
    total_timesteps=50000,
    seed=42,
    capture_video=True,        # Record training videos
    use_wandb=True,           # Enable W&B logging
    wandb_project="my-rl-experiments",
    exp_name="dqn-cartpole-v1"
)
```

### Atari Games Training

Train DQN on Atari games like Breakout with convolutional networks:

```python
import torch.nn as nn
from neatrl import train_dqn

# Define convolutional Q-network for Atari
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

# Train on Breakout
model = train_dqn(
    env_id="BreakoutNoFrameskip-v4",
    total_timesteps=1000000,
    seed=42,
    learning_rate=2.5e-4,
    buffer_size=100000,
    gamma=0.99,
    batch_size=32,
    atari_wrapper=True,  # Apply Atari preprocessing
    custom_agent=AtariQNet(4, 4),  # 4 stacked frames, 4 actions
    use_wandb=True,
    wandb_project="atari-experiments",
    exp_name="dqn-breakout"
)

```

## 🔧 Function Arguments

The `train_dqn` function accepts the following arguments for customizing your DQN training:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `env_id` | str | `"BreakoutNoFrameskip-v4"` | Gymnasium environment ID to train on |
| `total_timesteps` | int | `20000` | Total number of environment steps to train for |
| `seed` | int | `42` | Random seed for reproducibility |
| `learning_rate` | float | `2.5e-4` | Learning rate for the Adam optimizer |
| `buffer_size` | int | `10000` | Size of the replay buffer |
| `gamma` | float | `0.99` | Discount factor for future rewards |
| `tau` | float | `1.0` | Target network update rate (1.0 = hard update, <1.0 = soft update) |
| `target_network_frequency` | int | `50` | How often to update the target network (in steps) |
| `batch_size` | int | `128` | Batch size for training |
| `start_e` | float | `1.0` | Initial epsilon for ε-greedy exploration |
| `end_e` | float | `0.05` | Final epsilon for ε-greedy exploration |
| `exploration_fraction` | float | `0.5` | Fraction of training steps for epsilon decay |
| `learning_starts` | int | `1000` | Number of steps to collect before starting training |
| `train_frequency` | int | `10` | How often to train the network (in steps) |
| `max_grad_norm` | float | `0.0` | Maximum gradient norm for clipping (0.0 to disable) |
| `capture_video` | bool | `False` | Whether to record training videos |
| `use_wandb` | bool | `False` | Whether to log to Weights & Biases |
| `wandb_project` | str | `"cleanRL"` | W&B project name |
| `wandb_entity` | str | `""` | W&B username/entity |
| `exp_name` | str | `"DQN"` | Experiment name for logging |
| `eval_every` | int | `1000` | Frequency of evaluation during training (in steps) |
| `save_every` | int | `1000` | Frequency of saving model checkpoints (in steps) |
| `atari_wrapper` | bool | `False` | Whether to apply Atari preprocessing wrappers |
| `custom_agent` | nn.Module | `None` | Custom Q-network class/instance (overrides default) |
| `num_eval_eps` | int | `10` | Number of episodes for evaluation |
| `n_envs` | int | `4` | Number of parallel environments for vectorized training |
| `capture_video` | bool | `False` | Whether to record evaluation videos |
| `device` | str | `"cpu"` | Device for training ("cpu", "cuda", "mps") |

## 🎮 Supported Environments

DQN works with any Gymnasium environment. Here are some popular choices:

### Classic Control
- `CartPole-v1` - Balance a pole on a cart
- `MountainCar-v0` - Drive up a hill
- `Acrobot-v1` - Swing up a two-link robot

### Grid Environments
NeatRL now supports grid-based environments with automatic one-hot encoding for discrete states!

**Example: FrozenLake**

```python
from neatrl import train_dqn

# Train DQN on FrozenLake with automatic one-hot encoding
model = train_dqn(
    env_id="FrozenLake-v1",
    total_timesteps=50000,
    seed=42,
    grid_env=False,  # Enable one-hot encoding for discrete states
    use_wandb=True,
    wandb_project="grid-experiments",
    exp_name="dqn-frozenlake"
)
```

**Example: Taxi**

```python
from neatrl import train_dqn

# Train DQN on Taxi with automatic one-hot encoding
model = train_dqn(
    env_id="Taxi-v3",
    total_timesteps=50000,
    seed=42,
    grid_env=False,  # Enable one-hot encoding for discrete states
    use_wandb=True,
    wandb_project="grid-experiments",
    exp_name="dqn-taxi"
)
```

The `grid_env=False` parameter automatically applies one-hot encoding to discrete state observations, making them suitable for neural network input.

### Box2D
- `LunarLander-v2` - Land a spacecraft safely
- `LunarLander-v3` - Land a spacecraft safely (updated version)
- `CarRacing-v2` - Race a car around a track if discrete action space is chosen

## Toy Text

- `FrozenLake-v1` - Navigate a slippery frozen lake
- `Taxi-v3` - Pick up and drop off passengers
- `CliffWalking-v0` - Avoid cliffs while walking


### Atari Games
- `BreakoutNoFrameskip-v4` - Breakout game
- `ALE/Pong-v5` - Classic Pong game
- `ALE/SpaceInvaders-v5` - Space Invaders
- and many more...

## 📊 Experiment Tracking with Weights & Biases

### Setting up W&B

1. Install Weights & Biases:
```bash
pip install wandb
```

2. Login to W&B:
```bash
wandb login
```

3. Train with logging:
```python
model = train_dqn(
    env_id="CartPole-v1",
    total_timesteps=50000,
    use_wandb=True,
    wandb_project="my-rl-project",
    wandb_entity="your-username",  # Optional: your W&B username
    exp_name="cartpole-experiment"
)
```

### What gets logged:

- **Metrics**: Episode returns, episode lengths, epsilon values, training loss
- **Videos**: Training progress videos (recorded every 100 steps by default)
- **Hyperparameters**: All training configuration
- **System info**: Hardware usage, training time

## 🎥 Video Recording

NeatRL can automatically record and upload training videos:

```python
model = train_dqn(
    env_id="CartPole-v1",
    total_timesteps=50000,
    capture_video=True,     # Enable video recording
    use_wandb=True,         # Upload to W&B
    upload_every=100        # Upload frequency (steps)
)
```

Videos are:
- Recorded every 100 training steps (configurable)
- Uploaded directly to W&B (no local storage)
- Automatically cleaned up after upload


## 📚 Examples

Check out these example scripts:

- [`run_dqn_cartpole.py`](./run_dqn_cartpole.py) - Basic DQN training on CartPole
- [`run_dqn_frozenlake.py`](./run_dqn_frozenlake.py) - DQN training on FrozenLake (with one-hot encoding)
- [`run_dqn_mountaincar.py`](./run_dqn_mountaincar.py) - DQN training on MountainCar
- [`run_dqn_acrobot.py`](./run_dqn_acrobot.py) - DQN training on Acrobot
- [`run_dqn_lunarlander.py`](./run_dqn_lunarlander.py) - DQN training on LunarLander
- [`run_dqn_carracing.py`](./run_dqn_carracing.py) - DQN training on CarRacing (discrete)
- [`run_dqn_atari.py`](./run_dqn_atari.py) - DQN training on Atari-Breakout

## Installation

```bash
# Install base package
pip install neatrl

# Install extras based on environments you want to use
pip install neatrl[atari]      # For CarRacing, Atari-Breakout
pip install neatrl[box2d]      # For LunarLander
pip install neatrl[classic]    # For CartPole, FrozenLake, MountainCar, Acrobot

# Or install all extras at once
pip install neatrl[atari,box2d,classic]
```

---

Happy training! 🚀

## PyPI

For installation and more information, visit [NeatRL on PyPI](https://pypi.org/project/neatrl/)</content>
<parameter name="filePath">/Users/yuvrajsingh9886/Desktop/NeatRL/NeatRL/neatrl/docs/README.md
