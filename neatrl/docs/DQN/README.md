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

## 🎮 Supported Environments

DQN works with any Gymnasium environment. Here are some popular choices:

### Classic Control
- `CartPole-v1` - Balance a pole on a cart
<!-- - `MountainCar-v0` - Drive up a hill -->
- `Acrobot-v1` - Swing up a two-link robot
<!-- - `Pendulum-v1` - Swing up a pendulum -->

### Box2D
- `LunarLander-v2` - Land a spacecraft safely
- `BipedalWalker-v3` - Make a robot walk
- `CarRacing-v2` - Race a car around a track


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
- ['run_dqn_atari.py`](./run_dqn_atari.py) - DQN training on Atari-Breakout
- *More examples coming soon...*

---

Happy training! 🚀</content>
<parameter name="filePath">/Users/yuvrajsingh9886/Desktop/NeatRL/NeatRL/neatrl/docs/README.md
