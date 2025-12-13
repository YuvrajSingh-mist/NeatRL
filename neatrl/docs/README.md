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

### Advanced Configuration

Fine-tune your DQN training with custom hyperparameters:

```python
from neatrl import train_dqn

model = train_dqn(
    env_id="LunarLander-v2",
    total_timesteps=100000,
    seed=42,

    # Network hyperparameters
    learning_rate=2.5e-4,
    gamma=0.99,
    tau=1.0,

    # Training hyperparameters
    buffer_size=100000,
    batch_size=128,
    target_network_frequency=500,

    # Exploration
    start_e=1.0,
    end_e=0.05,
    exploration_fraction=0.5,

    # Training schedule
    learning_starts=1000,
    train_frequency=4,

    # Monitoring & logging
    capture_video=True,
    use_wandb=True,
    wandb_project="lunar-lander-experiments",
    exp_name="dqn-lunar-lander",

    # Evaluation & saving
    eval_every=1000,
    save_every=5000,
    upload_every=100,  # Upload videos every 100 steps
)
```

## ⚙️ Complete Parameter Reference

The `train_dqn()` function accepts the following parameters:

### Environment & Experiment Settings
- **`env_id`** (str): Gymnasium environment ID (default: "CartPole-v1")
- **`exp_name`** (str): Experiment name for logging (default: "DQN")
- **`seed`** (int): Random seed for reproducibility (default: 42)

### Core Training Parameters
- **`total_timesteps`** (int): Total number of training timesteps (default: 20000)
- **`learning_rate`** (float): Learning rate for optimizer (default: 2.5e-4)
- **`buffer_size`** (int): Size of the replay buffer (default: 10000)
- **`gamma`** (float): Discount factor for future rewards (default: 0.99)
- **`tau`** (float): Soft update parameter for target network (default: 1.0)
- **`target_network_frequency`** (int): How often to update target network (default: 50)
- **`batch_size`** (int): Batch size for training updates (default: 128)

### Exploration Parameters
- **`start_e`** (float): Initial epsilon for exploration (default: 1.0)
- **`end_e`** (float): Final epsilon for exploration (default: 0.05)
- **`exploration_fraction`** (float): Fraction of timesteps for epsilon decay (default: 0.5)

### Training Schedule
- **`learning_starts`** (int): Timesteps before starting to learn (default: 1000)
- **`train_frequency`** (int): How often to perform training updates (default: 10)

### Logging & Monitoring
- **`capture_video`** (bool): Whether to record training videos (default: False)
- **`use_wandb`** (bool): Whether to use Weights & Biases logging (default: False)
- **`wandb_project`** (str): W&B project name (default: "cleanRL")
- **`wandb_entity`** (str): W&B username/team (default: "")

### Evaluation & Saving
- **`eval_every`** (int): Frequency of evaluation during training (default: 1000)
- **`save_every`** (int): Frequency of saving model checkpoints (default: 1000)
- **`upload_every`** (int): Frequency of uploading videos to W&B (default: 100)

### Custom Agent
- **`agent`** (nn.Module): Custom neural network instance (default: None, uses QNet)

## 🎮 Supported Environments

DQN works with any Gymnasium environment. Here are some popular choices:

### Classic Control
- `CartPole-v1` - Balance a pole on a cart
- `MountainCar-v0` - Drive up a hill
- `Acrobot-v1` - Swing up a two-link robot
- `Pendulum-v1` - Swing up a pendulum

### Box2D
- `LunarLander-v2` - Land a spacecraft safely
- `BipedalWalker-v3` - Make a robot walk
- `CarRacing-v2` - Race a car around a track

### Atari Games
- `ALE/Pong-v5` - Classic Pong game
- `ALE/Breakout-v5` - Breakout game
- `ALE/SpaceInvaders-v5` - Space Invaders

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

## 🧠 Custom Neural Network Agents

Use your own neural network architecture instead of the default QNet:

```python
import torch.nn as nn
from neatrl import train_dqn

class CustomQNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)

# Train with custom agent
model = train_dqn(
    env_id="CartPole-v1",
    total_timesteps=50000,
    agent=CustomQNet,  # Pass instantiated nn.Module
    use_wandb=True,
    exp_name="custom-agent-dqn"
)
```

You can pass any `nn.Module` instance that takes state observations as input and outputs Q-values for each action.

## 📈 Evaluating Trained Models

After training, evaluate your model's performance:

```python
from neatrl import evaluate
import torch

# Load or use trained model
# model = train_dqn(...)  # Your trained model

# Evaluate on 10 episodes
returns, lengths = evaluate(
    model=model,
    device="cpu",  # or "cuda" if using GPU
    run_name="evaluation-test",
    num_eval_eps=10,
    record=False  # Set to True to save evaluation videos
)

print(f"Average return: {sum(returns)/len(returns):.2f}")
print(f"Average episode length: {sum(lengths)/len(lengths):.2f}")
print(f"Best episode: {max(returns):.2f}")
```

### Getting Help

- Check the [GitHub Issues](https://github.com/YuvrajSingh-mist/NeatRL/issues) for common problems
- Review the [examples](./) in this directory
- Join the discussion in [GitHub Discussions](https://github.com/YuvrajSingh-mist/NeatRL/discussions)

## 📚 Examples

Check out these example scripts:

- [`run_dqn_cartpole.py`](./run_dqn_cartpole.py) - Basic DQN training on CartPole
- *More examples coming soon...*

---

Happy training! 🚀</content>
<parameter name="filePath">/Users/yuvrajsingh9886/Desktop/NeatRL/NeatRL/neatrl/docs/README.md
