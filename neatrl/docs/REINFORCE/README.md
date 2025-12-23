# 🎯 NeatRL Documentation - REINFORCE

Welcome to the NeatRL REINFORCE documentation! This guide shows you how to use NeatRL's REINFORCE algorithm implementation, with a focus on practical examples and best practices.

## 🚀 Quick Start with REINFORCE

### Basic Training

Train a REINFORCE agent on CartPole in just a few lines:

```python
from neatrl import train_reinforce

# Train REINFORCE on CartPole
model = train_reinforce(
    env_id="CartPole-v1",
    episodes=2000,
    seed=42
)

print("Training completed! 🎉")
```

### Training with Experiment Tracking

Enable Weights & Biases for experiment tracking and video recording:

```python
from neatrl import train_reinforce

model = train_reinforce(
    env_id="CartPole-v1",
    episodes=2000,
    seed=42,
    capture_video=True,        # Record training videos
    use_wandb=True,           # Enable W&B logging
    wandb_project="my-rl-experiments",
    exp_name="reinforce-cartpole-v1"
)
```

### Parallel Training for Speed

Train faster with multiple parallel environments:

```python
from neatrl import train_reinforce

model = train_reinforce(
    env_id="CartPole-v1",
    episodes=2000,
    n_envs=4,                # Use 4 parallel environments
    seed=42,
    use_wandb=True,
    wandb_project="parallel-experiments",
    exp_name="reinforce-cartpole-parallel"
)
```

### Atari Game Training

REINFORCE now supports Atari games with automatic CNN architecture:

```python
from neatrl import train_reinforce

# Train REINFORCE on Atari Breakout
model = train_reinforce(
    env_id="BreakoutNoFrameskip-v4",
    episodes=2000,
    seed=42,
    atari_wrapper=True,       # Enable Atari preprocessing
    n_envs=4,                 # Use 4 parallel environments
    use_wandb=True,
    wandb_project="atari-experiments",
    exp_name="reinforce-breakout"
)
```

### Custom Policy Network

You can also provide a custom policy network:

```python
import torch.nn as nn
from neatrl import train_reinforce

class CustomPolicyNet(nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.fc1 = nn.Linear(state_space, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, action_space)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.nn.functional.softmax(self.out(x), dim=-1)

    def get_action(self, x):
        action_probs = self.forward(x)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action, dist.log_prob(action)

# Train with custom policy
model = train_reinforce(
    env_id="CartPole-v1",
    episodes=2000,
    seed=42,
    custom_agent=CustomPolicyNet(4, 2),  # 4 state dims, 2 actions
    use_wandb=True,
    wandb_project="reinforce-experiments",
    exp_name="reinforce-cartpole-custom"
)
```

## 🔧 Function Arguments

The `train_reinforce` function accepts the following arguments for customizing your REINFORCE training:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `env_id` | str | `"CartPole-v1"` | Gymnasium environment ID to train on |
| `episodes` | int | `2000` | Number of episodes to train for |
| `seed` | int | `42` | Random seed for reproducibility |
| `learning_rate` | float | `2e-3` | Learning rate for the Adam optimizer |
| `gamma` | float | `0.99` | Discount factor for future rewards |
| `max_grad_norm` | float | `1.0` | Maximum gradient norm for clipping (0.0 to disable) |
| `capture_video` | bool | `False` | Whether to record training videos |
| `use_wandb` | bool | `False` | Whether to log to Weights & Biases |
| `wandb_project` | str | `"cleanRL"` | W&B project name |
| `wandb_entity` | str | `""` | W&B username/entity |
| `exp_name` | str | `"REINFORCE"` | Experiment name for logging |
| `eval_every` | int | `100` | Frequency of evaluation during training (in episodes) |
| `save_every` | int | `1000` | Frequency of saving model checkpoints (in episodes) |
| `atari_wrapper` | bool | `False` | Whether to apply Atari preprocessing wrappers |
| `n_envs` | int | `4` | Number of parallel environments for training |
| `custom_agent` | nn.Module | `None` | Custom policy network class/instance (overrides default) |
| `num_eval_eps` | int | `10` | Number of episodes for evaluation |
| `device` | str | `"cpu"` | Device for training ("cpu", "cuda", "mps") |
| `grid_env` | bool | `False` | Whether the environment uses discrete grid observations |

## 🎮 Supported Environments

REINFORCE works with any Gymnasium environment. Here are some popular choices:

### Classic Control
- `CartPole-v1` - Balance a pole on a cart
- `MountainCar-v0` - Drive up a hill
- `Acrobot-v1` - Swing up a two-link robot
- `Pendulum-v1` - Swing up a pendulum

### Atari Games
REINFORCE now supports Atari games with automatic CNN architecture and preprocessing!

**Example: Breakout**

```python
from neatrl import train_reinforce

model = train_reinforce(
    env_id="BreakoutNoFrameskip-v4",
    episodes=2000,
    atari_wrapper=True,  # Enables Atari preprocessing
    n_envs=4,            # Parallel environments
    seed=42
)
```

**Supported Atari Games:**
- `BreakoutNoFrameskip-v4`
- `PongNoFrameskip-v4`
- `SpaceInvadersNoFrameskip-v4`
- And many more!

### Box2D
- `LunarLander-v3` - Land a spacecraft safely
- `CarRacing-v2` - Race a car around a track (if discrete action space is chosen)

### Grid Environments (Toy Text)
NeatRL now supports grid-based environments with automatic one-hot encoding for discrete states!

**Example: FrozenLake**

```python
from neatrl import train_reinforce

# Train REINFORCE on FrozenLake with automatic one-hot encoding
model = train_reinforce(
    env_id="FrozenLake-v1",
    episodes=2000,
    seed=42,
    grid_env=True,  # Enable one-hot encoding for discrete states
    use_wandb=True,
    wandb_project="grid-experiments",
    exp_name="reinforce-frozenlake"
)
```

**Example: Taxi**

```python
from neatrl import train_reinforce

# Train REINFORCE on Taxi with automatic one-hot encoding
model = train_reinforce(
    env_id="Taxi-v3",
    episodes=2000,
    seed=42,
    grid_env=True,  # Enable one-hot encoding for discrete states
    use_wandb=True,
    wandb_project="grid-experiments",
    exp_name="reinforce-taxi"
)
```

The `grid_env=True` parameter automatically applies one-hot encoding to discrete state observations, making them suitable for neural network input.

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
model = train_reinforce(
    env_id="CartPole-v1",
    episodes=2000,
    use_wandb=True,
    wandb_project="my-rl-project",
    wandb_entity="your-username",  # Optional: your W&B username
    exp_name="cartpole-experiment"
)
```

### What gets logged:

- **Metrics**: Episode returns, episode lengths, policy loss
- **Videos**: Training progress videos (recorded every 100 episodes by default)
- **Hyperparameters**: All training configuration
- **System info**: Hardware usage, training time

## 🎥 Video Recording

NeatRL can automatically record and upload training videos:

```python
model = train_reinforce(
    env_id="CartPole-v1",
    episodes=2000,
    capture_video=True,     # Enable video recording
    use_wandb=True,         # Upload to W&B
)
```

Videos are:
- Recorded during evaluation episodes
- Uploaded directly to W&B (no local storage)
- Automatically cleaned up after upload

## 📚 Examples

Check out these example scripts:

- [`run_reinforce_cartpole.py`](./run_reinforce_cartpole.py) - Basic REINFORCE training on CartPole with parallel environments
- [`run_reinforce_frozenlake.py`](./run_reinforce_frozenlake.py) - REINFORCE training on FrozenLake (with one-hot encoding)
- [`run_reinforce_pendulum.py`](./run_reinforce_pendulum.py) - REINFORCE training on Pendulum (continuous actions - experimental)
- [`run_reinforce_car_racing.py`](./run_reinforce_car_racing.py) - REINFORCE training on CarRacing (CNN architecture)

---

Happy training! 🚀</content>