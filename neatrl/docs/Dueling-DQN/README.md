# 🎯 NeatRL Documentation - Dueling DQN

Welcome to the NeatRL documentation for Dueling DQN! This guide shows you how to use NeatRL's Dueling DQN implementation, which separates value and advantage streams for better performance on certain environments.

## 🚀 Quick Start with Dueling DQN

### Basic Training

Train a Dueling DQN agent on CliffWalking in just a few lines:

```python
from neatrl import train_dueling_dqn

# Train Dueling DQN on CliffWalking
model = train_dueling_dqn(
    env_id="CliffWalking-v0",
    total_timesteps=10000,
    seed=42
)

print("Training completed! 🎉")
```

### Training with Experiment Tracking

Enable Weights & Biases for experiment tracking and video recording:

```python
from neatrl import train_dueling_dqn

model = train_dueling_dqn(
    env_id="CliffWalking-v0",
    total_timesteps=50000,
    seed=42,
    capture_video=True,        # Record training videos
    use_wandb=True,           # Enable W&B logging
    wandb_project="my-rl-experiments",
    exp_name="dueling-dqn-cliffwalking"
)
```

### Custom Dueling Architecture

You can also define custom dueling architectures:

```python
import torch.nn as nn
from neatrl import train_dueling_dqn

class CustomDuelingQNet(nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(state_space, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.values = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.adv = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_space)
        )

    def forward(self, x):
        feat = self.features(x)
        values = self.values(feat)
        adv = self.adv(feat)
        # Dueling architecture: Q = V + A - mean(A)
        q_values = values + adv - adv.mean(dim=1, keepdim=True)
        return q_values, values, adv, feat

# Train with custom architecture
model = train_dueling_dqn(
    env_id="LunarLander-v2",
    total_timesteps=100000,
    seed=42,
    custom_agent=CustomDuelingQNet(8, 4),  # 8 state dims, 4 actions
    use_wandb=True,
    wandb_project="dueling-experiments",
    exp_name="custom-dueling-lunar"
)
```

## 🔧 Function Arguments

The `train_dueling_dqn` function accepts the following arguments for customizing your Dueling DQN training:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `env_id` | str | `"CliffWalking-v0"` | Gymnasium environment ID to train on |
| `total_timesteps` | int | `300000` | Total number of environment steps to train for |
| `seed` | int | `42` | Random seed for reproducibility |
| `learning_rate` | float | `2e-4` | Learning rate for the Adam optimizer |
| `buffer_size` | int | `30000` | Size of the replay buffer |
| `gamma` | float | `0.99` | Discount factor for future rewards |
| `tau` | float | `1.0` | Target network update rate (1.0 = hard update, <1.0 = soft update) |
| `target_network_frequency` | int | `50` | How often to update the target network (in steps) |
| `batch_size` | int | `128` | Batch size for training |
| `start_e` | float | `1.0` | Initial epsilon for ε-greedy exploration |
| `end_e` | float | `0.05` | Final epsilon for ε-greedy exploration |
| `exploration_fraction` | float | `0.4` | Fraction of training steps for epsilon decay |
| `learning_starts` | int | `1000` | Number of steps to collect before starting training |
| `train_frequency` | int | `4` | How often to train the network (in steps) |
| `max_grad_norm` | float | `4.0` | Maximum gradient norm for clipping (0.0 to disable) |
| `capture_video` | bool | `False` | Whether to record training videos |
| `use_wandb` | bool | `False` | Whether to log to Weights & Biases |
| `wandb_project` | str | `"cleanRL"` | W&B project name |
| `wandb_entity` | str | `""` | W&B username/entity |
| `exp_name` | str | `"Dueling-DQN"` | Experiment name for logging |
| `eval_every` | int | `10000` | Frequency of evaluation during training (in steps) |
| `save_every` | int | `100000` | Frequency of saving model checkpoints (in steps) |
| `atari_wrapper` | bool | `False` | Whether to apply Atari preprocessing wrappers |
| `custom_agent` | nn.Module | `None` | Custom dueling Q-network class/instance (overrides default) |
| `num_eval_eps` | int | `10` | Number of episodes for evaluation |
| `n_envs` | int | `1` | Number of parallel environments for vectorized training |
| `device` | str | `"cpu"` | Device for training ("cpu", "cuda", "mps") |
| `grid_env` | bool | `True` | Whether the environment uses discrete grid observations |

## 🎮 Supported Environments

Dueling DQN works with any Gymnasium environment. Here are some popular choices:

### Classic Control
- `CartPole-v1` - Balance a pole on a cart
- `MountainCar-v0` - Drive up a hill
- `Acrobot-v1` - Swing up a two-link robot

### Grid Environments
Dueling DQN excels on grid-based environments with automatic one-hot encoding!

**Example: CliffWalking**

```python
from neatrl import train_dueling_dqn

# Train Dueling DQN on CliffWalking with automatic one-hot encoding
model = train_dueling_dqn(
    env_id="CliffWalking-v0",
    total_timesteps=50000,
    seed=42,
    grid_env=False,  # Enable one-hot encoding for discrete states
    use_wandb=True,
    wandb_project="grid-experiments",
    exp_name="dueling-dqn-cliffwalking"
)
```

**Example: FrozenLake**

```python
from neatrl import train_dueling_dqn

# Train Dueling DQN on FrozenLake with automatic one-hot encoding
model = train_dueling_dqn(
    env_id="FrozenLake-v1",
    total_timesteps=50000,
    seed=42,
    grid_env=False,  # Enable one-hot encoding for discrete states
    use_wandb=True,
    wandb_project="grid-experiments",
    exp_name="dueling-dqn-frozenlake"
)
```

The `grid_env=False` parameter automatically applies one-hot encoding to discrete state observations, making them suitable for neural network input.

### Box2D
- `LunarLander-v2` - Land a spacecraft safely
- `LunarLander-v3` - Land a spacecraft safely (updated version)

### Toy Text
- `FrozenLake-v1` - Navigate a slippery frozen lake
- `Taxi-v3` - Pick up and drop off passengers
- `CliffWalking-v0` - Avoid cliffs while walking

## 🏗️ Dueling Architecture

Dueling DQN separates the Q-value estimation into two streams:

1. **Value Stream**: Estimates the value of being in a state V(s)
2. **Advantage Stream**: Estimates the advantage of each action A(s,a)

The final Q-values are computed as: **Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))**

This architecture helps the agent better distinguish between:
- States that are generally good/bad (value function)
- Which actions are better/worse in those states (advantage function)

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
model = train_dueling_dqn(
    env_id="CliffWalking-v0",
    total_timesteps=50000,
    use_wandb=True,
    wandb_project="my-rl-project",
    wandb_entity="your-username",  # Optional: your W&B username
    exp_name="cliffwalking-experiment"
)
```

### What gets logged:

- **Metrics**: Episode returns, episode lengths, epsilon values, training loss
- **Videos**: Training progress videos (recorded every 100 steps by default)
- **Hyperparameters**: All training configuration
- **System info**: Hardware usage, training time
- **Network stats**: Value/advantage stream outputs, gradient norms

## 🎥 Video Recording

NeatRL can automatically record and upload training videos:

```python
model = train_dueling_dqn(
    env_id="CliffWalking-v0",
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

- [`run_duel_dqn_cliff.py`](./run_duel_dqn_cliff.py) - Dueling DQN training on CliffWalking
- [`run_duel_dqn_frozenlake.py`](./run_duel_dqn_frozenlake.py) - Dueling DQN training on FrozenLake
- [`run_duel_dqn_lunar.py`](./run_duel_dqn_lunar.py) - Dueling DQN training on LunarLander

---

Happy training with Dueling DQN! 🚀