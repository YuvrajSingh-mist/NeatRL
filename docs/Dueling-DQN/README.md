# NeatRL Documentation - Dueling DQN

This guide shows you how to use NeatRL's Dueling DQN implementation, which separates value and advantage streams for better performance on certain environments.

## Quick Start with Dueling DQN

### Basic Training

```python
from neatrl.dueling_dqn_mlp import train_dueling_dqn

model = train_dueling_dqn(
    env_id="CliffWalking-v0",
    total_timesteps=10000,
    seed=42
)
```

### Training with Experiment Tracking

```python
from neatrl.dueling_dqn_mlp import train_dueling_dqn

model = train_dueling_dqn(
    env_id="CliffWalking-v0",
    total_timesteps=50000,
    seed=42,
    capture_video=True,
    use_wandb=True,
    wandb_project="my-rl-experiments",
    exp_name="dueling-dqn-cliffwalking"
)
```

### Custom Dueling Architecture

```python
import torch.nn as nn
from neatrl.dueling_dqn_mlp import train_dueling_dqn

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
        q_values = values + adv - adv.mean(dim=1, keepdim=True)
        return q_values, values, adv, feat

model = train_dueling_dqn(
    env_id="LunarLander-v2",
    total_timesteps=100000,
    seed=42,
    custom_agent=CustomDuelingQNet(8, 4),
    use_wandb=True,
    exp_name="custom-dueling-lunar"
)
```

## Function Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `env_id` | str | `"CliffWalking-v0"` | Gymnasium environment ID |
| `total_timesteps` | int | `300000` | Total number of environment steps |
| `seed` | int | `42` | Random seed |
| `learning_rate` | float | `2e-4` | Learning rate for the Adam optimizer |
| `buffer_size` | int | `30000` | Size of the replay buffer |
| `gamma` | float | `0.99` | Discount factor |
| `tau` | float | `1.0` | Target network update rate (1.0 = hard update) |
| `target_network_frequency` | int | `50` | How often to update the target network |
| `batch_size` | int | `128` | Batch size for training |
| `start_e` | float | `1.0` | Initial epsilon for exploration |
| `end_e` | float | `0.05` | Final epsilon for exploration |
| `exploration_fraction` | float | `0.4` | Fraction of training steps for epsilon decay |
| `learning_starts` | int | `1000` | Steps before training begins |
| `train_frequency` | int | `4` | How often to train the network |
| `max_grad_norm` | float | `4.0` | Maximum gradient norm for clipping |
| `capture_video` | bool | `False` | Whether to record training videos |
| `use_wandb` | bool | `False` | Whether to log to Weights & Biases |
| `wandb_project` | str | `"cleanRL"` | W&B project name |
| `wandb_entity` | str | `""` | W&B username/entity |
| `exp_name` | str | `"Dueling-DQN"` | Experiment name |
| `eval_every` | int | `10000` | Evaluation frequency (steps) |
| `save_every` | int | `100000` | Model save frequency (steps) |
| `atari_wrapper` | bool | `False` | Whether to apply Atari preprocessing |
| `custom_agent` | nn.Module | `None` | Custom dueling Q-network class/instance |
| `num_eval_eps` | int | `10` | Number of evaluation episodes |
| `n_envs` | int | `1` | Number of parallel environments |
| `device` | str | `"cpu"` | Training device ("cpu", "cuda", "mps") |
| `grid_env` | bool | `True` | Whether the environment uses discrete grid observations |

## Supported Environments

### Classic Control
- `CartPole-v1`
- `MountainCar-v0`
- `Acrobot-v1`

### Grid Environments

```python
from neatrl.dueling_dqn_mlp import train_dueling_dqn

model = train_dueling_dqn(
    env_id="CliffWalking-v0",
    total_timesteps=50000,
    seed=42,
    grid_env=True,
    use_wandb=True,
    exp_name="dueling-dqn-cliffwalking"
)
```

### Box2D
- `LunarLander-v2`
- `LunarLander-v3`

### Toy Text
- `FrozenLake-v1`
- `Taxi-v3`
- `CliffWalking-v0`

## Dueling Architecture

Dueling DQN separates Q-value estimation into two streams:

1. **Value Stream**: Estimates the value of being in a state V(s)
2. **Advantage Stream**: Estimates the advantage of each action A(s,a)

Final Q-values:
```
Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
```

This helps the agent distinguish between states that are generally good or bad (value function) and which actions are better in those states (advantage function).

## Experiment Tracking with Weights & Biases

```python
model = train_dueling_dqn(
    env_id="CliffWalking-v0",
    total_timesteps=50000,
    use_wandb=True,
    wandb_project="my-rl-project",
    wandb_entity="your-username",
    exp_name="cliffwalking-experiment"
)
```

What gets logged:
- Episode returns, epsilon values, training loss
- Value/advantage stream outputs, gradient norms
- Training progress videos (if `capture_video=True`)

## Example Scripts

- `run_duel_dqn_cliff.py` - Dueling DQN training on CliffWalking
- `run_duel_dqn_frozenlake.py` - Dueling DQN training on FrozenLake
- `run_duel_dqn_lunar.py` - Dueling DQN training on LunarLander

## Installation

```bash
pip install neatrl

pip install neatrl[box2d]      # For LunarLander
pip install neatrl[classic]    # For CliffWalking, FrozenLake
```

## PyPI

For installation and more information, visit [NeatRL on PyPI](https://pypi.org/project/neatrl/)
