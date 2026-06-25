# NeatRL Documentation - REINFORCE

This guide shows you how to use NeatRL's REINFORCE algorithm implementation.

## Quick Start with REINFORCE

### Basic Training

```python
from neatrl import train_reinforce

model = train_reinforce(
    env_id="CartPole-v1",
    total_steps=2000,
    seed=42
)
```

### Training with Experiment Tracking

```python
from neatrl import train_reinforce

model = train_reinforce(
    env_id="CartPole-v1",
    total_steps=2000,
    seed=42,
    capture_video=True,
    use_wandb=True,
    wandb_project="my-rl-experiments",
    exp_name="reinforce-cartpole-v1"
)
```

### Parallel Training

```python
from neatrl import train_reinforce

model = train_reinforce(
    env_id="CartPole-v1",
    episodes=2000,
    n_envs=4,
    seed=42,
    use_wandb=True,
    wandb_project="parallel-experiments",
    exp_name="reinforce-cartpole-parallel"
)
```

### Atari Game Training

```python
from neatrl import train_reinforce

model = train_reinforce(
    env_id="BreakoutNoFrameskip-v4",
    total_steps=2000,
    seed=42,
    atari_wrapper=True,
    n_envs=4,
    use_wandb=True,
    wandb_project="atari-experiments",
    exp_name="reinforce-breakout"
)
```

### Custom Policy Network

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

model = train_reinforce(
    env_id="CartPole-v1",
    total_steps=2000,
    seed=42,
    custom_agent=CustomPolicyNet(4, 2),
    use_wandb=True,
    exp_name="reinforce-cartpole-custom"
)
```

## Function Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `env_id` | str | `"CartPole-v1"` | Gymnasium environment ID |
| `total_steps` | int | `2000` | Number of episodes to train for |
| `seed` | int | `42` | Random seed |
| `learning_rate` | float | `2.5e-4` | Learning rate for the Adam optimizer |
| `gamma` | float | `0.99` | Discount factor |
| `max_grad_norm` | float | `1.0` | Maximum gradient norm for clipping |
| `capture_video` | bool | `False` | Whether to record training videos |
| `use_wandb` | bool | `False` | Whether to log to Weights & Biases |
| `wandb_project` | str | `"cleanRL"` | W&B project name |
| `wandb_entity` | str | `""` | W&B username/entity |
| `exp_name` | str | `"REINFORCE"` | Experiment name |
| `eval_every` | int | `100` | Evaluation frequency (episodes) |
| `save_every` | int | `1000` | Model save frequency (episodes) |
| `atari_wrapper` | bool | `False` | Whether to apply Atari preprocessing |
| `n_envs` | int | `4` | Number of parallel environments |
| `custom_agent` | nn.Module | `None` | Custom policy network class/instance |
| `num_eval_eps` | int | `10` | Number of evaluation episodes |
| `device` | str | `"auto"` | Training device ("auto", "cpu", "cuda", "mps") |
| `grid_env` | bool | `False` | Whether the environment uses discrete grid observations |
| `use_entropy` | bool | `False` | Whether to use entropy regularization |
| `entropy_coeff` | float | `0.01` | Entropy regularization coefficient |
| `normalize_obs` | bool | `False` | Whether to normalize observations |
| `normalize_reward` | bool | `False` | Whether to normalize rewards |
| `log_gradients` | bool | `False` | Whether to log gradient norms |
| `anneal_lr` | bool | `True` | Whether to anneal learning rate |
| `env_wrapper` | callable | `None` | Custom environment wrapper function |

## Supported Environments

### Classic Control
- `CartPole-v1`
- `MountainCar-v0`
- `Acrobot-v1`
- `Pendulum-v1`

### Atari Games

```python
from neatrl import train_reinforce

model = train_reinforce(
    env_id="BreakoutNoFrameskip-v4",
    total_steps=2000,
    atari_wrapper=True,
    n_envs=4,
    seed=42
)
```

Supported Atari games include:
- `BreakoutNoFrameskip-v4`
- `PongNoFrameskip-v4`
- `SpaceInvadersNoFrameskip-v4`

### Box2D
- `LunarLander-v3`
- `CarRacing-v2`

### Grid Environments

```python
from neatrl import train_reinforce

model = train_reinforce(
    env_id="FrozenLake-v1",
    total_steps=2000,
    seed=42,
    grid_env=True,
    use_wandb=True,
    exp_name="reinforce-frozenlake"
)
```

## Experiment Tracking with Weights & Biases

```python
model = train_reinforce(
    env_id="CartPole-v1",
    total_steps=2000,
    use_wandb=True,
    wandb_project="my-rl-project",
    wandb_entity="your-username",
    exp_name="cartpole-experiment"
)
```

What gets logged:
- Episode returns, episode lengths, policy loss
- Training progress videos (if `capture_video=True`)
- All training configuration

## Example Scripts

- `run_reinforce_cartpole.py` - Basic REINFORCE training on CartPole with parallel environments
- `run_reinforce_frozenlake.py` - REINFORCE training on FrozenLake (with one-hot encoding)
- `run_reinforce_pendulum.py` - REINFORCE training on Pendulum (continuous actions)
- `run_reinforce_car_racing.py` - REINFORCE training on CarRacing (CNN architecture)

## Installation

```bash
pip install neatrl

pip install neatrl[atari]      # For Atari games
pip install neatrl[classic]    # For CartPole, FrozenLake, Pendulum
```

## PyPI

For installation and more information, visit [NeatRL on PyPI](https://pypi.org/project/neatrl/)
