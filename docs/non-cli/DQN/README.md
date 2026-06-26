# NeatRL Documentation - DQN

This guide shows you how to use NeatRL's DQN implementation.

## Quick Start with DQN

### CLI

```bash
neatrl train dqn cartpole
neatrl train dqn lunar
neatrl train dqn mountaincar
neatrl train dqn frozenlake
neatrl train dqn breakout
```

### Python API

```python
from neatrl.dqn_mlp import train_dqn

model = train_dqn(
    env_id="CartPole-v1",
    total_timesteps=100000,
    seed=42,
    use_wandb=False,
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


## Installation

```bash
pip install neatrl

pip install neatrl[atari]      # For Atari games
pip install neatrl[box2d]      # For LunarLander
pip install neatrl[classic]    # For CartPole, FrozenLake, MountainCar, Acrobot
```

## PyPI

For installation and more information, visit [NeatRL on PyPI](https://pypi.org/project/neatrl/)
