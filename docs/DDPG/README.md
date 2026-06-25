# DDPG (Deep Deterministic Policy Gradient)

DDPG is an actor-critic algorithm for continuous action spaces that uses deep neural networks. This implementation includes both standard DDPG for vector observations and CNN-based DDPG for image observations.

Note: DDPG is designed for continuous action spaces. For discrete action environments, consider using DQN instead.

## Features

- **Standard DDPG**: For environments with vector observations (e.g., HalfCheetah, BipedalWalker)
- **CNN DDPG**: For environments with image observations (e.g., Atari games)
- **Experience Replay**: Stabilizes training with off-policy learning
- **Target Networks**: Prevents moving target problem with soft updates
- **Exploration**: Ornstein-Uhlenbeck noise for action exploration
- **W&B Integration**: Built-in logging and video recording

## Usage

### Standard DDPG (Vector Observations)

```python
from neatrl.ddpg_mlp import train_ddpg

train_ddpg(
    env_id="HalfCheetah-v5",
    total_timesteps=1000000,
    use_wandb=True
)
```

### CNN DDPG (Image Observations)

```python
from neatrl.ddpg_cnn import train_ddpg_cnn

train_ddpg_cnn(
    env_id="PongNoFrameskip-v4",
    total_timesteps=100000,
    use_wandb=True
)
```

## Network Architectures

### Actor Network (Standard)
- Input: State vector
- Hidden: 256 -> 256 units with Mish activation
- Output: Continuous actions (tanh activation)

### Q-Network (Standard)
- State path: 256 -> 256 units
- Action path: Linear projection
- Combined: 512 -> 512 -> 256 -> 1 units

### Actor Network (CNN)
- Conv layers: 32@8x8 -> 64@4x4 -> 64@3x3
- FC layers: 3136 -> 512 -> action_dim
- Output: Continuous actions (tanh activation)

### Q-Network (CNN)
- State conv: Same as Actor CNN
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
| `train_frequency` | int | `2` | How often to perform gradient updates |
| `target_network_frequency` | int | `50` | How often to update target networks |
| `exploration_fraction` | float | `0.1` | Fraction of timesteps for exploration noise decay |
| `low` | float | `-1.0` | Lower bound for action space |
| `high` | float | `1.0` | Upper bound for action space |
| `noise_clip` | float | `0.5` | Maximum absolute value for exploration noise |
| `normalize_obs` | bool | `False` | Whether to normalize observations |
| `normalize_reward` | bool | `False` | Whether to normalize rewards |
| `seed` | int | `42` | Random seed |
| `exp_name` | str | `"DDPG-Experiment"` | Experiment name |
| `use_wandb` | bool | `True` | Whether to log to Weights & Biases |
| `wandb_project` | str | `"cleanRL"` | W&B project name |
| `wandb_entity` | str | `""` | W&B username/team |
| `capture_video` | bool | `True` | Whether to record evaluation videos |
| `eval_every` | int | `500` | Evaluation frequency (steps) |
| `save_every` | int | `10000` | Model save frequency (steps) |
| `num_eval_episodes` | int | `10` | Number of evaluation episodes |
| `log_gradients` | bool | `True` | Whether to log gradient norms |
| `actor_class` | nn.Module | `ActorNet` | Custom actor network class or instance |
| `critic_class` | nn.Module | `QNet` | Custom Q-network class or instance |
| `max_grad_norm` | float | `0.0` | Maximum gradient norm for clipping (0.0 to disable) |
| `device` | str | `"cpu"` | Training device ("cpu", "cuda", "mps", "auto") |

### Example: Custom Configuration

```python
from neatrl.ddpg_mlp import train_ddpg

model = train_ddpg(
    env_id="BipedalWalker-v3",
    total_timesteps=500000,
    learning_rate=1e-3,
    buffer_size=200000,
    gamma=0.99,
    tau=0.005,
    batch_size=128,
    learning_starts=10000,
    exploration_fraction=0.2,
    use_wandb=True,
    wandb_project="my-ddpg-experiments",
    exp_name="bipedal-walker-custom",
    capture_video=True,
    eval_every=1000,
    num_eval_episodes=5
)
```

## Atari Environments

For Atari games, the implementation automatically applies:
- Grayscale conversion
- Frame scaling to 84x84
- Frame stacking (4 frames)
- Episode statistics recording

## Example Scripts

- `run_ddpg_pendulum.py` - DDPG training on Pendulum
- `run_ddpg_half_cheetah.py` - DDPG training on HalfCheetah
- `run_ddpg_bipedal_walker.py` - DDPG training on BipedalWalker
- `run_ddpg_cnn_car_racing.py` - DDPG CNN training on CarRacing

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
