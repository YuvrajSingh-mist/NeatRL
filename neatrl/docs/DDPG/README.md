# DDPG (Deep Deterministic Policy Gradient)

DDPG is an actor-critic algorithm for continuous action spaces that uses deep neural networks. This implementation includes both standard DDPG for vector observations and CNN-based DDPG for image observations (like Atari games).

**Note**: DDPG is designed for continuous action spaces. For discrete action environments, consider using DQN instead.

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
from neatrl.ddpg import train_ddpg

train_ddpg(
    env_id="HalfCheetah-v5",
    total_timesteps=1000000,
    use_wandb=True
)
```

### CNN DDPG (Image Observations)

```python
from neatrl.ddpg import train_ddpg_cnn

train_ddpg_cnn(
    env_id="PongNoFrameskip-v4",
    total_timesteps=100000,
    use_wandb=True
)
```

## Network Architectures

### Actor Network (Standard)
- Input: State vector
- Hidden: 256 → 256 units with Mish activation
- Output: Continuous actions (tanh activation)

### Q-Network (Standard)
- State path: 256 → 256 units
- Action path: Linear projection
- Combined: 512 → 512 → 256 → 1 units

### Actor Network (CNN)
- Conv layers: 32@8x8 → 64@4x4 → 64@3x3
- FC layers: 3136 → 512 → action_dim
- Output: Continuous actions (tanh activation)

### Q-Network (CNN)
- State conv: Same as Actor CNN
- State FC: 3136 → 512
- Action FC: action_dim → 512
- Combined: 1024 → 512 → 1

## 🔧 Configuration Arguments

The `train_ddpg` and `train_ddpg_cnn` functions accept the following arguments for customizing your DDPG training:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| **Environment Settings** | | | |
| `env_id` | str | `"HalfCheetah-v5"` | Gymnasium environment ID to train on |
| `env` | gym.Env | `None` | Optional pre-created environment instance |
| `grid_env` | bool | `False` | Whether to apply one-hot encoding for discrete state spaces |
| `atari_wrapper` | bool | `False` | Whether to apply Atari preprocessing (grayscale, frame stack, etc.) |
| `env_wrapper` | Callable | `None` | Optional custom environment wrapper function |
| `n_envs` | int | `1` | Number of parallel environments for vectorized training |
| **Training Parameters** | | | |
| `total_timesteps` | int | `1000000` | Total number of environment steps to train for |
| `learning_rate` | float | `3e-4` | Learning rate for both actor and critic Adam optimizers |
| `buffer_size` | int | `100000` | Maximum size of the replay buffer |
| `gamma` | float | `0.99` | Discount factor for future rewards |
| `tau` | float | `0.005` | Soft update coefficient for target networks (0 < τ ≤ 1) |
| `batch_size` | int | `256` | Batch size for training on replay buffer samples |
| `learning_starts` | int | `25000` | Number of steps to collect before starting training |
| `train_frequency` | int | `2` | How often to perform gradient updates (in steps) |
| `target_network_frequency` | int | `50` | How often to update target networks (in steps) |
| **Exploration** | | | |
| `exploration_fraction` | float | `0.1` | Fraction of total timesteps for exploration noise decay |
| `low` | float | `-1.0` | Lower bound for action space |
| `high` | float | `1.0` | Upper bound for action space |
| `noise_clip` | float | `0.5` | Maximum absolute value for exploration noise clipping |
| **Normalization** | | | |
| `normalize_obs` | bool | `False` | Whether to normalize observations using running statistics |
| `normalize_reward` | bool | `False` | Whether to normalize rewards using running statistics |
| **Logging & Evaluation** | | | |
| `seed` | int | `42` | Random seed for reproducibility |
| `exp_name` | str | `"DDPG-Experiment"` | Experiment name for logging and saving |
| `use_wandb` | bool | `True` | Whether to log metrics to Weights & Biases |
| `wandb_project` | str | `"cleanRL"` | W&B project name |
| `wandb_entity` | str | `""` | W&B username/team name |
| `capture_video` | bool | `True` | Whether to record evaluation videos |
| `eval_every` | int | `500` | Frequency of evaluation during training (in steps) |
| `save_every` | int | `10000` | Frequency of saving model checkpoints (in steps) |
| `num_eval_episodes` | int | `10` | Number of episodes to run during evaluation |
| `log_gradients` | bool | `True` | Whether to log gradient norms to W&B |
| **Network & Device** | | | |
| `actor_class` | nn.Module | `ActorNet` | Custom actor network class or instance |
| `critic_class` | nn.Module | `QNet` | Custom Q-network class or instance |
| `max_grad_norm` | float | `0.0` | Maximum gradient norm for clipping (0.0 to disable) |
| `device` | str | `"cpu"` | Device for training: "cpu", "cuda", "mps", or "auto" |

### Example: Custom Configuration

```python
from neatrl import train_ddpg

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

## Requirements

- PyTorch
- Gymnasium
- Stable Baselines 3 (for ReplayBuffer)
- Weights & Biases (optional)
- ale_py (for Atari environments)

## Examples

Check out these example scripts:

- [`run_ddpg_pendulum.py`](./run_ddpg_pendulum.py) - DDPG training on Pendulum
- [`run_ddpg_half_cheetah.py`](./run_ddpg_half_cheetah.py) - DDPG training on HalfCheetah
- [`run_ddpg_bipedal_walker.py`](./run_ddpg_bipedal_walker.py) - DDPG training on BipedalWalker
- [`run_ddpg_cnn_car_racing.py`](./run_ddpg_cnn_car_racing.py) - DDPG CNN training on CarRacing

## Installation

```bash
# Install base package
pip install neatrl

# Install extras based on environments you want to use
pip install neatrl[atari]      # For CarRacing
pip install neatrl[box2d]      # For BipedalWalker
pip install neatrl[classic]    # For Pendulum
pip install neatrl[mujoco]     # For HalfCheetah

# Or install all extras at once
pip install neatrl[atari,box2d,classic,mujoco]
```

## PyPI

For installation and more information, visit [NeatRL on PyPI](https://pypi.org/project/neatrl/)