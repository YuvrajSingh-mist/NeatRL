# 🎯 NeatRL Documentation - SAC (Soft Actor-Critic)

Welcome to the NeatRL documentation for SAC (Soft Actor-Critic)! This guide shows you how to use NeatRL's SAC implementation, which provides maximum entropy reinforcement learning for both continuous and discrete action spaces.

## 🚀 Quick Start with SAC

### Basic Training

Train a SAC agent on Pendulum in just a few lines:

```python
from neatrl.sac import train_sac

# Train SAC on Pendulum
model = train_sac(
    env_id="Pendulum-v1",
    total_timesteps=10000,
    seed=42
)

print("Training completed! 🎉")
```

### Training with Experiment Tracking

Enable Weights & Biases for experiment tracking and video recording:

```python
from neatrl.sac import train_sac

model = train_sac(
    env_id="Pendulum-v1",
    total_timesteps=50000,
    seed=42,
    capture_video=True,        # Record training videos
    use_wandb=True,           # Enable W&B logging
    wandb_project="my-rl-experiments",
    exp_name="sac-pendulum"
)
```

### SAC for Atari Environments

Train a SAC agent on Breakout with CNN:

```python
from neatrl.sac import train_sac_cnn

# Train SAC with CNN on Breakout
model = train_sac_cnn(
    env_id="BreakoutNoFrameskip-v4",
    total_timesteps=100000,
    seed=42,
    atari_wrapper=True, # Automatic Atari preprocessing
    use_wandb=True
)
```

## 🔧 Function Arguments

The `train_sac` and `train_sac_cnn` functions accept the following arguments for customizing your SAC training:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `env_id` | str | `"HalfCheetah-v5"` | Gymnasium environment ID to train on |
| `total_timesteps` | int | `1000000` | Total number of environment steps to train for |
| `seed` | int | `42` | Random seed for reproducibility |
| `learning_rate` | float | `3e-4` | Learning rate for the Adam optimizer |
| `buffer_size` | int | `1000000` | Size of the replay buffer |
| `gamma` | float | `0.99` | Discount factor for future rewards |
| `tau` | float | `0.005` | Soft update parameter for target networks |
| `target_network_frequency` | int | `1` | How often to update target networks |
| `batch_size` | int | `256` | Batch size for training |
| `alpha` | float | `0.2` | Entropy regularization coefficient |
| `autotune_alpha` | bool | `True` | Whether to automatically tune alpha |
| `target_entropy_scale` | float | `-1.0` | Target entropy multiplier |
| `learning_starts` | int | `5000` | Steps before training begins |
| `policy_frequency` | int | `1` | How often to update the policy |
| `capture_video` | bool | `True` | Whether to record training videos |
| `use_wandb` | bool | `True` | Whether to log to Weights & Biases |
| `wandb_project` | str | `"cleanRL"` | W&B project name |
| `wandb_entity` | str | `""` | Your WandB username/team |
| `eval_every` | int | `5000` | Frequency of evaluation during training |
| `save_every` | int | `10000` | Frequency of saving model checkpoints |
| `num_eval_episodes` | int | `10` | Number of episodes for evaluation |
| `normalize_reward` | bool | `False` | Whether to normalize rewards |
| `normalize_obs` | bool | `False` | Whether to normalize observations |
| `atari_wrapper` | bool | `False` | Whether to apply Atari preprocessing wrappers |
| `env_wrapper` | Callable | `None` | Optional environment wrapper function |
| `grid_env` | bool | `False` | Whether the environment uses discrete grid observations |
| `n_envs` | int | `1` | Number of parallel environments |
| `max_grad_norm` | float | `0.0` | Maximum gradient norm for clipping |
| `log_gradients` | bool | `True` | Whether to log gradient norms to W&B |
| `device` | str | `"cpu"` | Device for training ("cpu", "cuda", "mps") |

## 🎮 Supported Environments

SAC works with any Gymnasium environment. Here are some popular choices:

### Classic Control
- `Pendulum-v1` - Balance a pendulum with continuous torque
- `MountainCarContinuous-v0` - Drive up a hill with continuous actions

### Box2D
- `BipedalWalker-v3` - Walk forward with continuous joint torques
- `LunarLanderContinuous-v2` - Land a spacecraft with continuous controls

### MuJoCo
- `HalfCheetah-v5` - Run forward with continuous joint controls
- `Ant-v5` - Walk forward with continuous joint controls
- `Humanoid-v5` - Walk forward with full humanoid control

## Installation

```bash
# Install base package
pip install neatrl

# Install extras based on environments you want to use
pip install neatrl[mujoco]     # For HalfCheetah, Ant, Humanoid
pip install neatrl[box2d]      # For BipedalWalker, LunarLander
pip install neatrl[classic]    # For Pendulum, MountainCar
pip install neatrl[atari]      # For CarRacing, Breakout

# Or install all extras at once
pip install neatrl[atari,box2d,classic,mujoco]
```

---
### Atari
For Atari environments, use `train_sac_cnn`:

```python
from neatrl.sac import train_sac_cnn

# Train SAC with CNN on Breakout
model = train_sac_cnn(
    env_id="BreakoutNoFrameskip-v4",
    total_timesteps=1000000,
    seed=42,
    alpha=0.2,
    autotune_alpha=True,
    atari_wrapper=True, # Automatic Atari preprocessing
    use_wandb=True,
    wandb_project="atari-experiments",
    exp_name="sac-breakout"
)
```

## 🏗️ SAC Architecture

Soft Actor-Critic (SAC) is a maximum entropy reinforcement learning algorithm that balances exploration and exploitation through entropy regularization:

1. **Actor Network**: Learns a stochastic policy π(a|s) that maximizes expected return plus entropy
2. **Twin Q-Networks**: Two Q-value networks to reduce overestimation bias
3. **Automatic Entropy Tuning**: Adaptive temperature parameter α for entropy regularization

The SAC objective combines extrinsic rewards with policy entropy:

**J(π) = Σ E[(r + γ(r' + ...)) + α * H(π(·|s))]**

Where:
- `H(π(·|s))` is the entropy of the policy at state s
- `α` controls the exploration-exploitation trade-off

### Key Components

- **Reparameterization Trick**: Stable policy updates with reparameterized sampling
- **Target Networks**: Soft updates for stable Q-learning
- **Twin Critics**: Reduces overestimation bias in Q-value estimation
- **Entropy Regularization**: Encourages exploration through stochastic policies

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
model = train_sac(
    env_id="Pendulum-v1",
    total_timesteps=50000,
    use_wandb=True,
    wandb_project="my-rl-project",
    exp_name="pendulum-experiment"
)
```

### What gets logged:

- **Metrics**: Episode returns, episode lengths, Q-values, policy losses, entropy, alpha parameter
- **Videos**: Training progress videos (recorded during evaluation)
- **Hyperparameters**: All training configuration
- **System info**: Hardware usage, training time
- **Network stats**: Gradient norms, Q-value distributions

## 🎥 Video Recording

NeatRL can automatically record and upload training videos:

```python
model = train_sac(
    env_id="Pendulum-v1",
    total_timesteps=50000,
    capture_video=True,     # Enable video recording
    use_wandb=True,         # Upload to W&B
    eval_every=5000         # Evaluation frequency (steps)
)
```

Videos are:
- Recorded during evaluation episodes
- Uploaded directly to W&B (no local storage)
- Automatically cleaned up after upload

## 📚 Examples

Check out these example scripts:

- [`run_sac_pendulum.py`](./run_sac_pendulum.py) - SAC training on Pendulum
- [`run_sac_halfcheetah.py`](./run_sac_halfcheetah.py) - SAC training on HalfCheetah
- [`run_sac_cnn_car_racing.py`](./run_sac_cnn_car_racing.py) - SAC with CNN on CarRacing
- [`run_sac_breakout.py`](./run_sac_breakout.py) - SAC adapted for discrete actions on Breakout


Happy training with SAC! 🚀

## PyPI

For installation and more information, visit [NeatRL on PyPI](https://pypi.org/project/neatrl/)
