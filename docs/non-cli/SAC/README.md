# NeatRL Documentation - SAC (Soft Actor-Critic)

This guide shows you how to use NeatRL's SAC implementation for continuous and discrete action spaces.

## Quick Start with SAC

### CLI

```bash
neatrl train sac Pendulum-v1 --no-wandb
neatrl train sac-cnn BreakoutNoFrameskip-v4 --no-wandb --atari
```

### Python API

### Basic Training

```python
from neatrl.sac_mlp import train_sac

model = train_sac(
    env_id="Pendulum-v1",
    total_timesteps=10000,
    seed=42
)
```

### Training with Experiment Tracking

```python
from neatrl.sac_mlp import train_sac

model = train_sac(
    env_id="Pendulum-v1",
    total_timesteps=50000,
    seed=42,
    capture_video=True,
    use_wandb=True,
    wandb_project="my-rl-experiments",
    exp_name="sac-pendulum"
)
```

### SAC for Atari Environments

```python
from neatrl.sac_cnn import train_sac_cnn

model = train_sac_cnn(
    env_id="BreakoutNoFrameskip-v4",
    total_timesteps=100000,
    seed=42,
    atari_wrapper=True,
    use_wandb=True
)
```

## Function Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `env_id` | str | `"HalfCheetah-v5"` | Gymnasium environment ID |
| `total_timesteps` | int | `1000000` | Total number of environment steps |
| `seed` | int | `42` | Random seed |
| `learning_rate` | float | `3e-4` | Learning rate for the Adam optimizer |
| `buffer_size` | int | `1000000` | Size of the replay buffer |
| `gamma` | float | `0.99` | Discount factor |
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
| `wandb_entity` | str | `""` | W&B username/team |
| `eval_every` | int | `5000` | Evaluation frequency (steps) |
| `save_every` | int | `10000` | Model save frequency (steps) |
| `num_eval_episodes` | int | `10` | Number of evaluation episodes |
| `normalize_reward` | bool | `False` | Whether to normalize rewards |
| `normalize_obs` | bool | `False` | Whether to normalize observations |
| `atari_wrapper` | bool | `False` | Whether to apply Atari preprocessing |
| `env_wrapper` | Callable | `None` | Optional environment wrapper function |
| `grid_env` | bool | `False` | Whether the environment uses discrete grid observations |
| `n_envs` | int | `1` | Number of parallel environments |
| `max_grad_norm` | float | `0.0` | Maximum gradient norm for clipping |
| `log_gradients` | bool | `True` | Whether to log gradient norms to W&B |
| `device` | str | `"cpu"` | Training device ("cpu", "cuda", "mps") |

## Supported Environments

### Classic Control
- `Pendulum-v1`
- `MountainCarContinuous-v0`

### Box2D
- `BipedalWalker-v3`
- `LunarLanderContinuous-v2`

### MuJoCo
- `HalfCheetah-v5`
- `Ant-v5`
- `Humanoid-v5`

### Atari

```python
from neatrl.sac_cnn import train_sac_cnn

model = train_sac_cnn(
    env_id="BreakoutNoFrameskip-v4",
    total_timesteps=1000000,
    seed=42,
    alpha=0.2,
    autotune_alpha=True,
    atari_wrapper=True,
    use_wandb=True,
    wandb_project="atari-experiments",
    exp_name="sac-breakout"
)
```

## SAC Architecture

Soft Actor-Critic combines entropy regularization with actor-critic learning:

1. **Actor Network**: Learns a stochastic policy that maximizes expected return plus entropy
2. **Twin Q-Networks**: Two Q-value networks to reduce overestimation bias
3. **Automatic Entropy Tuning**: Adaptive temperature parameter for entropy regularization

The objective combines extrinsic rewards with policy entropy:

```
J(pi) = Sum E[(r + gamma*(r' + ...)) + alpha * H(pi(.|s))]
```

Where `H(pi(.|s))` is the entropy of the policy and `alpha` controls the exploration-exploitation trade-off.

### Key Components

- Reparameterization trick for stable policy updates
- Soft target network updates for stable Q-learning
- Twin critics to reduce overestimation bias

## Experiment Tracking with Weights & Biases

```python
model = train_sac(
    env_id="Pendulum-v1",
    total_timesteps=50000,
    use_wandb=True,
    wandb_project="my-rl-project",
    exp_name="pendulum-experiment"
)
```

What gets logged:
- Episode returns, Q-values, policy losses, entropy, alpha parameter
- Training progress videos (if `capture_video=True`)
- All training configuration

## Example Scripts

- `run_sac_pendulum.py` - SAC training on Pendulum
- `run_sac_halfcheetah.py` - SAC training on HalfCheetah
- `run_sac_cnn_car_racing.py` - SAC with CNN on CarRacing
- `run_sac_breakout.py` - SAC adapted for discrete actions on Breakout

## Installation

```bash
pip install neatrl

pip install neatrl[mujoco]     # For HalfCheetah, Ant, Humanoid
pip install neatrl[box2d]      # For BipedalWalker, LunarLander
pip install neatrl[classic]    # For Pendulum, MountainCar
pip install neatrl[atari]      # For Breakout
```

## PyPI

For installation and more information, visit [NeatRL on PyPI](https://pypi.org/project/neatrl/)
