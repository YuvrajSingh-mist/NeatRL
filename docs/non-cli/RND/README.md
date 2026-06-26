# NeatRL Documentation - RND (Random Network Distillation)

This guide shows you how to use NeatRL's RND implementation, which combines PPO with intrinsic motivation for better exploration.

## Quick Start with RND

### CLI

```bash
neatrl train rnd CliffWalking-v0 --no-wandb --grid-env
neatrl train rnd-cnn BreakoutNoFrameskip-v4 --no-wandb --atari
```

### Python API

### Basic Training

```python
from neatrl.rnd_mlp import train_ppo_rnd

model = train_ppo_rnd(
    env_id="CliffWalking-v0",
    total_timesteps=10000,
    seed=42
)
```

### Training with Experiment Tracking

```python
from neatrl.rnd_mlp import train_ppo_rnd

model = train_ppo_rnd(
    env_id="CliffWalking-v0",
    total_timesteps=50000,
    seed=42,
    capture_video=True,
    use_wandb=True,
    wandb_project="my-rl-experiments",
    exp_name="rnd-ppo-cliffwalking"
)
```

### RND-PPO for Atari Environments

```python
from neatrl.rnd_cnn import train_ppo_rnd_cnn

model = train_ppo_rnd_cnn(
    env_id="CarRacing-v3",
    total_timesteps=500000,
    seed=42,
    n_envs=4,
    use_wandb=True
)
```

## Function Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `env_id` | str | `"CliffWalking-v0"` | Gymnasium environment ID |
| `total_timesteps` | int | `5000000` | Total number of environment steps |
| `seed` | int | `42` | Random seed |
| `lr` | float | `3e-4` | Learning rate |
| `ext_gamma` | float | `0.99` | Discount factor for extrinsic rewards |
| `int_gamma` | float | `0.95` | Discount factor for intrinsic rewards |
| `n_envs` | int | `1` | Number of parallel environments |
| `max_steps` | int | `128` | Maximum steps per rollout |
| `PPO_EPOCHS` | int | `4` | Number of PPO epochs per update |
| `clip_value` | float | `0.2` | PPO clipping value |
| `ENTROPY_COEFF` | float | `0.01` | Entropy coefficient |
| `VALUE_COEFF` | float | `0.5` | Value loss coefficient |
| `EXT_COEFF` | float | `2.0` | Extrinsic advantage coefficient |
| `INT_COEFF` | float | `1.0` | Intrinsic advantage coefficient |
| `capture_video` | bool | `True` | Whether to record training videos |
| `use_wandb` | bool | `True` | Whether to log to Weights & Biases |
| `wandb_project` | str | `"cleanRL"` | W&B project name |
| `exp_name` | str | `"PPO-RND-Vectorized-ClipWalking"` | Experiment name |
| `grid_env` | bool | `True` | Whether the environment uses discrete grid observations |
| `max_grad_norm` | float | `0.0` | Maximum gradient norm for clipping |
| `eval_every` | int | `10000` | Evaluation frequency (steps) |
| `save_every` | int | `10000` | Model save frequency (steps) |
| `atari_wrapper` | bool | `False` | Whether to apply Atari preprocessing |
| `num_minibatches` | int | `4` | Number of minibatches for PPO updates |
| `num_eval_episodes` | int | `5` | Number of evaluation episodes |
| `anneal_lr` | bool | `True` | Whether to anneal learning rate |
| `normalize_obs` | bool | `True` | Whether to normalize observations |
| `normalize_reward` | bool | `False` | Whether to normalize rewards |
| `device` | str | `"cpu"` | Training device ("cpu", "cuda", "mps") |
| `log_gradients` | bool | `True` | Whether to log gradient norms |
| `env_wrapper` | Callable | `None` | Optional environment wrapper |
| `actor_class` | Any | `ActorNet` | Custom actor network class |
| `critic_class` | Any | `CriticNet` | Custom critic network class |
| `predictor_class` | Any | `PredictorNet` | Custom predictor network class |
| `target_class` | Any | `TargetNet` | Custom target network class |

## Supported Environments

### Classic Control
- `CartPole-v1`
- `MountainCar-v0`
- `Acrobot-v1`

### Grid Environments

```python
from neatrl.rnd_mlp import train_ppo_rnd

model = train_ppo_rnd(
    env_id="CliffWalking-v0",
    total_timesteps=50000,
    seed=42,
    grid_env=True,
    use_wandb=True,
    exp_name="rnd-ppo-cliffwalking"
)
```

### Toy Text
- `FrozenLake-v1`
- `Taxi-v3`
- `CliffWalking-v0`

### Atari

```python
from neatrl.rnd_cnn import train_ppo_rnd_cnn

model = train_ppo_rnd_cnn(
    env_id="BreakoutNoFrameskip-v4",
    total_timesteps=1000000,
    seed=42,
    n_envs=8,
    atari_wrapper=True,
    use_wandb=True,
    exp_name="rnd-ppo-breakout"
)
```

## RND Architecture

Random Network Distillation generates intrinsic rewards by measuring prediction error:

1. **Target Network**: A randomly initialized network that remains fixed throughout training
2. **Predictor Network**: A learnable network that tries to predict the target network's output

The intrinsic reward is:
```
intrinsic_reward = ||predictor(state) - target(state)||^2
```

This encourages the agent to visit novel states where prediction error is high.

Combined advantage:
```
total_advantage = EXT_COEFF * ext_advantage + INT_COEFF * int_advantage
```

## Experiment Tracking with Weights & Biases

```python
model = train_ppo_rnd(
    env_id="CliffWalking-v0",
    total_timesteps=50000,
    use_wandb=True,
    wandb_project="my-rl-project",
    exp_name="cliffwalking-experiment"
)
```

What gets logged:
- Episode returns, intrinsic/extrinsic rewards, policy/value losses
- Network prediction errors, advantage distributions, gradient norms
- Training progress videos (if `capture_video=True`)

## Example Scripts

- `run_rnd_cliffwalking.py` - RND-PPO training on CliffWalking
- `run_rnd_fronzenlake.py` - RND-PPO training on FrozenLake
- `run_rnd_ppo_carracing.py` - RND-PPO with CNN on CarRacing
- `run_rnd_mountain_car.py` - RND-PPO training on MountainCar

## Installation

```bash
pip install neatrl

pip install neatrl[atari]      # For CarRacing
pip install neatrl[classic]    # For CliffWalking, FrozenLake, MountainCar
```

## PyPI

For installation and more information, visit [NeatRL on PyPI](https://pypi.org/project/neatrl/)
