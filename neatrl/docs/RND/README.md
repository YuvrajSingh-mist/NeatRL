# 🎯 NeatRL Documentation - RND (Random Network Distillation)

Welcome to the NeatRL documentation for RND (Random Network Distillation)! This guide shows you how to use NeatRL's RND implementation, which combines PPO with intrinsic motivation for better exploration in reinforcement learning.

## 🚀 Quick Start with RND

### Basic Training

Train an RND-PPO agent on CliffWalking in just a few lines:

```python
from neatrl.rnd import train_ppo_rnd

# Train RND-PPO on CliffWalking
model = train_ppo_rnd(
    env_id="CliffWalking-v0",
    total_timesteps=10000,
    seed=42
)

print("Training completed! 🎉")
```

### Training with Experiment Tracking

Enable Weights & Biases for experiment tracking and video recording:

```python
from neatrl.rnd import train_ppo_rnd

model = train_ppo_rnd(
    env_id="CliffWalking-v0",
    total_timesteps=50000,
    seed=42,
    capture_video=True,        # Record training videos
    use_wandb=True,           # Enable W&B logging
    wandb_project="my-rl-experiments",
    exp_name="rnd-ppo-cliffwalking"
)
```

### RND-PPO for Atari Environments

Train an RND-PPO agent on CarRacing with CNN:

```python
from neatrl.rnd import train_ppo_rnd_cnn

# Train RND-PPO with CNN on CarRacing
model = train_ppo_rnd_cnn(
    env_id="CarRacing-v3",
    total_timesteps=500000,
    seed=42,
    n_envs=4,
    use_wandb=True
)
```

## 🔧 Function Arguments

The `train_ppo_rnd` and `train_ppo_rnd_cnn` functions accept the following arguments for customizing your RND-PPO training:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `env_id` | str | `"CliffWalking-v0"` | Gymnasium environment ID to train on |
| `total_timesteps` | int | `5000000` | Total number of environment steps to train for |
| `seed` | int | `42` | Random seed for reproducibility |
| `lr` | float | `3e-4` | Learning rate for the Adam optimizer |
| `ext_gamma` | float | `0.99` | Discount factor for extrinsic rewards |
| `int_gamma` | float | `0.95` | Discount factor for intrinsic rewards |
| `n_envs` | int | `1` | Number of parallel environments for vectorized training |
| `max_steps` | int | `128` | Maximum steps per rollout before updating |
| `PPO_EPOCHS` | int | `4` | Number of PPO epochs per update |
| `clip_value` | float | `0.2` | PPO clipping value for policy ratio |
| `ENTROPY_COEFF` | float | `0.01` | Entropy coefficient for exploration |
| `VALUE_COEFF` | float | `0.5` | Value loss coefficient in total loss |
| `EXT_COEFF` | float | `2.0` | Extrinsic advantage coefficient for combined advantages |
| `INT_COEFF` | float | `1.0` | Intrinsic advantage coefficient for combined advantages |
| `capture_video` | bool | `True` | Whether to record training videos |
| `use_wandb` | bool | `True` | Whether to log to Weights & Biases |
| `wandb_project` | str | `"cleanRL"` | W&B project name |
| `exp_name` | str | `"PPO-RND-Vectorized-ClipWalking"` | Experiment name for logging |
| `grid_env` | bool | `True` | Whether the environment uses discrete grid observations |
| `max_grad_norm` | float | `0.0` | Maximum gradient norm for clipping (0.0 to disable) |
| `eval_every` | int | `10000` | Frequency of evaluation during training (in steps) |
| `save_every` | int | `10000` | Frequency of saving model checkpoints (in steps) |
| `atari_wrapper` | bool | `False` | Whether to apply Atari preprocessing wrappers |
| `num_minibatches` | int | `4` | Number of minibatches for PPO updates |
| `num_eval_episodes` | int | `5` | Number of episodes for evaluation |
| `anneal_lr` | bool | `True` | Whether to anneal learning rate over time |
| `normalize_obs` | bool | `True` | Whether to normalize observations |
| `normalize_reward` | bool | `False` | Whether to normalize rewards |
| `device` | str | `"cpu"` | Device for training ("cpu", "cuda", "mps") |
| `log_gradients` | bool | `True` | Whether to log gradient norms to W&B |
| `env_wrapper` | Callable | `None` | Optional environment wrapper function |
| `actor_class` | Any | `ActorNet` | Custom actor network class |
| `critic_class` | Any | `CriticNet` | Custom critic network class |
| `predictor_class` | Any | `PredictorNet` | Custom predictor network class |
| `target_class` | Any | `TargetNet` | Custom target network class |

## 🎮 Supported Environments

RND-PPO works with any Gymnasium environment. Here are some popular choices:

### Classic Control
- `CartPole-v1` - Balance a pole on a cart
- `MountainCar-v0` - Drive up a hill
- `Acrobot-v1` - Swing up a two-link robot

### Grid Environments
RND-PPO excels on grid-based environments with automatic one-hot encoding!

**Example: CliffWalking**

```python
from neatrl.rnd import train_ppo_rnd

# Train RND-PPO on CliffWalking with automatic one-hot encoding
model = train_ppo_rnd(
    env_id="CliffWalking-v0",
    total_timesteps=50000,
    seed=42,
    grid_env=False,  # Enable one-hot encoding for discrete states
    use_wandb=True,
    wandb_project="grid-experiments",
    exp_name="rnd-ppo-cliffwalking"
)
```

**Example: FrozenLake**

```python
from neatrl.rnd import train_ppo_rnd

# Train RND-PPO on FrozenLake with automatic one-hot encoding
model = train_ppo_rnd(
    env_id="FrozenLake-v1",
    total_timesteps=50000,
    seed=42,
    grid_env=False,  # Enable one-hot encoding for discrete states
    use_wandb=True,
    wandb_project="grid-experiments",
    exp_name="rnd-ppo-frozenlake"
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

### Atari
For Atari environments, use `train_ppo_rnd_cnn`:

```python
from neatrl.rnd import train_ppo_rnd_cnn

# Train RND-PPO with CNN on Breakout
model = train_ppo_rnd_cnn(
    env_id="BreakoutNoFrameskip-v4",
    total_timesteps=1000000,
    seed=42,
    n_envs=8,
    atari_wrapper=True,  # Apply Atari preprocessing
    use_wandb=True,
    wandb_project="atari-experiments",
    exp_name="rnd-ppo-breakout"
)
```

## 🏗️ RND Architecture

Random Network Distillation (RND) generates intrinsic rewards by measuring how well a learnable predictor network can predict the output of a fixed random target network:

1. **Target Network**: A randomly initialized network that remains fixed throughout training
2. **Predictor Network**: A learnable network that tries to predict the target network's output

The intrinsic reward is computed as: **intrinsic_reward = ||predictor(state) - target(state)||²**

This encourages the agent to visit novel states where the prediction error is high, providing curiosity-driven exploration.

RND-PPO combines this intrinsic motivation with PPO by:
- Computing extrinsic advantages from environment rewards
- Computing intrinsic advantages from RND rewards  
- Combining both advantages: **total_advantage = EXT_COEFF × ext_advantage + INT_COEFF × int_advantage**

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
model = train_ppo_rnd(
    env_id="CliffWalking-v0",
    total_timesteps=50000,
    use_wandb=True,
    wandb_project="my-rl-project",
    exp_name="cliffwalking-experiment"
)
```

### What gets logged:

- **Metrics**: Episode returns, episode lengths, intrinsic/extrinsic rewards, policy/value losses
- **Videos**: Training progress videos (recorded every 100 steps by default)
- **Hyperparameters**: All training configuration
- **System info**: Hardware usage, training time
- **Network stats**: Prediction errors, advantage distributions, gradient norms

## 🎥 Video Recording

NeatRL can automatically record and upload training videos:

```python
model = train_ppo_rnd(
    env_id="CliffWalking-v0",
    total_timesteps=50000,
    capture_video=True,     # Enable video recording
    use_wandb=True,         # Upload to W&B
    eval_every=1000         # Evaluation frequency (steps)
)
```

Videos are:
- Recorded during evaluation episodes
- Uploaded directly to W&B (no local storage)
- Automatically cleaned up after upload

## 📚 Examples

Check out these example scripts:

- [`run_rnd_cliffwalking.py`](./run_rnd_cliffwalking.py) - RND-PPO training on CliffWalking
- [`run_rnd_fronzenlake.py`](./run_rnd_fronzenlake.py) - RND-PPO training on FrozenLake
- [`run_rnd_ppo_carracing.py`](./run_rnd_ppo_carracing.py) - RND-PPO with CNN on CarRacing
- [`run_rnd_mountain_car.py`](./run_rnd_mountain_car.py) - RND-PPO training on MountainCar

## Installation

```bash
# Install base package
pip install neatrl

# Install extras based on environments you want to use
pip install neatrl[atari]      # For CarRacing
pip install neatrl[classic]    # For CliffWalking, FrozenLake, MountainCar

# Or install all extras at once
pip install neatrl[atari,classic]
```

---

Happy training with RND-PPO! 🚀

## PyPI

For installation and more information, visit [NeatRL on PyPI](https://pypi.org/project/neatrl/)