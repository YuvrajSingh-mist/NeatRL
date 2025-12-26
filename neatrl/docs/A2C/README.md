# A2C (Advantage Actor-Critic)

A2C is an on-policy actor-critic algorithm that uses advantage estimates to reduce variance in policy gradient updates. This implementation includes both standard A2C for vector observations and CNN-based A2C for image observations.

**Note**: A2C works with both discrete and continuous action spaces, making it versatile for various environments.

## Features

- **Standard A2C**: For environments with vector observations (e.g., Acrobot, CartPole)
- **CNN A2C**: For environments with image observations (e.g., CarRacing, Atari games)
- **Advantage Estimation**: Uses Monte Carlo returns for advantage calculation
- **Separate Optimizers**: Independent optimization for actor and critic networks
- **W&B Integration**: Built-in logging and video recording
- **Episode-based Updates**: Pure on-policy learning with full episode rollouts

## Usage

### Standard A2C (Vector Observations)

```python
from neatrl.a2c import train_a2c

train_a2c(
    env_id="Acrobot-v1",
    total_timesteps=500000,
    lr=3e-4,
    use_wandb=True
)
```

### CNN A2C (Image Observations)

```python
from neatrl.a2c import train_a2c_cnn

train_a2c_cnn(
    env_id="CarRacing-v3",
    total_timesteps=500000,
    lr=3e-4,
    use_wandb=True,
    env_wrapper=car_racing_wrapper,
    actor_class=ActorNet,
    critic_class=CriticNet
)
```

## Network Architectures

### Actor Network (Standard)
- Input: State vector
- Hidden: 128 → 128 → 128 units with Tanh activation
- Output: Action probabilities (softmax for discrete) or action values (for continuous)

### Critic Network (Standard)
- Input: State vector
- Hidden: 128 → 128 → 128 units with Tanh activation
- Output: Single value estimate

### Actor Network (CNN)
- Input: Stacked frames (4, 84, 84)
- Conv layers: 32 → 64 → 64 filters
- FC: 512 units
- Output: Action probabilities/values

### Critic Network (CNN)
- Input: Stacked frames (4, 84, 84)
- Conv layers: 32 → 64 → 64 filters
- FC: 512 units
- Output: Single value estimate

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| **Experiment Settings** | | |
| `exp_name` | "A2C" | Experiment name for logging and saving |
| `seed` | 42 | Random seed for reproducibility |
| `env_id` | "LunarLander-v3" | Gymnasium environment ID |
| `total_timesteps` | 1000000 | Total timesteps for training |
| **A2C & Agent Settings** | | |
| `lr` | 2e-3 | Learning rate for optimizer |
| `gamma` | 0.99 | Discount factor for rewards |
| `VALUE_COEFF` | 0.5 | Value loss coefficient (not used in pure A2C) |
| `num_eval_episodes` | 10 | Number of evaluation episodes |
| `max_grad_norm` | 0.0 | Maximum gradient norm for clipping (0.0 to disable) |
| `max_steps` | 128 | Maximum steps per rollout (for safety) |
| **Compatibility Settings** | | |
| `n_envs` | 1 | Number of parallel environments |
| `update_epochs` | 1 | Update epochs (A2C uses 1) |
| `clip_value` | 0.2 | PPO clipping value (not used in A2C) |
| `ENTROPY_COEFF` | 0.01 | Entropy coefficient (not used in pure A2C) |
| `anneal_lr` | False | Learning rate annealing |
| **Logging & Saving** | | |
| `capture_video` | True | Whether to capture evaluation videos |
| `use_wandb` | True | Whether to use Weights & Biases for logging |
| `wandb_project` | "cleanRL" | W&B project name |
| `grid_env` | False | Whether environment uses discrete grid observations |
| `eval_every` | 10000 | Frequency of evaluation (in updates) |
| `save_every` | 10000 | Frequency of saving the model (in updates) |
| `atari_wrapper` | False | Whether to apply Atari preprocessing |
| `normalize_obs` | False | Whether to normalize observations |
| `normalize_reward` | False | Whether to normalize rewards |
| `log_gradients` | False | Whether to log gradient norms to W&B |
| `device` | "cpu" | Device for training: "auto", "cpu", "cuda", etc. |
| `custom_agent` | None | Custom neural network class or instance |
| `env` | None | Optional pre-created environment |

## Examples

### Acrobot Environment
```bash
cd docs/A2C
python run_a2c_acrobot.py
```

### CarRacing with CNN
```bash
cd docs/A2C
python run_a2c_cnn_car_racing.py
```

## Algorithm Details

A2C uses the following update rule:

**Policy Loss**:
```
L_policy = -mean(log_prob * advantage)
```

**Value Loss**:
```
L_value = MSE(value_estimate, monte_carlo_return)
```

**Advantages**:
```
A(s,a) = Q(s,a) - V(s) = Return - V(s)
```

Where returns are computed using Monte Carlo estimation from full episode rollouts.

## Example Scripts

Check out these example scripts:

- [`run_a2c_pendulum.py`](./run_a2c_pendulum.py) - A2C training on Pendulum
- [`run_a2c_lunarlander.py`](./run_a2c_lunarlander.py) - A2C training on LunarLander
- [`run_a2c_frozen_lake.py`](./run_a2c_frozen_lake.py) - A2C training on FrozenLake
- [`run_a2c_reacher.py`](./run_a2c_reacher.py) - A2C training on Reacher
- [`run_a2c_cnn_car_racing.py`](./run_a2c_cnn_car_racing.py) - A2C CNN training on CarRacing

## Installation

```bash
# Install base package
pip install neatrl

# Install extras based on environments you want to use
pip install neatrl[atari]      # For CarRacing
pip install neatrl[box2d]      # For LunarLander
pip install neatrl[classic]    # For Pendulum, FrozenLake
pip install neatrl[mujoco]     # For Reacher

# Or install all extras at once
pip install neatrl[atari,box2d,classic,mujoco]
```

## PyPI

For installation and more information, visit [NeatRL on PyPI](https://pypi.org/project/neatrl/)