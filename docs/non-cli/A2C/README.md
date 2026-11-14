# A2C (Actor-Critic)

A2C is an on-policy actor-critic algorithm. By default it weights the actor by
the advantage `A = G − V(s)` to cut variance in the policy gradient; pass
`advantage=False` to drop the baseline and weight by the raw return `G`
instead. Both the MLP trainer (vector observations) and the CNN trainer
(image observations) take the same flag.

Note: the built-in actor matches PPO — `Discrete` → categorical head,
`Box` → Normal head with a learnable `log_std`.

## Features

- **Standard A2C**: For environments with vector observations (e.g., Acrobot, CartPole)
- **CNN A2C**: For environments with image observations (e.g., CarRacing, Atari games)
- **Optional advantage baseline**: `advantage=True` (default) uses `A = G − V(s)`; `advantage=False` uses raw `G`
- **Separate Optimizers**: Independent optimization for actor and critic networks
- **W&B Integration**: Built-in logging and video recording
- **Episode-based Updates**: Pure on-policy learning with full episode rollouts

## Quick Start

### CLI

```bash
pip install neatrl

neatrl train a2c Acrobot-v1
neatrl train a2c-cnn CarRacing-v3
```

### Python API

## Usage

### Standard A2C (Vector Observations)

```python
from neatrl.a2c_mlp import train_a2c

# default — advantage actor-critic (A = G − V)
train_a2c(env_id="Acrobot-v1", total_timesteps=500000, lr=3e-4, use_wandb=True)

# classic actor-critic — no baseline (weight by G)
train_a2c(env_id="CartPole-v1", advantage=False, use_wandb=False)
```

### CNN A2C (Image Observations)

```python
from neatrl.a2c_cnn import train_a2c_cnn

train_a2c_cnn(
    env_id="CarRacing-v3",
    total_timesteps=500000,
    lr=3e-4,
    use_wandb=True,
    env_wrapper=car_racing_wrapper,
    actor_class=ActorNet,
    critic_class=CriticNet,
    advantage=True,  # set False to drop the V(s) baseline
)
```

## Network Architectures

### Actor Network (Standard)
- Input: State vector
- Hidden: 128 -> 128 -> 128 units with Tanh activation
- Output: Action probabilities (softmax for discrete) or action values (for continuous)

### Critic Network (Standard)
- Input: State vector
- Hidden: 128 -> 128 -> 128 units with Tanh activation
- Output: Single value estimate

### Actor Network (CNN)
- Input: Stacked frames (4, 84, 84)
- Conv layers: 32 -> 64 -> 64 filters
- FC: 512 units
- Output: Action probabilities/values

### Critic Network (CNN)
- Input: Stacked frames (4, 84, 84)
- Conv layers: 32 -> 64 -> 64 filters
- FC: 512 units
- Output: Single value estimate

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `exp_name` | "A2C" | Experiment name |
| `seed` | 42 | Random seed |
| `env_id` | "LunarLander-v3" | Gymnasium environment ID |
| `total_timesteps` | 1000000 | Total timesteps for training |
| `lr` | 2e-3 | Learning rate |
| `gamma` | 0.99 | Discount factor |
| `advantage` | True | Weight actor by `A = G − V(s)`; `False` uses raw return `G` |
| `VALUE_COEFF` | 0.5 | Value loss coefficient |
| `num_eval_episodes` | 10 | Number of evaluation episodes |
| `max_grad_norm` | 0.0 | Maximum gradient norm for clipping (0.0 to disable) |
| `max_steps` | 128 | Maximum steps per rollout |
| `n_envs` | 1 | Number of parallel environments |
| `update_epochs` | 1 | Update epochs (A2C uses 1) |
| `clip_value` | 0.2 | PPO clipping value (not used in A2C) |
| `ENTROPY_COEFF` | 0.01 | Entropy coefficient (not used in pure A2C) |
| `anneal_lr` | False | Learning rate annealing |
| `capture_video` | True | Whether to capture evaluation videos |
| `use_wandb` | True | Whether to use Weights & Biases |
| `wandb_project` | "cleanRL" | W&B project name |
| `grid_env` | False | Whether environment uses discrete grid observations |
| `eval_every` | 10000 | Evaluation frequency (in updates) |
| `save_every` | 10000 | Model save frequency (in updates) |
| `atari_wrapper` | False | Whether to apply Atari preprocessing |
| `normalize_obs` | False | Whether to normalize observations |
| `normalize_reward` | False | Whether to normalize rewards |
| `log_gradients` | False | Whether to log gradient norms to W&B |
| `device` | "cpu" | Training device |
| `custom_agent` | None | Custom neural network class or instance |
| `env` | None | Optional pre-created environment |

## Algorithm Details

Returns `G` are Monte Carlo estimates from the rollout. The critic always
regresses onto them. The actor weight depends on `advantage`:

**`advantage=True` (default)**

```
A_t = G_t − V(s_t)
L_policy = −mean(log π(a_t|s_t) · A_t)
L_value  = MSE(V(s_t), G_t)
```

**`advantage=False`**

```
L_policy = −mean(log π(a_t|s_t) · G_t)
L_value  = MSE(V(s_t), G_t)
```

With the flag off the critic still trains, but its prediction is not subtracted
from `G` for the actor update.

## Example Scripts

- `run_a2c_pendulum.py` - A2C training on Pendulum
- `run_a2c_lunarlander.py` - A2C training on LunarLander
- `run_a2c_frozen_lake.py` - A2C training on FrozenLake
- `run_a2c_reacher.py` - A2C training on Reacher
- `run_a2c_cnn_car_racing.py` - A2C CNN training on CarRacing

## Installation

```bash
pip install neatrl

pip install neatrl[atari]      # For CarRacing
pip install neatrl[box2d]      # For LunarLander
pip install neatrl[classic]    # For Pendulum, FrozenLake
pip install neatrl[mujoco]     # For Reacher
```

## PyPI

For installation and more information, visit [NeatRL on PyPI](https://pypi.org/project/neatrl/)
