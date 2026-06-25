# NeatRL Documentation - PPO (Proximal Policy Optimization)

This guide shows you how to use NeatRL's PPO implementation for reinforcement learning across various environments.

## Quick Start with PPO

### Basic Training

```python
from neatrl.ppo_mlp import train_ppo

model = train_ppo(
    env_id="LunarLander-v2",
    total_timesteps=10000,
    seed=42
)
```

### Training with Experiment Tracking

```python
from neatrl.ppo_mlp import train_ppo

model = train_ppo(
    env_id="LunarLander-v2",
    total_timesteps=50000,
    seed=42,
    capture_video=True,
    use_wandb=True,
    wandb_project="my-rl-experiments",
    exp_name="ppo-lunar-lander"
)
```

### PPO for Discrete Action Spaces

```python
from neatrl.ppo_mlp import train_ppo

model = train_ppo(
    env_id="Taxi-v3",
    total_timesteps=100000,
    seed=42,
    n_envs=4,
    max_steps=128,
    use_wandb=True,
    wandb_project="cleanRL",
    exp_name="ppo-taxi"
)
```

### PPO for Continuous Action Spaces

```python
from neatrl.ppo_mlp import train_ppo

model = train_ppo(
    env_id="BipedalWalker-v3",
    total_timesteps=1000000,
    seed=42,
    n_envs=8,
    max_steps=128,
    use_wandb=True,
    wandb_project="cleanRL",
    exp_name="ppo-bipedal-walker"
)
```

### PPO for Atari Environments

```python
from neatrl.ppo_mlp import train_ppo

model = train_ppo(
    env_id="BreakoutNoFrameskip-v4",
    total_timesteps=10000000,
    seed=42,
    n_envs=8,
    max_steps=128,
    atari_wrapper=True,
    use_wandb=True,
    wandb_project="cleanRL",
    exp_name="ppo-breakout"
)
```

## Function Parameters

### `train_ppo()` Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `env_id` | str | Required | Gymnasium environment ID |
| `total_timesteps` | int | Required | Total training timesteps |
| `seed` | int | 42 | Random seed for reproducibility |
| `lr` | float | 3e-4 | Learning rate |
| `gamma` | float | 0.99 | Discount factor |
| `GAE` | float | 0.95 | Generalized Advantage Estimation lambda |
| `n_envs` | int | 4 | Number of parallel environments |
| `max_steps` | int | 128 | Steps per rollout per environment |
| `num_minibatches` | int | 4 | Number of minibatches for PPO updates |
| `PPO_EPOCHS` | int | 4 | Number of PPO update epochs |
| `clip_value` | float | 0.2 | PPO clipping parameter |
| `VALUE_COEFF` | float | 0.5 | Value loss coefficient |
| `ENTROPY_COEFF` | float | 0.01 | Entropy bonus coefficient |
| `max_grad_norm` | float | 0.5 | Gradient clipping threshold |
| `value_clip` | bool | False | Enable value function clipping |
| `anneal_lr` | bool | True | Linear learning rate annealing |
| `capture_video` | bool | False | Record training videos |
| `use_wandb` | bool | False | Enable Weights & Biases logging |
| `wandb_project` | str | "cleanRL" | W&B project name |
| `exp_name` | str | None | Experiment name |
| `eval_every` | int | 1000 | Evaluation frequency (timesteps) |
| `save_every` | int | 1000 | Model save frequency (timesteps) |
| `num_eval_episodes` | int | 10 | Number of evaluation episodes |
| `grid_env` | bool | False | Enable grid environment wrapper |
| `atari_wrapper` | bool | False | Enable Atari preprocessing wrapper |
| `env_wrapper` | callable | None | Custom environment wrapper |
| `actor_class` | class | ActorNet | Custom actor network class |
| `critic_class` | class | CriticNet | Custom critic network class |
| `log_gradients` | bool | False | Log gradient norms |
| `normalize_obs` | bool | False | Normalize observations |
| `normalize_reward` | bool | False | Normalize rewards |
| `device` | str | "cpu" | Training device ("cpu", "cuda", etc.) |

## Environment-Specific Examples

### Lunar Lander (Discrete Actions)

```python
from neatrl.ppo_mlp import train_ppo

model = train_ppo(
    env_id="LunarLander-v2",
    total_timesteps=200000,
    seed=42,
    n_envs=4,
    max_steps=512,
    use_wandb=True,
    exp_name="ppo-lunar-lander"
)
```

### Taxi (Discrete Actions)

```python
from neatrl.ppo_mlp import train_ppo

model = train_ppo(
    env_id="Taxi-v3",
    total_timesteps=100000,
    seed=42,
    n_envs=4,
    max_steps=128,
    use_wandb=True,
    exp_name="ppo-taxi"
)
```

### Breakout (Atari, Discrete Actions)

```python
from neatrl.ppo_mlp import train_ppo

model = train_ppo(
    env_id="BreakoutNoFrameskip-v4",
    total_timesteps=10000000,
    seed=42,
    n_envs=8,
    max_steps=128,
    atari_wrapper=True,
    use_wandb=True,
    exp_name="ppo-breakout"
)
```

### Bipedal Walker (Continuous Actions)

```python
from neatrl.ppo_mlp import train_ppo

model = train_ppo(
    env_id="BipedalWalker-v3",
    total_timesteps=2000000,
    seed=42,
    n_envs=8,
    max_steps=512,
    use_wandb=True,
    exp_name="ppo-bipedal-walker"
)
```

## Monitoring Training

### Weights & Biases Integration

When `use_wandb=True`, PPO logs:

- **Losses**: Policy loss, value loss, entropy loss, total loss
- **Rewards**: Episodic returns, extrinsic rewards
- **Advantages**: Mean and std of advantages
- **Gradients**: Gradient norms (if `log_gradients=True`)
- **Approximate KL**: Policy update magnitude
- **Clip Fraction**: Percentage of clipped policy updates
- **Videos**: Training progress videos (if `capture_video=True`)

### Evaluation Metrics

- Average episodic return during evaluation
- Training SPS (steps per second)
- Learning rate annealing progress

## Advanced Usage

### Custom Environment Wrappers

```python
from neatrl.ppo_mlp import train_ppo
import gymnasium as gym

def custom_wrapper(env):
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    return env

model = train_ppo(
    env_id="YourCustomEnv-v0",
    total_timesteps=100000,
    env_wrapper=custom_wrapper
)
```

### Grid Environments

```python
from neatrl.ppo_mlp import train_ppo

model = train_ppo(
    env_id="CliffWalking-v0",
    total_timesteps=50000,
    grid_env=True,
    use_wandb=True
)
```

### Value Function Clipping

```python
from neatrl.ppo_mlp import train_ppo

model = train_ppo(
    env_id="LunarLander-v2",
    total_timesteps=200000,
    value_clip=True,
    clip_value=0.2,
    use_wandb=True
)
```

## Example Scripts

- `run_ppo_lunar_lander.py` - Lunar Lander training
- `run_ppo_taxi.py` - Taxi environment training
- `run_ppo_breakout.py` - Breakout Atari training
- `run_ppo_bipedal_walker.py` - Bipedal Walker training

## Tips

1. Start with simpler environments like Lunar Lander before scaling up
2. Adjust `clip_value`, `VALUE_COEFF`, and `ENTROPY_COEFF` based on your environment
3. Try `value_clip=True` for more stable value function learning
4. Use `max_grad_norm=0.5` to prevent gradient explosions
5. More `n_envs` means faster wall-clock training on vectorized environments
6. Enable `anneal_lr` for stable training over long horizons

## Installation

```bash
pip install neatrl

pip install neatrl[atari]      # For Breakout
pip install neatrl[box2d]      # For LunarLander, BipedalWalker
pip install neatrl[classic]    # For Taxi
```

## PyPI

For installation and more information, visit [NeatRL on PyPI](https://pypi.org/project/neatrl/)
