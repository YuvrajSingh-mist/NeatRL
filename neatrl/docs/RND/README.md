# 🎯 NeatRL Documentation - RND (Random Network Distillation)

Welcome to the NeatRL RND documentation! This guide shows you how to use NeatRL's RND algorithm implementations, including RND-QNet for discrete actions and RND-PPO for both discrete and continuous actions, with a focus on practical examples and best practices.

## 🚀 Quick Start with RND

### RND-QNet for Discrete Actions

Train an RND-QNet agent on Breakout:

```python
from neatrl.rnd import train_rnd_qnet

# Train RND-QNet on Breakout
model = train_rnd_qnet(
    env_id="BreakoutNoFrameskip-v4",
    total_timesteps=10000,
    seed=42,
    atari_wrapper=True,
    use_wandb=True
)
```

### RND-PPO for Atari Environments

Train an RND-PPO agent on CarRacing:

```python
from neatrl.rnd import train_rnd_ppo

# Train RND-PPO on CarRacing
model = train_rnd_ppo(
    env_id="CarRacing-v3",
    total_timesteps=500000,
    seed=42,
    num_envs=4,
    use_wandb=True
)
```

## 📋 Features

- **RND-QNet**: DQN with RND for exploration in discrete action spaces
- **RND-PPO**: PPO with RND for both discrete and continuous action spaces
- **Multi-environment support**: Parallel training with SyncVectorEnv
- **WandB logging**: Integrated experiment tracking
- **Video capture**: Record training and evaluation videos
- **Custom networks**: Support for custom neural network architectures

## 📁 Examples

- `run_rnd_qnet_breakout.py`: RND-QNet on Atari Breakout
- `run_rnd_ppo_carracing.py`: RND-PPO on CarRacing

Run any example with:

```bash
python run_rnd_qnet_breakout.py
```

## 🔧 Configuration

Both functions support extensive configuration via parameters. Key parameters include:

- `int_coeff`: Weight for intrinsic rewards
- `rnd_lr`: Learning rate for RND predictor
- `num_envs`: Number of parallel environments (for PPO)
- `use_wandb`: Enable WandB logging
- `capture_video`: Record videos

See the function docstrings for full parameter lists.