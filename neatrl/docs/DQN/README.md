# 🎯 NeatRL Documentation

Welcome to the NeatRL documentation! This guide shows you how to use NeatRL's reinforcement learning algorithms, with a focus on practical examples and best practices.

## 🚀 Quick Start with DQN

### Basic Training

Train a DQN agent on CartPole in just a few lines:

```python
from neatrl import train_dqn

# Train DQN on CartPole
model = train_dqn(
    env_id="CartPole-v1",
    total_timesteps=10000,
    seed=42
)

print("Training completed! 🎉")
```

### Training with Experiment Tracking

Enable Weights & Biases for experiment tracking and video recording:

```python
from neatrl import train_dqn

model = train_dqn(
    env_id="CartPole-v1",
    total_timesteps=50000,
    seed=42,
    capture_video=True,        # Record training videos
    use_wandb=True,           # Enable W&B logging
    wandb_project="my-rl-experiments",
    exp_name="dqn-cartpole-v1"
)
```

### Advanced Configuration

Fine-tune your DQN training with custom hyperparameters:

```python
from neatrl import train_dqn

model = train_dqn(
    env_id="LunarLander-v2",
    total_timesteps=100000,
    seed=42,

    # Network hyperparameters
    learning_rate=2.5e-4,
    gamma=0.99,
    tau=1.0,

    # Training hyperparameters
    buffer_size=100000,
    batch_size=128,
    target_network_frequency=500,

    # Exploration
    start_e=1.0,
    end_e=0.05,
    exploration_fraction=0.5,

    # Training schedule
    learning_starts=1000,
    train_frequency=4,

    # Monitoring & logging
    capture_video=True,
    use_wandb=True,
    wandb_project="lunar-lander-experiments",
    exp_name="dqn-lunar-lander",

    # Evaluation & saving
    eval_every=1000,
    save_every=5000,
    upload_every=100,  # Upload videos every 100 steps
)
```

## 🎮 Supported Environments

DQN works with any Gymnasium environment. Here are some popular choices:

### Classic Control
- `CartPole-v1` - Balance a pole on a cart
<!-- - `MountainCar-v0` - Drive up a hill -->
<!-- - `Acrobot-v1` - Swing up a two-link robot -->
<!-- - `Pendulum-v1` - Swing up a pendulum -->
<!-- 
### Box2D
- `LunarLander-v2` - Land a spacecraft safely
- `BipedalWalker-v3` - Make a robot walk
- `CarRacing-v2` - Race a car around a track

### Atari Games
- `ALE/Pong-v5` - Classic Pong game
- `ALE/Breakout-v5` - Breakout game
- `ALE/SpaceInvaders-v5` - Space Invaders -->

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
model = train_dqn(
    env_id="CartPole-v1",
    total_timesteps=50000,
    use_wandb=True,
    wandb_project="my-rl-project",
    wandb_entity="your-username",  # Optional: your W&B username
    exp_name="cartpole-experiment"
)
```

### What gets logged:

- **Metrics**: Episode returns, episode lengths, epsilon values, training loss
- **Videos**: Training progress videos (recorded every 100 steps by default)
- **Hyperparameters**: All training configuration
- **System info**: Hardware usage, training time

## 🎥 Video Recording

NeatRL can automatically record and upload training videos:

```python
model = train_dqn(
    env_id="CartPole-v1",
    total_timesteps=50000,
    capture_video=True,     # Enable video recording
    use_wandb=True,         # Upload to W&B
    upload_every=100        # Upload frequency (steps)
)
```

Videos are:
- Recorded every 100 training steps (configurable)
- Uploaded directly to W&B (no local storage)
- Automatically cleaned up after upload


## 📚 Examples

Check out these example scripts:

- [`run_dqn_cartpole.py`](./run_dqn_cartpole.py) - Basic DQN training on CartPole
- ['run_dqn_atari.py`](./run_dqn_atari.py) - DQN training on Atari-Breakout
- *More examples coming soon...*

---

Happy training! 🚀</content>
<parameter name="filePath">/Users/yuvrajsingh9886/Desktop/NeatRL/NeatRL/neatrl/docs/README.md
