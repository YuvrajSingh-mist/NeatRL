# NeatRL

**A clean Python library for reinforcement learning algorithms**

NeatRL provides readable implementations of popular RL algorithms with a focus on simplicity and ease of use. Built with PyTorch and designed for research and experimentation.

## Features

- **Single-file implementations**: Each algorithm lives entirely in one self-contained file -- no hunting across modules
- **Experiment Tracking**: Built-in support for Weights & Biases logging
- **Gymnasium Compatible**: Works with Gymnasium environments and adding many more!
- **Atari Support**: Full support for Atari games with automatic CNN architectures
- **Parallel Training**: Vectorized environments for faster data collection
- **Easy to Extend**: Modular design for adding new algorithms
- **Video Recording**: Automatic video capture and WandB integration


## Installation

```bash
pip install neatrl
```

With optional environment extras:

```bash
pip install "neatrl[atari]"    # Atari games (ALE)
pip install "neatrl[box2d]"    # Box2D environments (LunarLander, BipedalWalker, …)
pip install "neatrl[classic]"  # Classic control (CartPole, Pendulum, …)
pip install "neatrl[mujoco]"   # MuJoCo continuous control
```

> Browse supported environments: [MuJoCo](https://gymnasium.farama.org/environments/mujoco/) · [Box2D](https://gymnasium.farama.org/environments/box2d/) · [Classic Control](https://gymnasium.farama.org/environments/classic_control/) · [Atari](https://gymnasium.farama.org/environments/atari/)

## Quick Start

### Train DQN on CartPole

```python
from neatrl.dqn_mlp import train_dqn

model = train_dqn(
    env_id="CartPole-v1",
    total_timesteps=10000,
    seed=42
)
```

### Train PPO on Classic Control

```python
from neatrl.ppo_mlp import train_ppo

model = train_ppo(
    env_id="CartPole-v1",
    total_timesteps=50000,
    n_envs=4,           # Parallel environments
    GAE=0.95,           # Generalized Advantage Estimation lambda
    clip_value=0.2,     # PPO clipping parameter
    use_wandb=True,     # Track with WandB
    seed=42
)
```

### Train SAC on Continuous Control

```python
from neatrl.sac_mlp import train_sac

model = train_sac(
    env_id="Pendulum-v1",
    total_timesteps=50000,
    alpha=0.2,           # Entropy regularization coefficient
    autotune_alpha=True, # Automatically tune alpha
    use_wandb=True,      # Track with WandB
    seed=42
)
```

### Train SAC on Atari

```python
from neatrl.sac_cnn import train_sac_cnn

model = train_sac_cnn(
    env_id="BreakoutNoFrameskip-v4",
    total_timesteps=100000,
    alpha=0.2,
    autotune_alpha=True,
    atari_wrapper=True,  # Automatic Atari preprocessing
    use_wandb=True,
    seed=42
)
```


## Documentation

**[Complete Documentation](https://github.com/YuvrajSingh-mist/NeatRL/tree/master/docs)**

The docs include:
- Detailed usage examples for every algorithm
- Hyperparameter tuning guides
- Environment compatibility notes
- Experiment tracking setup (WandB)


## Supported Algorithms

### Current Implementations

- **DQN** (Deep Q-Network) - Classic value-based RL algorithm
  - Support for discrete action spaces
  - Experience replay and target networks
  - Atari preprocessing and frame stacking

- **Dueling DQN** - Enhanced DQN with separate value and advantage streams
  - Improved learning stability
  - Better performance on complex environments

- **REINFORCE** - Policy gradient method for discrete and continuous action spaces
  - Atari game support with automatic CNN architecture
  - Parallel environment training (`n_envs` support)
  - Continuous action space support
  - Episode-based Monte Carlo returns
  - Variance reduction through baseline subtraction

- **DDPG** (Deep Deterministic Policy Gradient) - Actor-critic method for continuous action spaces
  - Deterministic policy gradient for continuous control
  - Experience replay and target networks
  - Ornstein-Uhlenbeck noise for exploration
  - Support for exact continuous action spaces

- **A2C** (Advantage Actor-Critic) - Synchronous actor-critic algorithm
  - Advantage function for reduced variance
  - Support for both discrete and continuous action spaces
  - Parallel environment training with vectorized environments
  - Monte Carlo returns for value estimation

- **TD3** (Twin Delayed DDPG) - Actor-critic method for continuous control
  - Twin Q-networks to reduce overestimation bias
  - Delayed policy updates for improved stability
  - Target policy smoothing with noise
  - Experience replay and target networks
  - CNN support for image-based environments

- **SAC** (Soft Actor-Critic) - Maximum entropy reinforcement learning
  - Stochastic Gaussian policies with entropy regularization
  - Twin Q-networks for stable learning
  - Automatic entropy tuning (alpha parameter)
  - Balances exploration and exploitation
  - CNN support for complex environments

- **PPO** (Proximal Policy Optimization) - Policy gradient method with GAE
  - Full PPO implementation with Generalized Advantage Estimation (GAE)
  - Support for both discrete and continuous action spaces
  - Atari game support with automatic CNN architecture
  - Clipped surrogate objective for stable policy updates
  - Value function clipping and entropy regularization
  - Vectorized environments for parallel training

- **PPO-RND** (Proximal Policy Optimization with Random Network Distillation) - Curiosity-driven exploration
  - Intrinsic motivation through novelty detection
  - Combined extrinsic and intrinsic rewards for better exploration
  - Support for both discrete and continuous action spaces
  - PPO with clipped surrogate objective
  - Vectorized environments for parallel training
  - Intrinsic reward normalization and advantage calculation

- *More algorithms coming soon...*

## Contributing

Contributions are welcome. Please feel free to submit a Pull Request.

### Development Setup

```bash
git clone https://github.com/YuvrajSingh-mist/NeatRL.git
cd NeatRL/NeatRL
uv sync --extra dev   # installs neatrl in editable mode + dev tools
uv run pytest         # run the test suite
uv run ruff check src # lint
```

For the complete changelog, see [CHANGELOG.md](https://github.com/YuvrajSingh-mist/NeatRL/blob/master/CHANGELOG.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Made with ❤️ for the RL community**
