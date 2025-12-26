# 🎯 NeatRL

**A clean, modern Python library for reinforcement learning algorithms**

NeatRL provides high-quality implementations of popular RL algorithms with a focus on simplicity, performance, and ease of use. Built with PyTorch and designed for both research and production use.

## ✨ Features

- 📊 **Experiment Tracking**: Built-in support for Weights & Biases logging
- 🎮 **Gymnasium Compatible**: Works with Gymnasium environments and adding many more!
- 🎯 **Atari Support**: Full support for Atari games with automatic CNN architectures
- ⚡ **Parallel Training**: Vectorized environments for faster data collection
- 🔧 **Easy to Extend**: Modular design for adding new algorithms
- 📈 **State-of-the-Art**: Implements modern RL techniques and best practices
- 🎥 **Video Recording**: Automatic video capture and WandB integration
- 📉 **Advanced Logging**: Per-layer gradient monitoring and comprehensive metrics

## 🏗️ Supported Algorithms

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
  - Synchronous version of A3C for stable training
  - Advantage function for reduced variance
  - Support for both discrete and continuous action spaces
  - Parallel environment training with vectorized environments
  - Monte Carlo returns for value estimation

- **PPO (Proximal Policy Optimization)** - State-of-the-art policy gradient method with GAE
  - Full PPO implementation with Generalized Advantage Estimation (GAE)
  - Support for both discrete and continuous action spaces
  - Atari game support with automatic CNN architecture
  - Clipped surrogate objective for stable policy updates
  - Value function clipping and entropy regularization
  - Vectorized environments for parallel training


- **PPO-RND** (Proximal Policy Optimization with Random Network Distillation) - State-of-the-art exploration method
  - Intrinsic motivation through novelty detection
  - Combined extrinsic and intrinsic rewards for better exploration
  - Support for both discrete and continuous action spaces
  - PPO with clipped surrogate objective
  - Vectorized environments for parallel training
  - Intrinsic reward normalization and advantage calculation
  
- *More algorithms coming soon...*

## 📦 Installation

```bash
python -m venv neatrl-env
source neatrl-env/bin/activate 

pip install neatrl"[classic,box2d,atari]"
```

## 🚀 Quick Start

### Train DQN on CartPole

```python
from neatrl import train_dqn

model = train_dqn(
    env_id="CartPole-v1",
    total_timesteps=10000,
    seed=42
)
```

### Train PPO on Classic Control

```python
from neatrl import train_ppo

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

### Train PPO on Atari

```python
from neatrl import train_ppo_cnn

model = train_ppo_cnn(
    env_id="BreakoutNoFrameskip-v4",
    total_timesteps=100000,
    n_envs=8,           # More parallel environments for Atari
    atari_wrapper=True, # Automatic Atari preprocessing
    use_wandb=True,     # Track with WandB
    seed=42
)
```

## 📚 Documentation

📖 **[Complete Documentation](https://github.com/YuvrajSingh-mist/NeatRL/tree/master/neatrl/docs)**

The docs include:
- Detailed usage examples
- Hyperparameter tuning guides
- Environment compatibility
- Experiment tracking setup
- Troubleshooting tips

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup
```bash
git clone https://github.com/YuvrajSingh-mist/NeatRL.git
cd NeatRL
pip install -e .[dev]
```

For the complete changelog, see [CHANGELOG.md](https://github.com/YuvrajSingh-mist/NeatRL/tree/master/neatrl/CHANGELOG.md).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Made with ❤️ for the RL community**