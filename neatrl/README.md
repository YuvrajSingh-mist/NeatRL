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
  - **NEW**: Atari game support with automatic CNN architecture
  - **NEW**: Parallel environment training (`n_envs` support)
  - **NEW**: Continuous action space support
  - **NEW**: Per-layer gradient logging
  - Episode-based Monte Carlo returns
  - Variance reduction through baseline subtraction

- **PPO (Proximal Policy Optimization)** - State-of-the-art policy gradient method with GAE
  - **NEW**: Full PPO implementation with Generalized Advantage Estimation (GAE)
  - **NEW**: Support for both discrete and continuous action spaces
  - **NEW**: Atari game support with automatic CNN architecture (`train_ppo_cnn`)
  - **NEW**: Clipped surrogate objective for stable policy updates
  - **NEW**: Value function clipping and entropy regularization
  - **NEW**: Vectorized environments for parallel training
  - **NEW**: Comprehensive WandB logging with advantage distributions
  - **NEW**: Per-layer gradient monitoring and video recording
  - Generalized Advantage Estimation with configurable lambda
  - Flexible network architecture with custom actor/critic classes

- **PPO-RND** (Proximal Policy Optimization with Random Network Distillation) - State-of-the-art exploration method
  - **NEW**: Intrinsic motivation through novelty detection
  - **NEW**: Combined extrinsic and intrinsic rewards for better exploration
  - **NEW**: Support for both discrete and continuous action spaces
  - **NEW**: Automatic render mode handling for video recording
  - **NEW**: Comprehensive WandB logging with global step tracking
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

### Train REINFORCE on Atari

```python
from neatrl import train_reinforce

model = train_reinforce(
    env_id="BreakoutNoFrameskip-v4",
    total_steps=2000,
    atari_wrapper=True,  # Automatic Atari preprocessing
    n_envs=4,            # Parallel environments
    use_wandb=True,      # Track with WandB
    seed=42
)
```

### Train REINFORCE with Continuous Actions

```python
from neatrl import train_reinforce
import torch.nn as nn

# Custom policy for continuous actions
class ContinuousPolicyNet(nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.fc1 = nn.Linear(state_space, 32)
        self.fc2 = nn.Linear(32, 16)
        self.mean = nn.Linear(16, action_space)
        self.logstd = nn.Linear(16, action_space)
    
    def forward(self, x):
        x = torch.relu(self.fc2(torch.relu(self.fc1(x))))
        return self.mean(x), torch.exp(self.logstd(x))
    
    def get_action(self, x):
        mean, std = self.forward(x)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        return action, dist.log_prob(action).sum(dim=-1)

model = train_reinforce(
    env_id="Pendulum-v1",
    total_steps=2000,
    custom_agent=ContinuousPolicyNet(3, 1),
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

## 📋 Changelog

### [0.2.1] - 2025-12-17
- **Added**: REINFORCE Atari support with automatic CNN architecture
- **Added**: Parallel environment training (`n_envs` parameter)
- **Added**: Continuous action space support for REINFORCE
- **Added**: Advanced gradient logging (per-layer norms, clip ratios)
- **Changed**: REINFORCE parameter `episodes` → `total_steps`
- **Fixed**: Multi-environment action handling for vectorized training

### [0.2.0] - 2025-12-14
- **Added**: Grid environment support with automatic one-hot encoding
- **Changed**: Renamed `record` to `capture_video` for consistency

### [0.1.4] - 2025-12-13
- **Added**: Custom agent support for DQN training
- **Added**: Network architecture display using torchinfo
- **Improved**: Error handling for custom agent constructors
- **Changed**: Agent parameter now accepts nn.Module subclasses

### [0.1.3] - 2025-12-01
- Initial release with DQN implementation
- Weights & Biases integration
- Video recording capabilities
- Comprehensive documentation

For the complete changelog, see [CHANGELOG.md](CHANGELOG.md).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Made with ❤️ for the RL community**