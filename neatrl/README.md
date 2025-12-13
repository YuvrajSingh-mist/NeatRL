# 🎯 NeatRL

[![CI](https://github.com/YuvrajSingh-mist/NeatRL/actions/workflows/ci.yml/badge.svg)](https://github.com/YuvrajSingh-mist/NeatRL/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/neatrl.svg)](https://pypi.org/project/neatrl/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**A clean, modern Python library for reinforcement learning algorithms**

NeatRL provides high-quality implementations of popular RL algorithms with a focus on simplicity, performance, and ease of use. Built with PyTorch and designed for both research and production use.

## ✨ Features

- 🚀 **Fast & Efficient**: Optimized implementations using PyTorch
- 🎯 **Production Ready**: Clean APIs and comprehensive error handling
- 📊 **Experiment Tracking**: Built-in support for Weights & Biases logging
- 🎮 **Gymnasium Compatible**: Works with all Gymnasium environments
- 🔧 **Easy to Extend**: Modular design for adding new algorithms
- 📈 **State-of-the-Art**: Implements modern RL techniques and best practices

## 🏗️ Supported Algorithms

### Current Implementations
- **DQN** (Deep Q-Network) - Classic value-based RL algorithm
- *More algorithms coming soon...*

## 📦 Installation

### From PyPI (Recommended)
```bash
pip install neatrl
```

### From Source (Development)
```bash
git clone https://github.com/YuvrajSingh-mist/NeatRL.git
cd NeatRL
pip install -e .
```

### Development Setup
```bash
pip install -e .[dev]  # Includes testing and development tools
```

## 🚀 Quick Start

### Train DQN on CartPole in 3 lines:

```python
from neatrl import train_dqn

# Train DQN agent on CartPole
model = train_dqn(
    env_id="CartPole-v1",
    total_timesteps=10000,
    seed=42
)

print("Training completed! 🎉")
```

### Custom Training Configuration:

```python
from neatrl import train_dqn

model = train_dqn(
    env_id="CartPole-v1",
    total_timesteps=50000,
    learning_rate=2.5e-4,
    buffer_size=100000,
    gamma=0.99,
    batch_size=128,
    use_wandb=True,  # Enable experiment tracking
    exp_name="my-dqn-experiment"
)
```

### Advanced Usage:

```python
from neatrl import QNet, LinearEpsilonDecay, make_env, evaluate
import torch

# Create custom environment
env = make_env("CartPole-v1", seed=42)

# Create DQN network
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
model = QNet(state_dim, action_dim)

# Create epsilon decay schedule
epsilon_schedule = LinearEpsilonDecay(
    initial_eps=1.0,
    end_eps=0.05,
    total_timesteps=10000
)

# Evaluate trained model
returns, lengths = evaluate(model, env, num_episodes=10)
print(f"Average return: {sum(returns)/len(returns):.2f}")
```

## 📋 Requirements

- **Python**: 3.9+
- **PyTorch**: 2.0+
- **Gymnasium**: 0.29+
- **NumPy**: 1.21+
- **Stable Baselines3**: 2.0+

## 🛠️ Development

### Setup Development Environment
```bash
make install-dev  # Install all development dependencies
```

### Run Quality Checks
```bash
make all          # Run full CI pipeline (lint, format, build)
make lint         # Check code style
make format       # Auto-format code
```

### Testing
```bash
# Run DQN training test
python tests/test_dqn_cartpole.py
```

## 📚 Documentation

- [GitHub Repository](https://github.com/YuvrajSingh-mist/NeatRL)
- [Issue Tracker](https://github.com/YuvrajSingh-mist/NeatRL/issues)
- [PyPI Package](https://pypi.org/project/neatrl/)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Guidelines
1. Follow the existing code style
2. Add tests for new features
3. Update documentation as needed
4. Ensure all CI checks pass

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on top of [CleanRL](https://github.com/vwxyzjn/cleanrl) implementations
- Uses [Gymnasium](https://gymnasium.farama.org/) for environments
- Powered by [PyTorch](https://pytorch.org/)

---

**Made with ❤️ for the RL community**