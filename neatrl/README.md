# 🎯 NeatRL

**A clean, modern Python library for reinforcement learning algorithms**

NeatRL provides high-quality implementations of popular RL algorithms with a focus on simplicity, performance, and ease of use. Built with PyTorch and designed for both research and production use.

## ✨ Features

- 🚀 **Fast & Efficient**: Optimized implementations using PyTorch
- 🎯 **Production Ready**: Clean APIs and comprehensive error handling
- 📊 **Experiment Tracking**: Built-in support for Weights & Biases logging
- 🎮 **Gymnasium Compatible**: Works with all Gymnasium environments
- 🔧 **Easy to Extend**: Modular design for adding new algorithms
- 📈 **State-of-the-Art**: Implements modern RL techniques and best practices
- 🏗️ **Grid Environment Support**: Automatic one-hot encoding for discrete grid environments

## 🏗️ Supported Algorithms

### Current Implementations
- **DQN** (Deep Q-Network) - Classic value-based RL algorithm
- *More algorithms coming soon...*

## 📦 Installation

```bash
python -m venv neatrl-env
source neatrl-env/bin/activate  # On Windows use `neatrl-env\Scripts

pip install neatrl
```

## 🚀 Quick Start

Train a DQN agent on CartPole in 3 lines:

```python
from neatrl import train_dqn

model = train_dqn(
    env_id="CartPole-v1",
    total_timesteps=10000,
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