# DDPG (Deep Deterministic Policy Gradient)

DDPG is an actor-critic algorithm for continuous action spaces that uses deep neural networks. This implementation includes both standard DDPG for vector observations and CNN-based DDPG for image observations (like Atari games).

**Note**: DDPG is designed for continuous action spaces. For discrete action environments, consider using DQN instead.

## Features

- **Standard DDPG**: For environments with vector observations (e.g., HalfCheetah, BipedalWalker)
- **CNN DDPG**: For environments with image observations (e.g., Atari games)
- **Experience Replay**: Stabilizes training with off-policy learning
- **Target Networks**: Prevents moving target problem with soft updates
- **Exploration**: Ornstein-Uhlenbeck noise for action exploration
- **W&B Integration**: Built-in logging and video recording

## Usage

### Standard DDPG (Vector Observations)

```python
from neatrl.ddpg import train_ddpg

train_ddpg(
    env_id="HalfCheetah-v5",
    total_timesteps=1000000,
    use_wandb=True
)
```

### CNN DDPG (Image Observations)

```python
from neatrl.ddpg import train_ddpg_cnn

train_ddpg_cnn(
    env_id="PongNoFrameskip-v4",
    total_timesteps=100000,
    use_wandb=True
)
```

## Network Architectures

### Actor Network (Standard)
- Input: State vector
- Hidden: 256 → 256 units with Mish activation
- Output: Continuous actions (tanh activation)

### Q-Network (Standard)
- State path: 256 → 256 units
- Action path: Linear projection
- Combined: 512 → 512 → 256 → 1 units

### Actor Network (CNN)
- Conv layers: 32@8x8 → 64@4x4 → 64@3x3
- FC layers: 3136 → 512 → action_dim
- Output: Continuous actions (tanh activation)

### Q-Network (CNN)
- State conv: Same as Actor CNN
- State FC: 3136 → 512
- Action FC: action_dim → 512
- Combined: 1024 → 512 → 1

## Key Parameters

- `total_timesteps`: Total training steps
- `learning_rate`: Adam learning rate (3e-4)
- `buffer_size`: Replay buffer size (100k)
- `gamma`: Discount factor (0.99)
- `tau`: Target network soft update (0.005)
- `batch_size`: Training batch size (256)
- `exploration_fraction`: Exploration noise scale (0.1)
- `learning_starts`: Steps before training begins (25k)
- `target_network_frequency`: Target update frequency (1-50)

## Atari Environments

For Atari games, the implementation automatically applies:
- Grayscale conversion
- Frame scaling to 84x84
- Frame stacking (4 frames)
- Episode statistics recording

## Requirements

- PyTorch
- Gymnasium
- Stable Baselines 3 (for ReplayBuffer)
- Weights & Biases (optional)
- ale_py (for Atari environments)

## Examples

- `run_ddpg_half_cheetah.py`: Training on MuJoCo HalfCheetah environment
- `run_ddpg_cnn_car_racing.py`: Training on CarRacing environment
- `run_ddpg_bipedal_walker.py`: Training on BipedalWalker environment
- `run_ddpg_cliff_walking.py`: Training on CliffWalking environment (experimental)

## PyPI

For installation and more information, visit [NeatRL on PyPI](https://pypi.org/project/neatrl/)