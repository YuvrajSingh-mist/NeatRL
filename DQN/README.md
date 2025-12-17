# Deep Q-Learning for CartPole

This repository contains an implementation of Deep Q-Network (DQN) for solving the CartPole-v1 environment from OpenAI Gym (Gymnasium). The implementation includes features such as experience replay, target networks, and epsilon-greedy exploration.

![CartPole DQN Training Visualization](images/image.png)

## 🚀 Quick Training with NeatRL Library

**For the easiest way to train a DQN agent, use the NeatRL library** - it's simpler and more user-friendly than this individual implementation.

### Install NeatRL

```bash
pip install neatrl"[classic,box2d,atari]"
```

### Train DQN on CartPole (3 lines!)

```python
from neatrl import train_dqn

model = train_dqn(
    env_id="CartPole-v1",
    total_timesteps=10000,
    seed=42
)

print("Training completed! 🎉")
```

### Advanced Training with Experiment Tracking

```python
from neatrl import train_dqn

model = train_dqn(
    env_id="CartPole-v1",
    total_timesteps=50000,
    seed=42,
    capture_video=True,        # Record training videos
    use_wandb=True,           # Enable W&B logging
    wandb_project="my-rl-experiments",
    exp_name="cartpole-dqn"
)
```

### Using Custom Neural Networks

You can also provide your own neural network architecture:

```python
import torch.nn as nn
from neatrl import train_dqn

class MyCustomQNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        return self.layers(x)

model = train_dqn(
    env_id="CartPole-v1",
    total_timesteps=50000,
    agent=MyCustomQNet(state_dim=4, action_dim=2),  # Pass instantiated nn.Module
    use_wandb=True,
    exp_name="custom-dqn-cartpole"
)
```

**📖 [Complete NeatRL Documentation](../neatrl/docs/DQN/README.md)** - Includes detailed usage examples, hyperparameter tuning guides, and troubleshooting tips.

---

## 🏃‍♂️ Running Directly (Without NeatRL)

If you prefer to run this implementation directly instead of using the NeatRL library, follow these steps:

### Installation

```bash
# Clone the repository
git clone https://github.com/YuvrajSingh-mist/NeatRL.git
cd NeatRL/DQN

# Install dependencies
pip install -e .
```


### Quick Start

```bash
# Run the training script directly
python train.py
```
---

## Overview

The main training script (`train.py`) implements a DQN agent to solve the CartPole-v1 environment, where the goal is to balance a pole on a moving cart. The agent learns to take actions (move left or right) to keep the pole upright for as long as possible.

## Features

- **Deep Q-Network (DQN)**: Uses a neural network to approximate the Q-function
- **Experience Replay**: Stores transitions in a replay buffer to break correlations between consecutive samples
- **Target Network**: Uses a separate target network to stabilize learning
- **Epsilon-Greedy Exploration**: Balances exploration and exploitation with a decaying epsilon
- **Evaluation**: Periodically evaluates the model and saves videos of the agent's performance
- **Logging**: Includes logging to TensorBoard and Weights & Biases (wandb) for experiment tracking
- **Video Recording**: Records videos during training and evaluation for visualization


## Configuration

The project is configured through the `Config` class in `train.py`, which includes the following parameters:

- **Experiment Settings**:
  - `exp_name`: Experiment name for logging (default: "DQN")
  - `seed`: Random seed for reproducibility (default: 42)
  - `env_id`: The Gym environment ID (default: "CartPole-v1")

- **Training Parameters**:
  - `total_timesteps`: Total number of timesteps to train (default: 20,000)
  - `learning_rate`: Learning rate for the optimizer (default: 2.5e-4)
  - `buffer_size`: Size of the replay buffer (default: 10,000)
  - `gamma`: Discount factor (default: 0.99)
  - `tau`: Soft update parameter for target network (default: 1.0)
  - `target_network_frequency`: Frequency of target network updates (default: 50)
  - `batch_size`: Batch size for training (default: 128)
  - `start_e`: Initial exploration rate (default: 1.0)
  - `end_e`: Final exploration rate (default: 0.05)
  - `exploration_fraction`: Fraction of total timesteps over which to decay epsilon (default: 0.5)
  - `learning_starts`: Number of timesteps before starting to learn (default: 1,000)
  - `train_frequency`: Frequency of training steps (default: 10)

- **Logging & Saving**:
  - `capture_video`: Whether to capture videos (default: False)
  - `use_wandb`: Whether to use Weights & Biases for logging (default: False)
  - `wandb_project`: WandB project name (default: "cleanRL")
  - `wandb_entity`: WandB username/team (default: "")
  - `eval_every`: Frequency of evaluation during training (default: 1,000)
  - `save_every`: Frequency of saving the model (default: 1,000)
  - `upload_every`: Frequency of uploading videos to wandb (default: 100)



## Model Architecture

The Q-network (`QNet`) is a simple fully-connected neural network with the following architecture:
- Input layer: State space dimension
- Hidden layer 1: 256 units with ReLU activation
- Hidden layer 2: 512 units with ReLU activation
- Output layer: Action space dimension (2 for CartPole-v1)

## Evaluation

The agent is evaluated periodically during training (every 1,000 timesteps by default). Evaluation metrics include:
- Average return over multiple episodes
- Videos of the agent's performance


### Training Progress

![CartPole DQN Training Visualization](images/image.png)

### Agent Performance

Here's a video showing the trained agent in action:

<details>
  <summary>Click to see video (GIF format)</summary>
  
  <!-- <!-- ![CartPole Agent Performance](images/final.mp4) -->
  
![CartPole Demo](images/cartpole_demo.gif) -->
  
</details>




## Logging

Training metrics are logged to both TensorBoard and Weights & Biases (if enabled), including:
- Episodic returns
- Episodic lengths
- TD loss
- Q-values
- Steps per second (SPS)
- Exploration rate (epsilon)

## Results

After successful training, the agent should be able to balance the pole for the maximum episode length (500 timesteps in CartPole-v1).

## References

- [Deep Q-Network (DQN) Paper](https://www.nature.com/articles/nature14236)
- [CleanRL](https://github.com/vwxyzjn/cleanrl) - This implementation is inspired by the CleanRL project

## License

This project is licensed under the MIT License - see the LICENSE file for details.
