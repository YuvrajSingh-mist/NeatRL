# Deep Q-Learning for Atari Breakout

This repository contains an implementation of Deep Q-Network (DQN) for solving the BreakoutNoFrameskip-v4 environment from Atari. The implementation includes features such as experience replay, target networks, epsilon-greedy exploration, and convolutional neural networks for processing visual input.

![Atari Breakout DQN Training](images/image.png)

![Breakout Demo](images/output.gif)

## 🚀 Quick Training with NeatRL Library

**For the easiest way to train a DQN agent on Atari games, use the NeatRL library** - it handles preprocessing, custom networks, and device management automatically.

### Install NeatRL

```bash
pip install neatrl
```

### Train DQN on Breakout (Simple)

```python
from neatrl import train_dqn

# Define convolutional Q-network for Atari
import torch.nn as nn

class AtariQNet(nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.conv1 = nn.Conv2d(state_space, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, action_space)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = train_dqn(
    env_id="BreakoutNoFrameskip-v4",
    total_timesteps=1000000,
    seed=42,
    atari_wrapper=True,  # Apply Atari preprocessing
    custom_agent=AtariQNet(4, 4),  # 4 stacked frames, 4 actions
    use_wandb=True,
    wandb_project="atari-experiments",
    exp_name="dqn-breakout"
)

print("Training completed! 🎉")
```

### Advanced Training with Multiple Environments

```python
from neatrl import train_dqn

model = train_dqn(
    env_id="BreakoutNoFrameskip-v4",
    total_timesteps=2000000,
    seed=42,
    learning_rate=2.5e-4,
    buffer_size=100000,
    gamma=0.99,
    batch_size=32,
    atari_wrapper=True,
    custom_agent=AtariQNet(4, 4),
    n_envs=4,  # Parallel environments for faster training
    use_wandb=True,
    wandb_project="atari-experiments",
    exp_name="dqn-breakout-parallel"
)
```
### Training on Other Atari Games

You can easily adapt the code for other Atari games by changing the `env_id` and adjusting the `action_space` in your custom network:

```python
# For Pong (6 actions)
model = train_dqn(
    env_id="ALE/Pong-v5",
    custom_agent=AtariQNet(4, 6),  # 6 actions for Pong
    # ... other parameters
)

# For Space Invaders (6 actions)
model = train_dqn(
    env_id="ALE/SpaceInvaders-v5",
    custom_agent=AtariQNet(4, 6),  # 6 actions
    # ... other parameters
)
```

## Overview

The main training script (`train.py`) implements a DQN agent to solve the BreakoutNoFrameskip-v4 environment, where the goal is to control a paddle to bounce a ball and break bricks. The agent learns to take actions (move left, right, or stay still) to maximize the score by breaking as many bricks as possible while keeping the ball in play.

## Features

- **Deep Q-Network (DQN)**: Uses a convolutional neural network to approximate the Q-function from raw pixel input
- **Atari Preprocessing**: Includes frame skipping, grayscale conversion, frame stacking, and image resizing
- **Experience Replay**: Stores transitions in a replay buffer to break correlations between consecutive samples
- **Target Network**: Uses a separate target network to stabilize learning
- **Epsilon-Greedy Exploration**: Balances exploration and exploitation with a decaying epsilon
- **Evaluation**: Periodically evaluates the model and saves videos of the agent's performance
- **Logging**: Includes logging to TensorBoard for experiment tracking
- **Video Recording**: Records videos during training and evaluation for visualization

## Requirements

```
gymnasium
torch
numpy
tqdm
stable-baselines3
imageio
opencv-python
tensorboard
huggingface_hub
ale-py
```

## Configuration

The project is configured through the `Config` class, which includes the following parameters:

- **Environment Settings**:
  - `env_id`: The Atari environment ID (default: "BreakoutNoFrameskip-v4")
  - `seed`: Random seed for reproducibility (default: 42)

- **Training Parameters**:
  - `total_timesteps`: Total number of timesteps to train (default: 1,000,000)
  - `learning_rate`: Learning rate for the optimizer (default: 2.5e-4)
  - `buffer_size`: Size of the replay buffer (default: 20,000)
  - `gamma`: Discount factor (default: 0.99)
  - `tau`: Soft update parameter for target network (default: 1.0)
  - `target_network_frequency`: Frequency of target network updates (default: 50)
  - `batch_size`: Batch size for training (default: 256)
  - `start_e`: Initial exploration rate (default: 1.0)
  - `end_e`: Final exploration rate (default: 0.05)
  - `exploration_fraction`: Fraction of total timesteps over which to decay epsilon (default: 0.3)
  - `learning_starts`: Number of timesteps before starting to learn (default: 1,000)
  - `train_frequency`: Frequency of training steps (default: 4)

- **Logging & Saving**:
  - `capture_video`: Whether to capture videos (default: True)
  - `save_model`: Whether to save model checkpoints (default: True)
  - `upload_model`: Whether to upload model to Hugging Face Hub (default: True)
  - `hf_entity`: Hugging Face username (default: "")

## Model Architecture

The Q-network (`QNet`) is a convolutional neural network designed for processing Atari frames with the following architecture:
- **Conv Layer 1**: 32 filters, 8x8 kernel, stride 4, ReLU activation
- **Conv Layer 2**: 32 filters, 4x4 kernel, stride 2, ReLU activation  
- **Conv Layer 3**: 64 filters, 3x3 kernel, stride 3, ReLU activation
- **Fully Connected 1**: 512 units with ReLU activation
- **Fully Connected 2**: 512 units with ReLU activation
- **Output layer**: Action space dimension (4 for Breakout: NOOP, FIRE, RIGHT, LEFT)

## Usage

To run the training script:

```bash
python train.py
```

## Atari Preprocessing

The Breakout environment uses several preprocessing steps to make the raw pixel input suitable for the DQN:
- **Frame skipping**: Every action is repeated for 4 frames to speed up training
- **Grayscale conversion**: RGB frames are converted to grayscale
- **Frame resizing**: Images are resized to 84x84 pixels
- **Frame stacking**: 4 consecutive frames are stacked to provide temporal information
- **Pixel normalization**: Pixel values are scaled to [0, 1] range

## Evaluation

The agent is evaluated periodically during training. Evaluation metrics include:
- Average return over multiple episodes
- Videos of the agent's performance


### Training Progress

![Atari Breakout DQN Training](images/image.png)

### Agent Performance

Here's a video showing the trained agent in action:

<details>
  <summary>Click to see video (GIF format)</summary>
  
![Breakout Agent Performance](images/output.gif)
  
</details>




## Logging

Training metrics are logged to TensorBoard, including:
- Episodic returns
- Episodic lengths
- TD loss
- Q-values
- Exploration rate (epsilon)

## Results

After successful training, the agent should be able to achieve high scores in Breakout by learning to effectively control the paddle to break bricks and keep the ball in play. The game terminates when all bricks are destroyed or all lives are lost.

## References

- [Deep Q-Network (DQN) Paper](https://www.nature.com/articles/nature14236)
- [CleanRL](https://github.com/vwxyzjn/cleanrl) - This implementation is inspired by the CleanRL project

## License

This project is licensed under the MIT License - see the LICENSE file for details.
