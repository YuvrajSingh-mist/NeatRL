# DQN-Taxi: Deep Q-Network for OpenAI Gym Taxi-v3

This project implements a Deep Q-Network (DQN) agent to solve the classic Taxi-v3 environment from OpenAI Gym. The agent learns to efficiently pick up and drop off passengers in a grid world using reinforcement learning.

[![Taxi-v3 Demo](images/output.gif)]

## Environment
- **Taxi-v3** is a discrete environment with:
  - **State space:** 16 (or 500 for the full version)
  - **Action space:** 6 (South, North, East, West, Pickup, Dropoff)
- The agent receives positive rewards for successful drop-offs and negative rewards for illegal moves or time steps.

## Features
- DQN with experience replay and target network
- Epsilon-greedy exploration
- One-hot encoding for discrete state representation
- Logging of Q-values, advantage, and value estimates
- Integration with TensorBoard and Weights & Biases (WandB) for experiment tracking

## Using NeatRL

For a more streamlined and production-ready DQN implementation, you can use [NeatRL](https://github.com/YuvrajSingh-mist/NeatRL), a clean Python library for reinforcement learning algorithms.

### Installation

```bash
pip install neatrl"[classic,box2d,atari]"
```

### Training on Taxi

```python
from neatrl import train_dqn

# Train DQN on Taxi with automatic one-hot encoding
model = train_dqn(
    env_id="Taxi-v3",
    total_timesteps=50000,
    seed=42,
    grid_env=True,  # Enable one-hot encoding for discrete states
    use_wandb=True,
    wandb_project="taxi-experiments",
    exp_name="dqn-taxi"
)

print("Training completed! 🚕")
```

For a complete example script, see [run_dqn_taxi.py](https://github.com/YuvrajSingh-mist/NeatRL/blob/master/neatrl/docs/DQN/run_dqn_taxi.py).

The `grid_env=True` parameter automatically applies one-hot encoding to the discrete state observations, making them suitable for neural network input without manual preprocessing.

---

## 🏃‍♂️ Running Directly (Without NeatRL)

If you prefer to run this implementation directly instead of using the NeatRL library, follow these steps:

### Installation

```bash
# Clone the repository
git clone https://github.com/YuvrajSingh-mist/NeatRL.git
cd NeatRL/DQN-Taxi

# Install dependencies
pip install -e .
```

Or install the required packages manually:

```bash
pip install torch torchvision numpy gymnasium stable-baselines3 tqdm imageio wandb
```

### Quick Start

```bash
# Run the training script directly
python train.py
```
---

## Logging & Visualization
- Training logs and metrics are saved for visualization in TensorBoard and/or WandB.
- Q-values, advantage, and value estimates are logged for analysis.

## Customization
- Change hyperparameters and logging options in the `Config` class in `train.py`.
- You can switch between different exploration strategies or network architectures as needed.

## Results
The DQN agent should learn to solve the Taxi-v3 environment, achieving high average rewards after sufficient training.

## References
- [OpenAI Gym Taxi-v3](https://www.gymlibrary.dev/environments/toy_text/taxi/)
- [DQN Paper (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/)

## License
MIT License
