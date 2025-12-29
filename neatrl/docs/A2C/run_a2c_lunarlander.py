"""
A2C Example for Acrobot Environment

This example demonstrates how to use A2C for the Acrobot environment.
Acrobot has discrete action spaces and vector observations, making it suitable for standard A2C.
"""

from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from neatrl.a2c import train_a2c


# --- Networks ---
def layer_init(
    layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0
) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorNet(nn.Module):
    def __init__(
        self,
        state_space: Union[int, tuple[int, ...]],
        action_space: int,
    ) -> None:
        super().__init__()

        self.fc1 = layer_init(nn.Linear(state_space, 128))
        self.fc2 = layer_init(nn.Linear(128, 128))
        self.fc3 = layer_init(nn.Linear(128, 64))
        self.fc4 = layer_init(nn.Linear(64, action_space))

    def forward(self, x: torch.Tensor) -> torch.distributions.Distribution:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        probs = torch.nn.functional.softmax(self.fc4(x), dim=-1)
        dist = torch.distributions.Categorical(probs)
        return dist

    def get_action(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.distributions.Distribution]:
        dist = self.forward(x)

        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob, dist


class CriticNet(nn.Module):
    def __init__(self, state_space: Union[int, tuple[int, ...]]) -> None:
        super().__init__()
        self.fc1 = layer_init(nn.Linear(state_space, 128))
        self.fc2 = layer_init(nn.Linear(128, 128))
        self.fc3 = layer_init(nn.Linear(128, 64))
        self.value = layer_init(nn.Linear(64, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.value(x)

env = gym.make("LunarLander-v3", render_mode="rgb_array")

def train_run_lunarlander():
    """Train A2C on LunarLander environment."""

    train_a2c(
        env=env,# LunarLander environment
        total_timesteps=1000000,  # Total timesteps for training
        seed=42,
        lr=3e-4,  # Learning rate
        gamma=0.99,  # Discount factor
        use_wandb=True,  # Set to True to enable logging
        capture_video=True,
        eval_every=1000,  # Evaluate every 1k steps
        save_every=50000,  # Save model every 50k steps
        num_eval_episodes=4,
        normalize_obs=False,  # Pendulum doesn't need normalization
        device="cpu",  # Use "cuda" if you have GPU
        actor_class=ActorNet,
        critic_class=CriticNet,
        log_gradients=True,
        n_envs=4,  # Use multiple environments for A2C
    )


if __name__ == "__main__":
    train_run_lunarlander()
