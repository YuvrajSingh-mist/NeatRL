#!/usr/bin/env python3
"""
script for DQN training on FrozenLake using neatrl library.
"""

import torch
import torch.nn as nn
import gymnasium as gym
from neatrl import train_dqn


class FrozenLakeQNet(nn.Module):
    """Q-network for FrozenLake with one-hot encoded states."""

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.q_value = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.q_value(x)


def test_dqn_frozenlake():
    """Test DQN training on FrozenLake-v1."""
    print("Testing DQN training on FrozenLake-v1 with neatrl...")


    # Train DQN on FrozenLake
    model = train_dqn(
        env_id="FrozenLake-v1",
        total_timesteps=50000,
        seed=42,
        learning_rate=1e-3,
        buffer_size=10000,
        gamma=0.99,
        tau=1.0,
        target_network_frequency=100,
        batch_size=64,
        start_e=1.0,
        end_e=0.01,
        exploration_fraction=0.8,
        learning_starts=1000,
        train_frequency=4,
        capture_video=False,  # FrozenLake is text-based
        use_wandb=True,
        wandb_project="cleanRL",
        wandb_entity="",
        exp_name="DQN-FrozenLake-Test",
        custom_agent=FrozenLakeQNet(16, 4),  # 16 one-hot states, 4 actions
        atari_wrapper=False,
        n_envs=1,
        record=False,
        eval_every=5000,
        grid_env=True,
    )

    print(f"Training completed! Model type: {type(model)}")
    print(f"Model device: {next(model.parameters()).device}")



if __name__ == "__main__":
    test_dqn_frozenlake()