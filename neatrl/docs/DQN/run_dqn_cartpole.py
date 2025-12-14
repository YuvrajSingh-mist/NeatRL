#!/usr/bin/env python3
"""
script for DQN training on CartPole using neatrl library.
"""

import torch
import torch.nn as nn

from neatrl import train_dqn


class CustomQNet(nn.Module):
    """Custom Q-network for testing."""

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.q_value = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.q_value(x)


def test_dqn_cartpole():
    """Test DQN training on CartPole-v1."""
    print("Testing DQN training on CartPole-v1 with neatrl...")

    # Train DQN on CartPole
    model = train_dqn(
        env_id="CartPole-v1",
        total_timesteps=1000000,
        seed=42,
        learning_rate=2.5e-4,
        buffer_size=50000,
        gamma=0.99,
        tau=1.0,
        target_network_frequency=50,
        batch_size=128,
        start_e=1.0,
        end_e=0.05,
        exploration_fraction=0.5,
        learning_starts=1000,
        train_frequency=4,
        capture_video=True,
        use_wandb=True,
        wandb_project="cleanRL",
        wandb_entity="",
        exp_name="DQN-CartPole-Test",
        custom_agent=CustomQNet(4, 2),  # CartPole state and action dimensions
        atari_wrapper=False,
        n_envs=4,
        capture_video=True,
        eval_every=5000,
    )

    print(f"Training completed! Model type: {type(model)}")
    print(f"Model device: {next(model.parameters()).device}")

    # Test model inference
    print("Testing model inference...")
    test_obs = torch.randn(1, 4)  # CartPole observation shape
    with torch.no_grad():
        q_values = model(test_obs)
        action = q_values.argmax().item()

    print(f"Q-values shape: {q_values.shape}")
    print(f"Selected action: {action}")
    print("Test completed successfully!")

    return model


if __name__ == "__main__":
    test_dqn_cartpole()
