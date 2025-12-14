#!/usr/bin/env python3
"""
script for DQN training on CarRacing using neatrl library.
"""

import torch
import torch.nn as nn

from neatrl import train_dqn


class CarRacingQNet(nn.Module):
    """Convolutional Q-network for CarRacing."""

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(state_dim, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, action_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


def test_dqn_carracing():
    """Test DQN training on CarRacing-v2 with discrete actions."""
    print("Testing DQN training on CarRacing-v2 with neatrl...")

    # Note: CarRacing has continuous actions by default.
    # For discrete, you may need to wrap the environment.
    # Here we assume a discrete version or adjust accordingly.

    # Train DQN on CarRacing
    model = train_dqn(
        env_id="CarRacing-v2",
        total_timesteps=500000,
        seed=42,
        learning_rate=2.5e-4,
        buffer_size=100000,
        gamma=0.99,
        tau=1.0,
        target_network_frequency=50,
        batch_size=32,
        start_e=1.0,
        end_e=0.05,
        exploration_fraction=0.5,
        learning_starts=1000,
        train_frequency=4,
        capture_video=True,
        use_wandb=True,
        wandb_project="cleanRL",
        wandb_entity="",
        exp_name="DQN-CarRacing-Test",
        custom_agent=CarRacingQNet(3, 5),  # Assuming 3 channels, 5 discrete actions
        atari_wrapper=False,  # CarRacing needs custom preprocessing
        n_envs=1,
        eval_every=5000,
    )

    print(f"Training completed! Model type: {type(model)}")
    print(f"Model device: {next(model.parameters()).device}")

    # Test model inference
    print("Testing model inference...")
    test_obs = torch.randn(1, 3, 96, 96)  # CarRacing observation shape
    with torch.no_grad():
        q_values = model(test_obs)
        action = q_values.argmax().item()

    print(f"Q-values shape: {q_values.shape}")
    print(f"Selected action: {action}")
    print("Test completed successfully!")

    return model


if __name__ == "__main__":
    test_dqn_carracing()
