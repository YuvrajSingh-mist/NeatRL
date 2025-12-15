#!/usr/bin/env python3
"""
script for DQN training on Taxi using neatrl library.
"""

import torch
import torch.nn as nn

from neatrl import train_dqn


class QNet(nn.Module):
    def __init__(self, state_space, action_space):
        super(QNet, self).__init__()
        print(f"State space: {state_space}, Action space: {action_space}")
        self.fc1 = nn.Linear(state_space, 256)
        self.fc2 = nn.Linear(256, 512)
        self.q_value = nn.Linear(512, action_space)
    def forward(self, x):
        return self.q_value(torch.relu(self.fc2(torch.relu(self.fc1(x)))))
    
    

def test_dqn_taxi():
    """Test DQN training on Taxi-v3."""
    print("Testing DQN training on Taxi-v3 with neatrl...")

    # Train DQN on Taxi
    model = train_dqn(
        env_id="Taxi-v3",
        total_timesteps=6000000,
        seed=42,
        learning_rate=2.5e-4,
        buffer_size=20000,
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
        exp_name="DQN-Taxi-Test",
        custom_agent=QNet(500, 6),  # Taxi state and action dimensions (one-hot encoded)
        atari_wrapper=False,
        n_envs=1,
        eval_every=50000,
        grid_env=True,  # Enable one-hot encoding for discrete states
    )

    print(f"Training completed! Model type: {type(model)}")
    print(f"Model device: {next(model.parameters()).device}")

    # Test model inference
    print("Testing model inference...")
    test_obs = torch.randn(1, 500)  # Taxi observation shape (one-hot)
    with torch.no_grad():
        q_values = model(test_obs)
        action = q_values.argmax().item()

    print(f"Q-values shape: {q_values.shape}")
    print(f"Selected action: {action}")
    print("Test completed successfully!")

    return model


if __name__ == "__main__":
    test_dqn_taxi()