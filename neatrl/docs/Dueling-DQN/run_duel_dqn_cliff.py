#!/usr/bin/env python3
"""
script for Dueling DQN training on CliffWalking using neatrl library.
"""

import torch
import torch.nn as nn

from neatrl import train_dueling_dqn


class DuelingQNet(nn.Module):
    def __init__(self, state_space, action_space):
        super(DuelingQNet, self).__init__()
        print(f"State space: {state_space}, Action space: {action_space}")

        self.features = nn.Sequential(
            nn.Linear(state_space, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.values = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.adv = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_space)
        )

    def forward(self, x):
        feat = self.features(x)
        values = self.values(feat)
        adv = self.adv(feat)
        # Dueling architecture: Q = V + A - mean(A)
        q_values = values + adv - adv.mean(dim=1, keepdim=True)
        return q_values, values, adv, feat


def test_dueling_dqn_cliffwalking():
    """Test Dueling DQN training on CliffWalking-v0."""
    print("Testing Dueling DQN training on CliffWalking-v0 with neatrl...")

    # Train Dueling DQN on CliffWalking
    model = train_dueling_dqn(
        env_id="CliffWalking-v0",
        total_timesteps=100000,
        seed=42,
        learning_rate=2e-4,
        buffer_size=10000,  # Reduced from 30K to save memory
        gamma=0.99,
        tau=1.0,
        target_network_frequency=50,
        batch_size=4,
        start_e=1.0,
        end_e=0.05,
        exploration_fraction=0.4,
        learning_starts=1000,
        train_frequency=4,
        capture_video=False,  # Disabled to save memory during training
        use_wandb=True,
        wandb_project="cleanRL",
        wandb_entity="",
        exp_name="Dueling-DQN-CliffWalking-Test",
        custom_agent=DuelingQNet(48, 4),  # CliffWalking state and action dimensions
        atari_wrapper=False,
        n_envs=4,
        eval_every=5000,
        max_grad_norm=4.0,
        grid_env=True
    )

    print(f"Training completed! Model type: {type(model)}")
    print(f"Model device: {next(model.parameters()).device}")

    # Test model inference
    print("Testing model inference...")
    test_obs = torch.randn(1, 48)  # CliffWalking observation shape (one-hot encoded)
    with torch.no_grad():
        q_values, values, adv, feat = model(test_obs)
        action = q_values.argmax().item()

    print(f"Q-values shape: {q_values.shape}")
    print(f"Selected action: {action}")
    print("Test completed successfully!")

    return model


if __name__ == "__main__":
    test_dueling_dqn_cliffwalking()
