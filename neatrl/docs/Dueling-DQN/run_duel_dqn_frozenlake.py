#!/usr/bin/env python3
"""
script for DQN training on CartPole using neatrl library.
"""

import torch
import torch.nn as nn

from neatrl import train_dqn


class QNet(nn.Module):
    def __init__(self, state_space, action_space):
        super(QNet, self).__init__()
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
        res = values + adv - adv.mean(dim=-1, keepdim=True) #adding stuf --> big big grads thus normalize babyyy!
        return res


def test_dqn_frozenlake():
    """Test DQN training on FrozenLake-v1."""
    print("Testing DQN training on FrozenLake-v1 with neatrl...")

    # Train DQN on FrozenLake
    model = train_dqn(
        env_id="FrozenLake-v1",
        total_timesteps=100000,
        seed=42,
        learning_rate=2.5e-4,
        buffer_size=10000,  # Reduced from 30K to save memory
        gamma=0.99,
        tau=1.0,
        target_network_frequency=50,
        batch_size=4,
        start_e=1.0,
        end_e=0.05,
        exploration_fraction=0.5,
        learning_starts=1000,
        train_frequency=4,
        capture_video=False,  # Disabled to save memory during training
        use_wandb=True,
        wandb_project="cleanRL",
        wandb_entity="",
        exp_name="DQN-FrozenLake-Test",
        custom_agent=QNet(16, 4),  # FrozenLake state and action dimensions
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
    test_obs = torch.randn(1, 4)  # CartPole observation shape
    with torch.no_grad():
        q_values = model(test_obs)
        action = q_values.argmax().item()

    print(f"Q-values shape: {q_values.shape}")
    print(f"Selected action: {action}")
    print("Test completed successfully!")

    return model


if __name__ == "__main__":
    test_dqn_frozenlake()
