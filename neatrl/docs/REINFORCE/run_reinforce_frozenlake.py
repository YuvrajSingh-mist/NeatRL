#!/usr/bin/env python3
"""
script for REINFORCE training on FrozenLake using neatrl library.
"""

from neatrl import train_reinforce
import torch
import torch.nn as nn

#For discrete actions
class PolicyNet(nn.Module):
    def __init__(self, state_space, action_space):
        super(PolicyNet, self).__init__()
        print(f"State space: {state_space}, Action space: {action_space}")
        self.fc1 = nn.Linear(state_space, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 16)
        self.out = nn.Linear(16, action_space)

    def forward(self, x):
        x = self.out(self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x))))))
        x = torch.nn.functional.softmax(x, dim=-1)  # Apply softmax to get probabilities
        return x

    def get_action(self, x):
        
        action_probs = self.forward(x)
        dist = torch.distributions.Categorical(action_probs)  # Create a categorical distribution from the probabilities
        action = dist.sample()  # Sample an action from the distribution
        return action, dist.log_prob(action)


def test_reinforce_frozenlake():
    """Test REINFORCE training on FrozenLake-v1."""
    print("Testing REINFORCE training on FrozenLake-v1 with neatrl...")

    # Train REINFORCE on FrozenLake with one-hot encoding
    model = train_reinforce(
        env_id="FrozenLake-v1",
        total_steps=100000,
        seed=42,
        learning_rate=2e-4,
        gamma=0.99,
        capture_video=False,
        use_wandb=False,
        wandb_project="cleanRL",
        wandb_entity="",
        exp_name="REINFORCE-FrozenLake",
        eval_every=100,
        save_every=1000,
        atari_wrapper=False,
        n_envs=4,
        num_eval_eps=10,
        device="cpu",
        grid_env=True,  # Enable one-hot encoding for discrete states
        custom_agent=PolicyNet(16, 4)
    )

    print("REINFORCE training on FrozenLake-v1 completed successfully!")
    return model


if __name__ == "__main__":
    test_reinforce_frozenlake()