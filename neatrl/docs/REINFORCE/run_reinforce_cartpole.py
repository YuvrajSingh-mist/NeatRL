"""
script for REINFORCE training on CartPole using neatrl library.
"""

import torch
import torch.nn as nn

from neatrl import train_reinforce


class CustomPolicyNet(nn.Module):
    """Custom Policy network for testing."""

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.out = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.nn.functional.softmax(self.out(x), dim=-1)

    def get_action(self, x):
        action_probs = self.forward(x)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        return action, dist.log_prob(action)


def test_reinforce_cartpole():
    """Test REINFORCE training on CartPole-v1."""
    print("Testing REINFORCE training on CartPole-v1 with neatrl...")

    # Train REINFORCE on CartPole
    model = train_reinforce(
        env_id="CartPole-v1",
        total_steps=2000,
        seed=42,
        learning_rate=2e-3,
        gamma=0.99,
        capture_video=True,
        use_wandb=True,
        wandb_project="cleanRL",
        wandb_entity="",
        exp_name="REINFORCE-CartPole",
        eval_every=100,
        save_every=1000,
        atari_wrapper=False,
        n_envs=1,
        num_eval_eps=10,
        device="cpu",
        grid_env=False,
        custom_agent=CustomPolicyNet(4, 2),
    )

    print("REINFORCE training on CartPole-v1 completed successfully!")
    return model


if __name__ == "__main__":
    test_reinforce_cartpole()
