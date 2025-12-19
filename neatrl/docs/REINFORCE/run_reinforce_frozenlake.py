"""
script for REINFORCE training on FrozenLake using neatrl library.
"""

import torch
import torch.nn as nn

from neatrl import train_reinforce


# For discrete actions
class PolicyNet(nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        print(f"State space: {state_space}, Action space: {action_space}")
        self.fc1 = nn.Linear(state_space, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 16)
        self.out = nn.Linear(16, action_space)

    def forward(self, x):
        x = self.out(self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x))))))
        x = torch.nn.functional.softmax(x, dim=-1)  # Apply softmax to get probabilities
        return x

    def get_action(self, x, eval=False):
        action_probs = self.forward(x)
        dist = torch.distributions.Categorical(
            action_probs
        )  # Create a categorical distribution from the probabilities
        if eval:
            action = torch.argmax(action_probs, dim=-1)
            return action
        
        action = dist.sample()  # Sample an action from the distribution
        return action, dist.log_prob(action), dist


def test_reinforce_frozenlake():
    """Test REINFORCE training on FrozenLake-v1."""
    print("Testing REINFORCE training on FrozenLake-v1 with neatrl...")

    # Train REINFORCE on FrozenLake with one-hot encoding
    model = train_reinforce(
        env_id="FrozenLake-v1",
        total_steps=1000000,
        seed=42,
        learning_rate=2e-3,
        gamma=0.99,
        capture_video=True,
        use_wandb=True,
        wandb_project="cleanRL",
        wandb_entity="",
        exp_name="REINFORCE-FrozenLake",
        eval_every=10000,
        save_every=10000,
        atari_wrapper=False,
        n_envs=1,
        num_eval_eps=10,
        device="cpu",
        grid_env=True,  # Enable one-hot encoding for discrete states
        custom_agent=PolicyNet(16, 4),
        use_entropy=True,
        entropy_coeff=0.01
    )

    print("REINFORCE training on FrozenLake-v1 completed successfully!")
    return model


if __name__ == "__main__":
    test_reinforce_frozenlake()
