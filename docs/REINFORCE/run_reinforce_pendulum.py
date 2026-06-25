"""
script for REINFORCE training on MountainCar using neatrl library.
"""

import torch
import torch.nn as nn

from neatrl import train_reinforce


class PolicyNet(nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        print(f"State space: {state_space}, Action space: {action_space}")
        self.fc1 = nn.Linear(state_space, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.mean = nn.Linear(64, action_space)
        self.logstd = nn.Parameter(torch.zeros(action_space), requires_grad=True)
        self.act_rng = 2.0  # Action range for Pendulum-v0

    def forward(self, x):
        x = self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))
        mean = self.act_rng * torch.tanh(self.mean(x))
        std = torch.exp(self.logstd)  # Ensure std is positive

        return mean, std

    def get_action(self, x):
        mean, std = self.forward(x)
        dist = torch.distributions.Normal(
            mean, std
        )  # Create a normal distribution from the mean and std

        action = dist.sample()  # Sample an action from the distribution
        return action, dist.log_prob(action), dist


def test_reinforce_pendulum():
    """Test REINFORCE training on Pendulum-v0."""
    print("Testing REINFORCE training on Pendulum-v0 with neatrl...")

    # Train REINFORCE on Pendulum
    model = train_reinforce(
        env_id="Pendulum-v1",
        total_steps=100000,
        seed=42,
        learning_rate=3e-4,
        gamma=0.99,
        capture_video=True,
        use_wandb=True,
        wandb_project="cleanRL",
        wandb_entity="",
        exp_name="REINFORCE-Pendulum",
        eval_every=1000,
        save_every=20000,
        atari_wrapper=False,
        n_envs=4,
        num_eval_eps=1,
        device="cpu",
        grid_env=False,
        custom_agent=PolicyNet(3, 1),
        anneal_lr=True,
        use_entropy=True,
        entropy_coeff=0.01,
        normalize_obs=True,
        normalize_reward=False,
    )

    print("REINFORCE training on Pendulum-v0 completed successfully!")
    return model


if __name__ == "__main__":
    test_reinforce_pendulum()
