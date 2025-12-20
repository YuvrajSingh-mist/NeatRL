"""
script for RND-QNet training on Breakout using neatrl library.
"""

import torch
import torch.nn as nn

from neatrl.rnd import train_rnd_qnet


class CustomQNet(nn.Module):
    """Custom Q-network for testing."""

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)


def test_rnd_qnet_breakout():
    """Test RND-QNet training on BreakoutNoFrameskip-v4."""
    print("Testing RND-QNet training on BreakoutNoFrameskip-v4 with neatrl...")

    # Train RND-QNet on Breakout
    model = train_rnd_qnet(
        env_id="BreakoutNoFrameskip-v4",
        total_timesteps=10000,
        seed=42,
        learning_rate=2.5e-4,
        buffer_size=10000,
        ext_gamma=0.99,
        tau=1.0,
        target_network_frequency=500,
        batch_size=32,
        start_e=1.0,
        end_e=0.05,
        exploration_fraction=0.1,
        learning_starts=1000,
        train_frequency=4,
        max_grad_norm=0.5,
        capture_video=True,
        use_wandb=True,
        wandb_project="cleanRL",
        wandb_entity="",
        exp_name="RND-QNet-Breakout",
        eval_every=1000,
        save_every=5000,
        atari_wrapper=True,
        custom_agent=None,  # Use default QNet
        num_eval_eps=5,
        n_envs=4,  # Use 4 parallel environments
        grid_env=False,
        int_coeff=1.0,
        int_gamma=0.99,
        rnd_lr=1e-4,
    )

    print("Training completed!")
    return model


if __name__ == "__main__":
    test_rnd_qnet_breakout()
