"""
TD3 Example for CarRacing with CNN

This example demonstrates TD3 with CNNs on pixel-based environments.
Uses the CarRacing-v2 environment with frame stacking.
"""

from typing import Union

import gymnasium as gym
import torch
import torch.nn as nn

from neatrl.td3 import train_td3_cnn


class ActorNetCNN(nn.Module):
    def __init__(
        self,
        state_space: Union[int, tuple[int, ...]],
        action_space: int,
    ) -> None:
        super().__init__()
        print(f"State space: {state_space}, Action dim: {action_space}")
        
        # CNN feature extractor
        self.conv1 = nn.Conv2d(state_space[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate flattened size
        conv_out_size = self._get_conv_out(state_space)
        
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.out = nn.Linear(512, action_space)

    def _get_conv_out(self, shape):
        o = torch.zeros(1, *shape)
        o = self.conv1(o)
        o = self.conv2(o)
        o = self.conv3(o)
        return int(torch.prod(torch.tensor(o.size())))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.out(x))
        return x


class QNetCNN(nn.Module):
    def __init__(self, state_space: Union[int, tuple[int, ...]], action_space: int):
        super().__init__()
        
        # CNN for state
        self.conv1 = nn.Conv2d(state_space[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate flattened size
        conv_out_size = self._get_conv_out(state_space)
        
        # Concatenate with action
        self.fc1 = nn.Linear(conv_out_size + action_space, 512)
        self.out = nn.Linear(512, 1)

    def _get_conv_out(self, shape):
        o = torch.zeros(1, *shape)
        o = self.conv1(o)
        o = self.conv2(o)
        o = self.conv3(o)
        return int(torch.prod(torch.tensor(o.size())))

    def forward(self, state, act):
        # Process state through CNN
        x = torch.relu(self.conv1(state))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        
        # Concatenate with action
        x = torch.cat((x, act), dim=1)
        x = torch.relu(self.fc1(x))
        x = self.out(x)
        return x


if __name__ == "__main__":
    # Train TD3 with CNN on CarRacing
    train_td3_cnn(
        env_id="CarRacing-v2",
        total_timesteps=1000000,
        learning_rate=1e-4,
        buffer_size=100000,
        gamma=0.99,
        tau=0.005,
        batch_size=128,
        learning_starts=5000,
        train_frequency=2,  # Delayed policy updates
        target_network_frequency=1,
        policy_noise=0.2,  # Target policy smoothing
        exploration_noise=0.1,
        noise_clip=0.5,
        low=-1.0,
        high=1.0,
        use_wandb=True,
        wandb_project="TD3-Vision",
        exp_name="td3_cnn_carracing",
        eval_every=10000,
        save_every=50000,
        num_eval_episodes=5,
        capture_video=True,
        actor_class=ActorNetCNN,
        q_network_class=QNetCNN,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
