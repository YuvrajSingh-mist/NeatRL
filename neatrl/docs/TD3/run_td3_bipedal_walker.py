"""
TD3 Example for BipedalWalker Environment

This example demonstrates TD3 on the BipedalWalker-v3 environment.
BipedalWalker is a challenging Box2D environment with continuous control.
"""

from typing import Union

import torch
import torch.nn as nn

from neatrl.td3 import train_td3


class ActorNet(nn.Module):
    def __init__(
        self,
        state_space: Union[int, tuple[int, ...]],
        action_space: int,
    ) -> None:
        super().__init__()
        state_dim = state_space[0] if isinstance(state_space, tuple) else state_space
        print(f"State dim: {state_dim}, Action dim: {action_space}")
        
        # Larger network for more complex environment
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.out = nn.Linear(300, action_space)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.out(x))
        x = x * 1.0  # Scale to action limits
        return x


class QNet(nn.Module):
    def __init__(self, state_space: Union[int, tuple[int, ...]], action_space: int):
        super().__init__()
        state_dim = state_space[0] if isinstance(state_space, tuple) else state_space

        # Larger network for more complex environment
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(action_space, 400)
        self.fc3 = nn.Linear(800, 300)
        self.out = nn.Linear(300, 1)

    def forward(self, state, act):
        st = torch.relu(self.fc1(state))
        action = torch.relu(self.fc2(act))
        temp = torch.cat((st, action), dim=1)
        x = torch.relu(self.fc3(temp))
        x = self.out(x)
        return x


if __name__ == "__main__":
    # Train TD3 on BipedalWalker
    train_td3(
        env_id="BipedalWalker-v3",
        total_timesteps=500000,
        learning_rate=3e-4,
        buffer_size=200000,
        gamma=0.99,
        tau=0.005,
        batch_size=256,
        learning_starts=10000,
        train_frequency=2,  # Delayed policy updates
        target_network_frequency=1,
        policy_noise=0.2,  # Target policy smoothing
        exploration_noise=0.1,
        noise_clip=0.5,
        low=-1.0,
        high=1.0,
        use_wandb=True,
        wandb_project="TD3-Box2D",
        exp_name="td3_bipedal_walker",
        eval_every=5000,
        save_every=25000,
        num_eval_episodes=10,
        capture_video=True,
        actor_class=ActorNet,
        q_network_class=QNet,
    )
