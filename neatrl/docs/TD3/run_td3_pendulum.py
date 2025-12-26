"""
TD3 Example for Pendulum Environment

This example demonstrates how to use TD3 for the Pendulum environment.
Pendulum has continuous action spaces and vector observations, making it perfect for standard TD3.
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
        # Handle state_space as tuple or int
        state_dim = state_space[0] if isinstance(state_space, tuple) else state_space
        print(f"State dim: {state_dim}, Action dim: {action_space}")
        
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, action_space)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(
            self.out(
                torch.nn.functional.mish(
                    self.fc2(torch.nn.functional.mish(self.fc1(x)))
                )
            )
        )
        x = x * 2.0  # Scale to action limits for Pendulum
        return x


class QNet(nn.Module):
    def __init__(self, state_space: Union[int, tuple[int, ...]], action_space: int):
        super().__init__()
        # Handle state_space as tuple or int
        state_dim = state_space[0] if isinstance(state_space, tuple) else state_space

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(action_space, 256)
        self.fc3 = nn.Linear(512, 512)
        self.reduce = nn.Linear(512, 256)
        self.out = nn.Linear(256, 1)

    def forward(self, state, act):
        st = torch.nn.functional.mish(self.fc1(state))
        action = torch.nn.functional.mish(self.fc2(act))
        temp = torch.cat((st, action), dim=1)  # Concatenate state and action
        x = torch.nn.functional.mish(self.fc3(temp))
        x = torch.nn.functional.mish(self.reduce(x))
        x = self.out(x)
        return x


if __name__ == "__main__":
    # Train TD3 on Pendulum
    train_td3(
        env_id="Pendulum-v1",
        total_timesteps=200000,
        learning_rate=3e-4,
        buffer_size=100000,
        gamma=0.99,
        tau=0.005,
        batch_size=256,
        learning_starts=10000,
        train_frequency=2,  # Delayed policy updates
        policy_noise=0.2,  # Target policy smoothing
        exploration_noise=0.1,  # Exploration noise
        noise_clip=0.5,
        low=-2.0,
        high=2.0,
        use_wandb=True,
        wandb_project="TD3-Pendulum",
        exp_name="td3_pendulum_baseline",
        eval_every=5000,
        num_eval_episodes=10,
        capture_video=True,
        actor_class=ActorNet,
        q_network_class=QNet,
    )
