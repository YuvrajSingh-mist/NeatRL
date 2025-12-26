"""
DDPG Example for Pendulum Environment

This example demonstrates how to use DDPG for the Pendulum environment.
Pendulum has continuous action spaces and vector observations, making it perfect for standard DDPG.
"""

from neatrl.ddpg import train_ddpg
import torch
import torch.nn as nn
from typing import Union


class ActorNet(nn.Module):
    def __init__(
        self,
        state_space: Union[int, tuple[int, ...]],
        action_space: int,
    ) -> None:
        super().__init__()
        # Handle state_space as tuple or int
        state_dim = state_space[0] if isinstance(state_space, tuple) else state_space
        print(state_dim)
        print(action_space)
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
        x = x * 2.0  # Scale to action limits
        return x


class QNet(nn.Module):
    def __init__(self, state_space: Union[int, tuple[int, ...]], action_space: int):
        super().__init__()
        # Handle state_space as tuple or int
        state_dim = state_space[0] if isinstance(state_space, tuple) else state_space

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(action_space, 256)
        self.fc3 = nn.Linear(512, 512)
        self.reduce = nn.Linear(512, 256)  # Output a single Q-value
        self.out = nn.Linear(256, 1)

    def forward(self, state, act):
        st = torch.nn.functional.mish(self.fc1(state))
        action = torch.nn.functional.mish(self.fc2(act))
        temp = torch.cat((st, action), dim=1)  # Concatenate state and action
        x = torch.nn.functional.mish(self.fc3(temp))
        x = torch.nn.functional.mish(self.reduce(x))
        x = self.out(x)
        return x
    
def train_ddpg_pendulum():
    """Train DDPG on Pendulum environment."""

    train_ddpg(
        env_id="Pendulum-v1",  # Pendulum environment
        total_timesteps=4000000,  # Pendulum needs many timesteps
        seed=42,
        learning_rate=3e-4,
        buffer_size=50000,
        batch_size=256,
        learning_starts=25000,
        train_frequency=4,
        target_network_frequency=50,
        gamma=0.99,
        tau=0.005,
        exploration_fraction=0.1,
        use_wandb=True,  # Set to True to enable logging
        capture_video=True,
        eval_every=10000,
        save_every=100000,
        num_eval_episodes=2,
        normalize_obs=False,
        device="cpu",  # Use "cuda" if you have GPU
        actor_class=ActorNet,
        q_network_class=QNet,
        low=-2.0,
        high=2.0,
    )


if __name__ == "__main__":
    train_ddpg_pendulum()
