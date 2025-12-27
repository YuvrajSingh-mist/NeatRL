"""
TD3 Example for HalfCheetah Environment

This example demonstrates TD3 on the HalfCheetah-v5 MuJoCo environment.
HalfCheetah is a standard benchmark for continuous control algorithms.
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
        
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, action_space)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.out(
                torch.nn.functional.mish(
                    self.fc2(torch.nn.functional.mish(self.fc1(x)))
                )
            )
        return x
    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.tanh(self.forward(state))
        x = x * 1.0  # Scale to action limits
        return x


class QNet(nn.Module):
    def __init__(self, state_space: Union[int, tuple[int, ...]], action_space: int):
        super().__init__()
        state_dim = state_space[0] if isinstance(state_space, tuple) else state_space

        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(action_space, 256)
        self.fc3 = nn.Linear(512, 512)
        self.reduce = nn.Linear(512, 256)
        self.out = nn.Linear(256, 1)

    def forward(self, state, act):
        st = torch.nn.functional.mish(self.fc1(state))
        action = torch.nn.functional.mish(self.fc2(act))
        temp = torch.cat((st, action), dim=1)
        x = torch.nn.functional.mish(self.fc3(temp))
        x = torch.nn.functional.mish(self.reduce(x))
        x = self.out(x)
        return x


if __name__ == "__main__":
    # Train TD3 on HalfCheetah with high-performance settings
    train_td3(
        env_id="HalfCheetah-v5",
        total_timesteps=1000000,
        learning_rate=3e-4,
        buffer_size=1000000,
        gamma=0.99,
        tau=0.005,
        batch_size=256,
        learning_starts=25000,
        train_frequency=2,  # Delayed policy updates (TD3 trick)
        target_network_frequency=1,  # Frequent soft updates
        policy_noise=0.2,  # Target policy smoothing (TD3 trick)
        exploration_noise=0.1,  # Exploration during training
        noise_clip=0.5,
        low=-1.0,
        high=1.0,
        use_wandb=True,
        wandb_project="TD3-MuJoCo",
        exp_name="td3_halfcheetah_1M",
        eval_every=10000,
        save_every=50000,
        num_eval_episodes=10,
        capture_video=True,
        actor_class=ActorNet,
        q_network_class=QNet,
        device="cpu",  # Use GPU if available
        n_envs=4
    )
