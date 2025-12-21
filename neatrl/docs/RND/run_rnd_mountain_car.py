"""
script for RND-PPO training on MountainCarContinuous using neatrl library.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union
import numpy as np

from neatrl import train_rnd

 
# --- Networks ---
def layer_init(
    layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0
) -> nn.Module:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorNet(nn.Module):
    def __init__(
        self,
        state_space: Union[int, tuple[int, ...]],
        action_space: int,
        
    ) -> None:
        super().__init__()
         
        self.fc1 = layer_init(nn.Linear(state_space, 32))
        self.fc2 = layer_init(nn.Linear(32, 32))
        self.fc3 = layer_init(nn.Linear(32, 16))
        self.mean = layer_init(nn.Linear(16, action_space))
        self.log_std = nn.Parameter(torch.zeros(action_space))

    def forward(self, x: torch.Tensor) -> torch.distributions.Distribution:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        mean = self.mean(x)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        return dist

    def get_action(
        self,
        x: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.distributions.Distribution]:
        dist = self.forward(x)
        if action is None:
            action = dist.rsample()  # reparameterized sample
            action = torch.tanh(action)  # squash to [-1,1]

        log_prob = dist.log_prob(action).sum(dim=-1)
        # adjust for tanh
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)

        return action, log_prob, dist


class CriticNet(nn.Module):
    def __init__(self, state_space: Union[int, tuple[int, ...]]) -> None:
        super().__init__()
        self.fc1 = layer_init(nn.Linear(state_space, 32))
        self.fc2 = layer_init(nn.Linear(32, 32))
        self.fc3 = layer_init(nn.Linear(32, 16))
        self.value_ext = layer_init(nn.Linear(16, 1))
        self.value_int = layer_init(nn.Linear(16, 1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.value_ext(x), self.value_int(x)


class PredictorNet(nn.Module):
    def __init__(self, state_space: Union[int, tuple[int, ...]]) -> None:
        super().__init__()
        self.fc1 = layer_init(nn.Linear(state_space, 64))
        self.fc2 = layer_init(nn.Linear(64, 64))
        self.fc3 = layer_init(nn.Linear(64, 32))
        self.out = layer_init(nn.Linear(32, 32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)


class ContinuousActorNet(ActorNet):
    def __init__(self, state_space, action_space):
        super().__init__(state_space, action_space, continuous=True)



def test_rnd_ppo_mountain_car():
    """Test RND-PPO training on MountainCarContinuous-v0."""
    print("Testing RND-PPO training on MountainCarContinuous-v0 with neatrl...")

    # Train RND-PPO on MountainCarContinuous
    model = train_rnd(
        env_id="MountainCarContinuous-v0",
        total_timesteps=5000000,
        seed=42,
        lr=3e-4,
        ext_gamma=0.99,
        int_gamma=0.95,
        n_envs=4,
        max_steps=128,
        num_minibatches=4,
        PPO_EPOCHS=4,
        clip_value=0.2,
        ENTROPY_COEFF=0.01,
        VALUE_COEFF=0.5,
        EXT_COEFF=1.0,
        INT_COEFF=2.0,
        capture_video=True,
        use_wandb=True,
        wandb_project="cleanRL",
        exp_name="RND-PPO-MountainCarContinuous",
        grid_env=False,
        eval_every=1000,
        save_every=1000,
        num_eval_episodes=1,
        anneal_lr=True,
        log_gradients=True,
        actor_class=ActorNet,
    )

    print("Training completed!")
    return model


if __name__ == "__main__":
    test_rnd_ppo_mountain_car()
