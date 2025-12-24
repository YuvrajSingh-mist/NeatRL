"""
script for PPO training on Bipedal Walker using neatrl library.
"""

from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from neatrl.ppo import train_ppo


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

        self.fc1 = layer_init(nn.Linear(state_space, 128))
        self.fc2 = layer_init(nn.Linear(128, 64))
        self.fc3 = layer_init(nn.Linear(64, 32))
        self.mu = layer_init(nn.Linear(32, action_space))
        self.log_std = nn.Parameter(torch.zeros(action_space))

    def forward(self, x: torch.Tensor) -> torch.distributions.Distribution:
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))

        mu = self.mu(x)
        logstd = self.log_std.expand_as(mu)
        std = logstd.exp()
        dist = torch.distributions.Normal(mu, std)
        return dist

    def get_action(
        self,
        x: torch.Tensor,
        action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.distributions.Distribution]:
        dist = self.forward(x)
        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)

        return action, log_prob, dist


class CriticNet(nn.Module):
    def __init__(self, state_space: Union[int, tuple[int, ...]]) -> None:
        super().__init__()
        self.fc1 = layer_init(nn.Linear(state_space, 128))
        self.fc2 = layer_init(nn.Linear(128, 64))
        self.fc3 = layer_init(nn.Linear(64, 32))
        self.value = layer_init(nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.value(x)


def test_ppo_bipedal_walker():
    """Test PPO training on BipedalWalker-v3."""
    print("Testing PPO training on BipedalWalker-v3 with neatrl...")

    # Train PPO on Bipedal Walker
    model = train_ppo(
        env_id="BipedalWalker-v3",
        total_timesteps=2000000,
        seed=42,
        lr=3e-4,
        gamma=0.99,
        GAE=0.95,
        n_envs=8,
        max_steps=512,
        num_minibatches=4,
        PPO_EPOCHS=4,
        clip_value=0.2,
        VALUE_COEFF=0.5,
        ENTROPY_COEFF=0.01,
        max_grad_norm=0.5,
        anneal_lr=True,
        capture_video=True,
        use_wandb=True,
        wandb_project="cleanRL",
        exp_name="PPO-BipedalWalker",
        eval_every=100,
        save_every=5000,
        num_eval_episodes=1,
        log_gradients=False,
        critic_class= CriticNet,
        actor_class= ActorNet,
    )

    print("Training completed!")
    return model


if __name__ == "__main__":
    test_ppo_bipedal_walker()
