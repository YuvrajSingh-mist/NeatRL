"""
script for PPO training on Breakout using neatrl library.
"""

import numpy as np
import torch
from torch import nn

from neatrl.ppo import train_ppo_cnn


# --- Networks ---
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class FeatureExtractor(nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        self.conv1 = layer_init(nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4))
        self.relu1 = nn.ReLU()
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.relu2 = nn.ReLU()
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear = layer_init(
            nn.Linear(64 * 7 * 7, 512)
        )  # For 84x84 input after convs
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.flatten(x)
        x = self.relu4(self.linear(x))
        return x  # Normalize images outside


class ActorNet(nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()

        self.network = FeatureExtractor(state_space)

        self.out = layer_init(nn.Linear(512, action_space), std=0.01)

    def forward(self, x):
        x = self.network(x / 255.0)
        out = torch.nn.functional.softmax(self.out(x), dim=-1)
        return out

    def get_action(self, x, action=None):
        features = self.network(x)
        logits = self.out(features)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)

        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob, dist


class CriticNet(nn.Module):
    def __init__(self, state_space):
        super().__init__()
        self.network = FeatureExtractor(state_space)
        self.value = layer_init(nn.Linear(512, 1), std=1.0)

    def forward(self, x):
        x = self.network(x / 255.0)
        return self.value(x)


def test_ppo_breakout():
    """Test PPO training on BreakoutNoFrameskip-v4."""
    print("Testing PPO training on BreakoutNoFrameskip-v4 with neatrl...")

    # Train PPO on Breakout
    model = train_ppo_cnn(
        env_id="BreakoutNoFrameskip-v4",
        total_timesteps=10000000,
        seed=42,
        lr=2.5e-4,
        gamma=0.99,
        GAE=0.95,
        n_envs=8,
        max_steps=128,
        num_minibatches=4,
        PPO_EPOCHS=4,
        clip_value=0.1,
        VALUE_COEFF=0.5,
        ENTROPY_COEFF=0.01,
        max_grad_norm=0.5,
        anneal_lr=True,
        capture_video=True,
        use_wandb=True,
        wandb_project="cleanRL",
        exp_name="PPO-Breakout",
        atari_wrapper=True,
        eval_every=1,
        save_every=10000,
        num_eval_episodes=2,
        actor_class=ActorNet,
        critic_class=CriticNet,
        log_gradients=False,
    )

    print("Training completed!")
    return model


if __name__ == "__main__":
    test_ppo_breakout()
