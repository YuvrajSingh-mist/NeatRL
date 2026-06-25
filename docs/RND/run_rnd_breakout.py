"""
script for RND-PPO training on CarRacing using neatrl library.
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from neatrl.rnd import train_ppo_rnd_cnn


# --- Networks ---
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class FeatureExtractor(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 7 * 7, 512)),  # For 84x84 input after convs
            nn.ReLU(),
        )

    def forward(self, x):
        return self.network(x)  # Normalize image


class ActorNet(nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()

        self.network = FeatureExtractor(state_space)

        self.out = layer_init(nn.Linear(512, action_space))

    def forward(self, x):
        x = self.network(x / 255.0)
        out = torch.nn.functional.softmax(self.out(x), dim=-1)
        return out

    def get_action(self, x, action=None):
        probs = self.forward(x)
        dist = torch.distributions.Categorical(probs)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob, dist


class CriticNet(nn.Module):
    def __init__(self, state_space):
        super().__init__()
        self.network = FeatureExtractor(state_space)
        self.value_ext = layer_init(nn.Linear(512, 1), std=1.0)
        self.value_int = layer_init(nn.Linear(512, 1), std=1.0)

    def forward(self, x):
        x = self.network(x / 255.0)
        return self.value_ext(x), self.value_int(x)


class PredictorNet(nn.Module):
    def __init__(self, state_space):
        super().__init__()

        self.network = FeatureExtractor(state_space)

        self.out = nn.Linear(512, 256)

    def forward(self, x):
        x = self.network(x / 255.0)

        return self.out(x)


class TargetNet(nn.Module):
    def __init__(self, state_space):
        super().__init__()
        self.network = FeatureExtractor(state_space)

        self.out = layer_init(nn.Linear(512, 256))

    def forward(self, x):
        x = self.network(x / 255.0)
        return self.out(x)


def test_rnd_ppo_carracing():
    """Test RND-PPO training on CarRacing-v3."""
    print("Testing RND-PPO training on CarRacing-v3 with neatrl...")

    # Train RND-PPO on CarRacing
    model = train_ppo_rnd_cnn(
        env=gym.make("BreakoutNoFrameskip-v4", render_mode="rgb_array"),
        total_timesteps=500000,
        seed=42,
        lr=3e-4,
        ext_gamma=0.99,
        int_gamma=0.99,
        n_envs=4,
        max_steps=128,
        num_minibatches=4,
        PPO_EPOCHS=4,
        clip_value=0.2,
        ENTROPY_COEFF=0.01,
        VALUE_COEFF=0.5,
        EXT_COEFF=2.0,
        INT_COEFF=1.0,
        capture_video=True,
        use_wandb=True,
        wandb_project="cleanRL",
        exp_name="RND-PPO-CarRacing",
        grid_env=False,
        atari_wrapper=True,
        actor_class=ActorNet,
        critic_class=CriticNet,
        predictor_class=PredictorNet,
        target_class=TargetNet,
    )

    print("Training completed!")
    return model


if __name__ == "__main__":
    test_rnd_ppo_carracing()
