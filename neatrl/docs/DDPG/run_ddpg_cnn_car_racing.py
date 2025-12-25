"""
DDPG CNN Example for CarRacing Environment

This example demonstrates how to use DDPG with CNN networks for the CarRacing environment.
CarRacing has image observations and continuous action spaces, making it perfect for DDPG with CNN.

"""

from typing import Union

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from neatrl.ddpg import train_ddpg_cnn


class PreprocessAndFrameStack(gym.ObservationWrapper):
    """
    A wrapper that extracts the 'screen' observation, resizes, grayscales,
    and then stacks frames. This simplifies the environment interaction loop.
    Note: The order of operations is important.
    """

    def __init__(self, env, height, width, num_stack):
        # 1. First, apply the stacking wrapper to the raw environment
        env = gym.wrappers.FrameStackObservation(env, num_stack)
        super().__init__(env)
        self.height = height
        self.width = width
        self.num_stack = num_stack

        # The new observation space after all transformations
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.num_stack, self.height, self.width),
            dtype=np.uint8,
        )

    def observation(self, obs):
        frames = []
        for i in range(self.num_stack):
            frame = np.array(obs[i], dtype=np.uint8)
            if frame.ndim == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = cv2.resize(
                frame, (self.width, self.height), interpolation=cv2.INTER_AREA
            )
            frames.append(frame)
        return np.stack(frames, axis=0)


def car_racing_wrapper(env):
    return PreprocessAndFrameStack(env, height=84, width=84, num_stack=4)


# --- Networks ---
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class FeatureExtractor(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        self.conv1 = layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4))
        self.relu1 = nn.ReLU()
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.relu2 = nn.ReLU()
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = layer_init(nn.Linear(64 * 7 * 7, 512))  # Adjusted for 96x96 input
        self.relu_fc = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.conv1(x))

        x = self.relu2(self.conv2(x))

        x = self.relu3(self.conv3(x))

        x = self.flatten(x)

        x = self.relu_fc(self.fc(x))
        return x


class ActorNet(nn.Module):
    def __init__(self, obs_shape, action_space):
        super().__init__()
        print(obs_shape, action_space)
        self.network = FeatureExtractor(obs_shape)

        self.out1 = layer_init(nn.Linear(512, 1))
        self.out2 = layer_init(nn.Linear(512, 1))
        self.out3 = layer_init(nn.Linear(512, 1))

    def forward(self, x):
        x = self.network(x / 255.0)
        out1 = torch.nn.functional.tanh(self.out1(x))
        out2 = torch.nn.functional.tanh(self.out2(x))
        out3 = torch.nn.functional.tanh(self.out3(x))
        action1 = out1
        action2 = (out2 + 1) / 2
        action3 = (out3 + 1) / 2

        return torch.cat([action1, action2, action3], dim=-1)


class QNet(nn.Module):
    def __init__(self, obs_shape: Union[int, tuple[int, ...]], action_space: int):
        super().__init__()
        # Handle state_space as tuple or int
        state_dim = obs_shape[0] if isinstance(obs_shape, tuple) else obs_shape
        self.network = FeatureExtractor(obs_shape)

        self.fc2 = nn.Linear(action_space, 256)
        self.fc3 = nn.Linear(768, 512)
        self.out = nn.Linear(512, 1)  # Output a single Q-value

    def forward(self, state, act):
        st = self.network(state / 255.0)
        action = torch.nn.functional.mish(self.fc2(act))
        combined = torch.cat((st, action), dim=-1)  # Concatenate state and action
        x = torch.nn.functional.mish(self.fc3(combined))
        x = self.out(x)
        return x


def main():
    """Train DDPG with CNN on CarRacing environment."""

    train_ddpg_cnn(
        env_id="CarRacing-v3",  # CarRacing environment
        total_timesteps=500000,  # CarRacing needs more timesteps
        seed=42,
        learning_rate=1e-4,
        buffer_size=100000,
        batch_size=64,
        learning_starts=1000,
        train_frequency=4,
        target_network_frequency=100,
        gamma=0.99,
        tau=0.005,
        exploration_fraction=0.1,
        use_wandb=True,  # Set to True to enable logging
        capture_video=True,
        eval_every=10000,
        save_every=50000,
        num_eval_episodes=5,
        device="cpu",  # Use "cuda" if you have GPU
        actor_class=ActorNet,
        q_network_class=QNet,
        env_wrapper=car_racing_wrapper,
    )


if __name__ == "__main__":
    main()
