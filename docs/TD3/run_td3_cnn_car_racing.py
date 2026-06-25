"""
TD3 Example for CarRacing with CNN

This example demonstrates TD3 with CNNs on pixel-based environments.
Uses the CarRacing-v2 environment with frame stacking.
"""

from typing import Union

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from neatrl.td3 import train_td3_cnn


def car_racing_wrapper(env):
    return PreprocessAndFrameStack(env, height=84, width=84, num_stack=4)


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
            low=0.0,
            high=1.0,
            shape=(self.num_stack, self.height, self.width),
            dtype=np.float32,
        )

    def observation(self, obs):
        # Convert LazyFrames to array if needed
        obs_array = np.array(obs)

        frames = []
        for i in range(self.num_stack):
            frame = obs_array[i]

            if frame.ndim == 3 and frame.shape[2] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

            frame = cv2.resize(
                frame, (self.width, self.height), interpolation=cv2.INTER_AREA
            )
            frames.append(frame)

        result = np.stack(frames, axis=0)
        return result / 255.0


# --- Networks ---
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class FeatureExtractor(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        self.conv1 = layer_init(nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4))
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.flatten = nn.Flatten()
        self.fc = layer_init(nn.Linear(64 * 7 * 7, 512))

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = self.flatten(x)
        x = torch.nn.functional.relu(self.fc(x))
        return x


class ActorNet(nn.Module):
    def __init__(self, obs_shape, action_space):
        super().__init__()
        self.network = FeatureExtractor(obs_shape)
        state_dim = obs_shape[0] if isinstance(obs_shape, tuple) else obs_shape
        print(f"State dim: {state_dim}, Action dim: {action_space}")

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, action_space)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.network(x)
        x = self.out(
            torch.nn.functional.mish(self.fc2(torch.nn.functional.mish(self.fc1(x))))
        )
        return x

    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        x = self.forward(state)
        action1 = x[:, 0:1]
        action2 = x[:, 1:2]
        action3 = x[:, 2:3]

        out1 = torch.tanh(action1)
        out2 = torch.sigmoid(action2)
        out3 = torch.sigmoid(action3)
        out = torch.cat([out1, out2, out3], dim=1)

        return out


class QNet(nn.Module):
    def __init__(self, state_space: Union[int, tuple[int, ...]], action_space: int):
        super().__init__()

        self.feature_extractor = FeatureExtractor(state_space)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(action_space, 256)
        self.fc3 = nn.Linear(512, 512)
        self.reduce = nn.Linear(512, 256)
        self.out = nn.Linear(256, 1)

    def forward(self, state, act):
        features = self.feature_extractor(state)
        st = torch.nn.functional.mish(self.fc1(features))
        action = torch.nn.functional.mish(self.fc2(act))
        combined = torch.cat((st, action), dim=1)
        combined_features = torch.nn.functional.mish(self.fc3(combined))
        out = torch.nn.functional.mish(self.reduce(combined_features))
        x = self.out(out)
        return x


env = gym.make("CarRacing-v3")

if __name__ == "__main__":
    # Train TD3 with CNN on CarRacing
    train_td3_cnn(
        # env_id="CarRacing-v3",
        env=env,
        total_timesteps=1000000,
        learning_rate=3e-4,
        buffer_size=10000,
        gamma=0.99,
        tau=0.005,
        batch_size=128,
        learning_starts=2500,
        train_frequency=4,  # Delayed policy updates
        target_network_frequency=10,
        policy_noise=0.2,  # Target policy smoothing
        exploration_noise=0.1,
        noise_clip=0.5,
        low=-1.0,
        high=1.0,
        use_wandb=True,
        wandb_project="TD3-Vision",
        exp_name="td3_cnn_carracing",
        eval_every=1000,
        save_every=50000,
        num_eval_episodes=5,
        capture_video=True,
        actor_class=ActorNet,
        q_network_class=QNet,
        device="cuda" if torch.cuda.is_available() else "cpu",
        env_wrapper=car_racing_wrapper,
    )
