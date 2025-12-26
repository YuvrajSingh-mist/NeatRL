"""
A2C CNN Example for CarRacing Environment

This example demonstrates how to use A2C with CNN networks for the CarRacing environment.
CarRacing has image observations and continuous action spaces, making it perfect for A2C with CNN.
"""

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from neatrl.a2c import train_a2c_cnn


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
            low=0,
            high=255,
            shape=(self.num_stack, self.height, self.width),
            dtype=np.uint8,
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
        return result


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
        self.fc = layer_init(nn.Linear(64 * 7 * 7, 512))
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
        self.network = FeatureExtractor(obs_shape)

        # CarRacing has 3 continuous actions: steering, gas, brake
        self.mu = layer_init(nn.Linear(512, action_space))
        self.sigma = nn.Parameter(torch.zeros(action_space)) 
        

    def forward(self, x):
        
        x = self.network(x / 255.0)
        dist = torch.distributions.Normal(
            self.mu(x), torch.exp(self.sigma.expand_as(self.mu(x)))
        ) 
        
        return dist
    
    def get_action(self, x):
        dist = self.forward(x)
        action = dist.sample()
        out1 = torch.nn.functional.tanh(action[:, 0:1])  # Steering: -1 to 1
        out2 = torch.nn.functional.sigmoid(action[:, 1:2])  # Gas: 0 to 1
        out3 = torch.nn.functional.sigmoid(action[:, 2:3])  # Brake: 0 to 1
        
        final = torch.cat([out1, out2, out3], dim=-1)
        logprobs = dist.log_prob(final)
        return final, logprobs, dist


class CriticNet(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()
        self.network = FeatureExtractor(obs_shape)
        self.out = layer_init(nn.Linear(512, 1), std=1.0)

    def forward(self, x):
        x = self.network(x / 255.0)
        return self.out(x)

# Create the CarRacing environment
env = gym.make("CarRacing-v3", continuous=False)

def main():
    """Train A2C with CNN on CarRacing environment."""

    train_a2c_cnn(
        # env_id="CarRacing-v3",  # CarRacing environment
        env=env,
        total_timesteps=500000,  # Total timesteps for training
        seed=42,
        lr=3e-4,  # Learning rate
        gamma=0.99,  # Discount factor
        use_wandb=True,  # Set to True to enable logging
        capture_video=True,
        eval_every=1,  # Evaluate every 10k steps
        save_every=50000,  # Save model every 50k steps
        num_eval_episodes=5,
        device="cpu",  # Use "cuda" if you have GPU
        actor_class=ActorNet,
        critic_class=CriticNet,
        env_wrapper=car_racing_wrapper,
    )


if __name__ == "__main__":
    main()
