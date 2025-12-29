"""
SAC CNN Example for CarRacing Environment

This example demonstrates how to use SAC with CNN networks for the CarRacing environment.
CarRacing has image observations and continuous actions, requiring CNN-based networks.
"""

from neatrl.sac import train_sac_cnn
import torch
import torch.nn as nn
import gymnasium as gym
from typing import Union
import numpy as np
import cv2

LOG_STD_MIN = -20
LOG_STD_MAX = 2


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
        self.action_dim = action_space
        
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mean = nn.Linear(256, action_space)
        self.log_std = nn.Linear(256, action_space)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.network(x)
        x = torch.nn.functional.relu(feat)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std
    
    def get_action(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Get action from the policy with reparameterization trick."""
        mean, log_std = self.forward(x)
        std = log_std.exp()
        
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # Reparameterization trick
        action1 = torch.tanh(x_t[:, 0:1])
        action2 = torch.sigmoid(x_t[:, 1:2])
        action3 = torch.sigmoid(x_t[:, 2:3])
        action = torch.cat([action1, action2, action3], dim=1)
        
        # Compute log probability with correction for tanh squashing
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob


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
    
env = gym.make("CarRacing-v3", continuous=True)

def test_car_racing():
    """Train SAC CNN on CarRacing environment."""

    train_sac_cnn(
        # env_id="CarRacing-v3",  # CarRacing environment
        env=env,
        total_timesteps=1000000,  # Total timesteps for training
        seed=42,
        learning_rate=3e-4,  # Learning rate
        alpha=0.2,  # Entropy regularization coefficient
        autotune_alpha=True,  # Automatically tune alpha
        target_entropy_scale=-1.0,  # Target entropy = scale * action_dim
        gamma=0.99,  # Discount factor
        tau=0.005,  # Soft update parameter
        batch_size=256,  # Batch size
        learning_starts=2500,  # Steps before training starts
        policy_frequency=4,  # Update policy every step
        use_wandb=True,  # Set to True to enable logging
        target_network_frequency=50,
        capture_video=True,
        eval_every=5000,  # Evaluate every 5k steps
        save_every=50000,  # Save model every 50k steps
        num_eval_episodes=5,
        # n_envs=4,
        device="cpu",  # Use "cuda" if you have GPU
        env_wrapper=car_racing_wrapper,
        log_gradients=True,
        actor_class=ActorNet,
        q_network_class=QNet
    )


if __name__ == "__main__":
    test_car_racing()