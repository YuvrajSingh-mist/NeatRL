"""
script for RND-PPO training on CarRacing using neatrl library.
"""

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from neatrl.reinforce import train_reinforce_cnn


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
        # `obs` here is a LazyFrames object from FrameStack of shape (num_stack, H, W, C)
        # 1. Convert LazyFrames to a single numpy array

        stack = np.array(obs, dtype=np.uint8)

        # 2. Extract 'screen' if obs is a dict (for VizDoom)
        if (
            isinstance(self.env.observation_space, gym.spaces.Dict)
            and "screen" in stack[0]
        ):
            stack = np.array(list(stack))
        else:
            stack = np.array(list(stack))

        # 3. Grayscale and Resize each frame in the stack
        processed_stack = []
        for frame in stack:
            if frame.ndim == 3 and frame.shape[2] == 3:  # H, W, C
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = cv2.resize(
                frame, (self.width, self.height), interpolation=cv2.INTER_AREA
            )
            processed_stack.append(frame)

        # 4. Stack frames along a new channel dimension
        return np.stack(processed_stack, axis=0)


def car_racing_wrapper(env):
    return PreprocessAndFrameStack(env, height=84, width=84, num_stack=4)


# --- Networks ---
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
    
        self.conv1 = layer_init(nn.Conv2d(3, 32, kernel_size=8, stride=4))
        self.relu1 = nn.ReLU()
        self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
        self.relu2 = nn.ReLU()
        self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
        self.relu3 = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = layer_init(nn.Linear(64 * 8 * 8, 512))  # Adjusted for 96x96 input
        self.relu_fc = nn.ReLU()
    

    def forward(self, x):
        
        x = self.relu1(self.conv1(x))
        
        x = self.relu2(self.conv2(x))
     
        x = self.relu3(self.conv3(x))
  
        x = self.flatten(x)
      
        x = self.relu_fc(self.fc(x))
        return x


class ActorNet(nn.Module):
    def __init__(self, action_space):
        super().__init__()

        self.network = FeatureExtractor()

        self.out = layer_init(nn.Linear(512, action_space))

    def forward(self, x):
        x = x.unsqueeze(0)  
        x = x.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]
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


def test_reinforce_carracing():
    """Test RND-PPO training on CarRacing-v3."""
    print("Testing RND-PPO training on CarRacing-v3 with neatrl...")

    # Train RND-PPO on CarRacing
    model = train_reinforce_cnn(
        
        env=gym.make("CarRacing-v3", render_mode="rgb_array", continuous=False),
        total_steps=2000,
        seed=42,
        learning_rate=2e-3,
        gamma=0.99,
        capture_video=True,
        use_wandb=True,
        wandb_project="cleanRL",
        wandb_entity="",
        exp_name="REINFORCE-CarRacing",
        eval_every=100,
        save_every=1000,
        atari_wrapper=False,
        n_envs=1,
        num_eval_eps=10,
        device="cpu",
        grid_env=False,
        custom_agent=ActorNet,
        env_wrapper=car_racing_wrapper
    )

    print("Training completed!")
    return model


if __name__ == "__main__":
    test_reinforce_carracing()
