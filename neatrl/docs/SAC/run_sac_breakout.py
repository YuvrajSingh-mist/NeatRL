"""
SAC CNN Example for Breakout Environment

This example demonstrates how to use SAC with CNN networks for the Breakout environment.
Breakout has image observations and discrete actions, adapted for SAC.
"""

from neatrl.sac import train_sac_cnn

import torch
import torch.nn as nn


class ActorNetCNN(nn.Module):
    def __init__(self, obs_shape, action_space):
        super().__init__()
        # Convolutional layers for image processing
        self.conv1 = nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.out = nn.Linear(512, action_space)

    def forward(self, x):
        
        # x = x.permute(0, 3, 1, 2)
        # x shape: (batch, channels, height, width)
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.nn.functional.relu(self.fc1(x))
        res = self.out(x)  # Output between -1 and 1
        return res
    
    def get_action(self, state: torch.Tensor) -> torch.Tensor:
        logits = self.forward(state)
        out = torch.nn.functional.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(out)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob.unsqueeze(-1)


class QNetCNN(nn.Module):
    def __init__(self, obs_shape, action_space):
        super().__init__()
        self.action_space = action_space  # Store for one-hot encoding
        # Convolutional layers for image processing
        self.conv1 = nn.Conv2d(obs_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # State processing
        self.state_fc = nn.Linear(64 * 7 * 7, 512)

        # Action processing
        self.action_fc = nn.Linear(action_space, 512)

        # Combined processing
        self.combined_fc = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 1)

    def forward(self, state, action):
        print(state.shape)
        # Process state through conv layers
        x = torch.nn.functional.relu(self.conv1(state))
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        state_features = torch.nn.functional.relu(self.state_fc(x))

        # Process action - one-hot encode discrete action
        action_onehot = torch.nn.functional.one_hot(action.long(), num_classes=self.action_space).float()
        action_features = torch.nn.functional.relu(self.action_fc(action_onehot))
        
        # Combine state and action features
        combined = torch.cat([state_features, action_features], dim=-1)
        x = torch.nn.functional.relu(self.combined_fc(combined))
        return self.out(x)


def test_breakout():
    """Train SAC CNN on CarRacing environment."""

    train_sac_cnn(
        env_id="BreakoutNoFrameskip-v4",  # CarRacing environment
        total_timesteps=1000000,  # Total timesteps for training
        seed=42,
        learning_rate=3e-4,  # Learning rate
        alpha=0.2,  # Entropy regularization coefficient
        autotune_alpha=True,  # Automatically tune alpha
        target_entropy_scale=-1.0,  # Target entropy = scale * action_dim
        gamma=0.99,  # Discount factor
        tau=0.005,  # Soft update parameter
        batch_size=256,  # Batch size
        learning_starts=25,  # Steps before training starts
        policy_frequency=4,  # Update policy every step
        use_wandb=True,  # Set to True to enable logging
        target_network_frequency=50,
        capture_video=True,
        eval_every=5000,  # Evaluate every 5k steps
        save_every=50000,  # Save model every 50k steps
        num_eval_episodes=5,
        device="cpu",  # Use "cuda" if you have GPU
        atari_wrapper=True,
        log_gradients=True,
        actor_class=ActorNetCNN,
        q_network_class=QNetCNN,
        n_envs=4
    )


if __name__ == "__main__":
    test_breakout()