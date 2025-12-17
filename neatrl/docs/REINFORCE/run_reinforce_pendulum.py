#!/usr/bin/env python3
"""
script for REINFORCE training on MountainCar using neatrl library.
"""

from neatrl import train_reinforce

import torch
import torch.nn as nn

class PolicyNet(nn.Module):
    def __init__(self, state_space, action_space):
        super(PolicyNet, self).__init__()
        print(f"State space: {state_space}, Action space: {action_space}")
        self.fc1 = nn.Linear(state_space, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 16)
        self.mean = nn.Linear(16, action_space)
        self.logstd = nn.Linear(16, action_space)

    def forward(self, x):
        x =  self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))
  
        mean = self.mean(x)
        std = torch.exp(self.logstd(x)) # Ensure std is positive
   
        return mean, std
    
    def get_action(self, x):
        
        mean, std = self.forward(x)
        dist = torch.distributions.Normal(mean, std)  # Create a normal distribution from the mean and std
        action = dist.sample()  # Sample an action from the distribution
        action = action.clamp(-2, 2)
        # print(action)
        return action, dist.log_prob(action).sum(dim=-1) 
    
    
def test_reinforce_pendulum():
    """Test REINFORCE training on Pendulum-v0."""
    print("Testing REINFORCE training on Pendulum-v0 with neatrl...")

    # Train REINFORCE on Pendulum
    model = train_reinforce(
        env_id="Pendulum-v1",
        total_steps=200000,
        seed=42,
        learning_rate=2e-3,
        gamma=0.99,
        capture_video=True,
        use_wandb=True,
        wandb_project="cleanRL",
        wandb_entity="",
        exp_name="REINFORCE-Pendulum",
        eval_every=10000,
        save_every=20000,
        atari_wrapper=False,
        n_envs=4,
        num_eval_eps=10,
        device="cpu",
        grid_env=False,
        
        custom_agent = PolicyNet(3,1)
    )

    print("REINFORCE training on Pendulum-v0 completed successfully!")
    return model


if __name__ == "__main__":
    test_reinforce_pendulum()