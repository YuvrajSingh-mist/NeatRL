"""
DDPG Example for BipedalWalker Environment

This example demonstrates how to use DDPG for the BipedalWalker environment.
BipedalWalker has continuous action spaces and vector observations, making it perfect for standard DDPG.
"""
import gymnasium as gym
from neatrl.ddpg import train_ddpg

env = gym.make("BipedalWalker-v3")

def main():
    """Train DDPG on BipedalWalker environment."""

    train_ddpg(
        # env_id="BipedalWalker-v3",  # BipedalWalker environment
        env=env, 
        total_timesteps=1000000,  # BipedalWalker needs many timesteps
        seed=42,
        learning_rate=3e-4,
        buffer_size=100000,
        batch_size=256,
        learning_starts=25000,
        train_frequency=2,
        target_network_frequency=50,
        gamma=0.99,
        tau=0.005,
        exploration_fraction=0.1,
        use_wandb=True,  # Set to True to enable logging
        capture_video=False,
        eval_every=10000,
        save_every=100000,
        num_eval_episodes=2,
        normalize_obs=True,
        device="cpu"  # Use "cuda" if you have GPU
    )

if __name__ == "__main__":
    main()