"""
SAC Example for BipedalWalker Environment

This example demonstrates how to use SAC for the BipedalWalker continuous control environment.
BipedalWalker has continuous action spaces and vector observations, making it suitable for standard SAC.
"""

import gymnasium as gym
from neatrl.sac import train_sac


def train_run_bipedal_walker():
    """Train SAC on BipedalWalker environment."""

    train_sac(
        env_id="BipedalWalker-v3",  # BipedalWalker environment
        total_timesteps=2000000,  # Total timesteps for training (needs more for this env)
        seed=42,
        learning_rate=3e-4,  # Learning rate
        alpha=0.2,  # Entropy regularization coefficient
        autotune_alpha=True,  # Automatically tune alpha
        target_entropy_scale=-1.0,  # Target entropy = scale * action_dim
        gamma=0.99,  # Discount factor
        tau=0.005,  # Soft update parameter
        batch_size=256,  # Batch size
        learning_starts=25000,  # Steps before training starts
        policy_frequency=4,  # Update policy every 4 steps
        target_network_frequency=50,  # Update target networks every step
        use_wandb=True,  # Set to True to enable logging
        capture_video=True,
        eval_every=10000,  # Evaluate every 10k steps
        save_every=100000,  # Save model every 100k steps
        num_eval_episodes=5,
        normalize_obs=False,  # Normalize observations
        normalize_reward=False,  # Normalize rewards
        device="cpu",  # Use "cuda" if you have GPU
        log_gradients=True,
    )


if __name__ == "__main__":
    train_run_bipedal_walker()