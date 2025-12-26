"""
DDPG Example for BipedalWalker Environment

This example demonstrates how to use DDPG for the BipedalWalker environment.
BipedalWalker has continuous action spaces and vector observations, making it perfect for standard DDPG.
"""

from neatrl.ddpg import train_ddpg


def train_ddpg_bipedal_walker():
    """Train DDPG on BipedalWalker environment."""

    train_ddpg(
        env_id="BipedalWalker-v3",  # BipedalWalker environment
        total_timesteps=4000000,  # BipedalWalker needs many timesteps
        seed=42,
        learning_rate=3e-4,
        buffer_size=50000,
        batch_size=256,
        learning_starts=25000,
        train_frequency=4,
        target_network_frequency=10,
        gamma=0.99,
        tau=0.005,
        exploration_fraction=0.1,
        use_wandb=True,  # Set to True to enable logging
        capture_video=True,
        eval_every=10000,
        save_every=100000,
        num_eval_episodes=2,
        normalize_obs=True,
        device="cpu",  # Use "cuda" if you have GPU
        low=-1.0,
        high=1.0,
    )


if __name__ == "__main__":
    train_ddpg_bipedal_walker()
