"""
A2C Example for FrozenLake Environment

This example demonstrates how to use A2C for the FrozenLake environment.
FrozenLake has discrete action spaces and grid observations, making it suitable for A2C with grid_env support.
"""

from neatrl.a2c import train_a2c


def main():
    """Train A2C on FrozenLake environment."""

    train_a2c(
        env_id="FrozenLake-v1",  # FrozenLake environment
        total_timesteps=500000,  # Total timesteps for training
        seed=42,
        lr=1e-3,  # Learning rate
        gamma=0.99,  # Discount factor
        max_grad_norm=0.5,  # Gradient clipping
        use_wandb=True,  # Set to True to enable logging
        capture_video=True,  # FrozenLake doesn't render well as video
        eval_every=10000,  # Evaluate every 10k steps
        save_every=50000,  # Save model every 50k steps
        num_eval_episodes=10,
        grid_env=True,  # Enable grid environment wrapper (one-hot encoding)
        normalize_obs=False,  # Grid env uses one-hot encoding
        device="cpu",  # Use "cuda" if you have GPU
    )


if __name__ == "__main__":
    main()
