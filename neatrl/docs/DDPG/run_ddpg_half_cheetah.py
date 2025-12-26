"""
DDPG Example for MuJoCo Half Cheetah Environment

This example demonstrates how to use DDPG on the MuJoCo Half Cheetah environment.
DDPG (Deep Deterministic Policy Gradient) is suitable for continuous action spaces,
and Half Cheetah is a classic continuous control benchmark with 17-dimensional state space
and 6-dimensional continuous action space.
"""

from neatrl.ddpg import train_ddpg


def main():
    """Train DDPG on MuJoCo Half Cheetah environment."""

    train_ddpg(
        env_id="HalfCheetah-v5",
        total_timesteps=1000000,  # MuJoCo environments need more timesteps
        seed=42,
        learning_rate=3e-4,
        buffer_size=1000000,
        batch_size=256,
        learning_starts=10000,
        train_frequency=1,
        target_network_frequency=1000,
        gamma=0.99,
        tau=0.005,
        exploration_fraction=0.1,
        use_wandb=True,  # Set to True to enable logging
        capture_video=True,
        eval_every=10000,
        save_every=50000,
        num_eval_episodes=2,
        device="cpu",  # Use "cuda" if you have GPU
    )


if __name__ == "__main__":
    main()
