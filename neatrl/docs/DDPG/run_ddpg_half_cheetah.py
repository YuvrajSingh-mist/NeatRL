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

    # Train DDPG on Half Cheetah (you would need to install mujoco and mujoco-py)
    # pip install mujoco mujoco-py

    train_ddpg(
        env_id="HalfCheetah-v4",  # MuJoCo Half Cheetah environment
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
        use_wandb=False,  # Set to True to enable logging
        capture_video=False,
        eval_every=10000,
        save_every=50000,
        num_eval_episodes=10,
        device="cpu"  # Use "cuda" if you have GPU
    )

if __name__ == "__main__":
    main()