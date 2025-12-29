"""
SAC Example for Pendulum Environment

This example demonstrates how to use SAC for the Pendulum continuous control environment.
Pendulum has continuous action spaces and vector observations, making it suitable for standard SAC.
"""

import gymnasium as gym
from neatrl.sac import train_sac

env = gym.make("Pendulum-v1")

def test_pendulum():
    """Train SAC on Pendulum environment."""

    train_sac(
        env=env,
        # env_id="Pendulum-v1",  # Pendulum environment
        total_timesteps=500000,  # Total timesteps for training
        seed=42,
        learning_rate=3e-4,  # Learning rate
        alpha=0.2,  # Entropy regularization coefficient
        autotune_alpha=True,  # Automatically tune alpha
        target_entropy_scale=-1.0,  # Target entropy = scale * action_dim
        gamma=0.99,  # Discount factor
        tau=0.005,  # Soft update parameter
        batch_size=256,  # Batch size
        learning_starts=25000,  # Steps before training starts
        policy_frequency=4,  # Update policy every step
        target_network_frequency=50,
        use_wandb=True,  # Set to True to enable logging
        capture_video=True,
        eval_every=5000,  # Evaluate every 5k steps
        save_every=25000,  # Save model every 25k steps
        num_eval_episodes=10,
        normalize_obs=False,  # Pendulum doesn't need normalization
        normalize_reward=False,  # Pendulum rewards are already scaled
        device="cpu",  # Use "cuda" if you have GPU
        log_gradients=True,
        n_envs=4
    )


if __name__ == "__main__":
    test_pendulum()