"""
SAC Example for HalfCheetah Environment

This example demonstrates how to use SAC for the HalfCheetah continuous control environment.
HalfCheetah has continuous action spaces and vector observations, making it suitable for standard SAC.
"""

from neatrl.sac import train_sac


def test_halfcheetah():
    """Train SAC on HalfCheetah environment."""

    train_sac(
        env_id="HalfCheetah-v5",  # HalfCheetah environment
        total_timesteps=1000000,  # Total timesteps for training
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
        eval_every=5000,  # Evaluate every 5k steps
        save_every=50000,  # Save model every 50k steps
        num_eval_episodes=3,
        normalize_obs=False,  # Normalize observations
        normalize_reward=False,  # Normalize rewards
        device="cpu",  # Use "cuda" if you have GPU
        log_gradients=True,
    )


if __name__ == "__main__":
    test_halfcheetah()