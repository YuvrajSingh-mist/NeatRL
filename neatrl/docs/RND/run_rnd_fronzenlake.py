"""
script for RND-PPO training on FrozenLake using neatrl library.
"""

import gymnasium as gym

from neatrl import train_ppo_rnd


def test_rnd_ppo_frozenlake():
    """Test RND-PPO training on FrozenLake-v1 with deterministic actions."""
    print("Testing RND-PPO training on FrozenLake-v1 (deterministic) with neatrl...")

    # Create the deterministic FrozenLake environment directly with render mode for video capture
    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="rgb_array")

    # Train RND-PPO on FrozenLake with direct env object
    model = train_ppo_rnd(
        # env_id="FrozenLake-v1",  # This will be ignored when env is provided
        total_timesteps=20000000,
        seed=42,
        lr=1e-4,
        ext_gamma=0.999,
        int_gamma=0.99,
        n_envs=8,
        max_steps=128,
        num_minibatches=4,
        PPO_EPOCHS=4,
        clip_value=0.2,
        ENTROPY_COEFF=0.01,
        VALUE_COEFF=0.5,
        EXT_COEFF=5.0,
        INT_COEFF=0.5,
        capture_video=True,
        use_wandb=True,
        wandb_project="cleanRL",
        exp_name="RND-PPO-FrozenLake-Deterministic",
        grid_env=False,
        eval_every=100,
        save_every=1000,
        num_eval_episodes=5,
        anneal_lr=True,
        # Pass the pre-created environment object directly
        env=env,
        # max_grad_norm=1.0,
    )

    print("Training completed!")
    return model


if __name__ == "__main__":
    test_rnd_ppo_frozenlake()
