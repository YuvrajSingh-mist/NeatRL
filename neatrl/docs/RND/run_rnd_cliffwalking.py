"""
script for RND-PPO training on CliffWalking using neatrl library.
"""

from neatrl import train_rnd


def test_rnd_ppo_cliffwalking():
    """Test RND-PPO training on CliffWalking-v0."""
    print("Testing RND-PPO training on CliffWalking-v0 with neatrl...")

    # Train RND-PPO on CliffWalking
    model = train_rnd(
        env_id="CliffWalking-v1",
        total_timesteps=5000000,
        seed=42,
        lr=3e-4,
        ext_gamma=0.99,
        int_gamma=0.95,
        n_envs=4,
        max_steps=128,
        num_minibatches=4,
        PPO_EPOCHS=4,
        clip_value=0.2,
        ENTROPY_COEFF=0.01,
        VALUE_COEFF=0.5,
        EXT_COEFF=0.5,
        INT_COEFF=1.0,
        capture_video=True,
        use_wandb=True,
        wandb_project="cleanRL",
        exp_name="RND-PPO-Cliffwalking",
        grid_env=True,
        eval_every=1000,
        save_every=1000,
        num_eval_episodes=1,
        anneal_lr=True,
        log_gradients=True,
    )

    print("Training completed!")
    return model


if __name__ == "__main__":
    test_rnd_ppo_cliffwalking()
