"""
script for RND-PPO training on FrozenLake using neatrl library.
"""

from neatrl import train_rnd


def test_rnd_ppo_frozenlake():
    """Test RND-PPO training on FrozenLake-v1."""
    print("Testing RND-PPO training on FrozenLake-v1 with neatrl...")

    # Train RND-PPO on FrozenLake
    model = train_rnd(
        env_id="FrozenLake-v1",
        total_timesteps=20000000,
        seed=42,
        lr=1e-4,
        ext_gamma=0.99,
        int_gamma=0.99,
        n_envs=1,
        max_steps=128,
        num_minibatches=4,
        PPO_EPOCHS=4,
        clip_value=0.2,
        ENTROPY_COEFF=0.01,
        VALUE_COEFF=0.5,
        EXT_COEFF=2.0,
        INT_COEFF=1.0,
        capture_video=True,
        use_wandb=True,
        wandb_project="cleanRL",
        exp_name="RND-PPO-FrozenLake",
        grid_env=True,
        eval_every=100,
        save_every=1000,
        num_eval_episodes=5,
        anneal_lr=True,
        # max_grad_norm=1.0,
    )

    print("Training completed!")
    return model


if __name__ == "__main__":
    test_rnd_ppo_frozenlake()
