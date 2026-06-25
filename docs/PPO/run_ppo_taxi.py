"""
script for PPO training on Taxi using neatrl library.
"""

from neatrl.ppo import train_ppo


def test_ppo_taxi():
    """Test PPO training on Taxi-v3."""
    print("Testing PPO training on Taxi-v3 with neatrl...")

    # Train PPO on Taxi
    model = train_ppo(
        env_id="Taxi-v3",
        total_timesteps=2000000,
        seed=42,
        lr=3e-4,
        gamma=0.99,
        GAE=0.95,
        n_envs=8,
        max_steps=512,
        num_minibatches=4,
        PPO_EPOCHS=4,
        clip_value=0.2,
        VALUE_COEFF=0.5,
        ENTROPY_COEFF=0.01,
        anneal_lr=True,
        capture_video=True,
        use_wandb=True,
        wandb_project="cleanRL",
        exp_name="PPO-Taxi",
        eval_every=1000,
        save_every=1000,
        num_eval_episodes=2,
        log_gradients=True,
        grid_env=True,
        max_grad_norm=0.5,
        value_clip=True,
    )

    print("Training completed!")
    return model


if __name__ == "__main__":
    test_ppo_taxi()
