"""
script for PPO training on Bipedal Walker using neatrl library.
"""

from neatrl.ppo import train_ppo


def test_ppo_bipedal_walker():
    """Test PPO training on BipedalWalker-v3."""
    print("Testing PPO training on BipedalWalker-v3 with neatrl...")

    # Train PPO on Bipedal Walker
    model = train_ppo(
        env_id="BipedalWalker-v3",
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
        max_grad_norm=0.5,
        anneal_lr=True,
        capture_video=True,
        use_wandb=True,
        wandb_project="cleanRL",
        exp_name="PPO-BipedalWalker",
        eval_every=5000,
        save_every=5000,
        num_eval_episodes=10,
        log_gradients=False,
    )

    print("Training completed!")
    return model


if __name__ == "__main__":
    test_ppo_bipedal_walker()