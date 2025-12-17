"""
script for REINFORCE training on Atari Breakout using neatrl library.
"""

from neatrl import train_reinforce


def test_reinforce_breakout():
    """Test REINFORCE training on Atari Breakout."""
    print("Testing REINFORCE training on BreakoutNoFrameskip-v4 with neatrl...")

    # Train REINFORCE on Atari Breakout with CNN
    model = train_reinforce(
        env_id="BreakoutNoFrameskip-v4",
        episodes=2000,
        seed=42,
        learning_rate=2.5e-4,
        gamma=0.99,
        capture_video=False,
        use_wandb=False,
        wandb_project="cleanRL",
        wandb_entity="",
        exp_name="REINFORCE-Breakout",
        eval_every=100,
        save_every=1000,
        atari_wrapper=True,  # Enable Atari preprocessing
        n_envs=4,  # Use 4 parallel environments
        num_eval_eps=10,
        device="cpu",
        grid_env=False,
    )

    print("REINFORCE training on BreakoutNoFrameskip-v4 completed successfully!")
    return model


if __name__ == "__main__":
    test_reinforce_breakout()
