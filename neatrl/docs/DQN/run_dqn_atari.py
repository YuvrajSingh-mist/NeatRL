#!/usr/bin/env python3
"""
script for DQN training on Atari-Breakout using neatrl library.
"""

import torch

from neatrl import train_dqn


def test_dqn_breakout():
    """Test DQN training on Atari-Breakout."""
    print("Testing DQN training on Atari-Breakout with neatrl...")

    # Train DQN on Breakout
    model = train_dqn(
        env_id="BreakoutNoFrameskip-v4",
        total_timesteps=100000,
        seed=42,
        learning_rate=2.5e-4,
        buffer_size=50000,
        gamma=0.99,
        tau=1.0,
        target_network_frequency=50,
        batch_size=128,
        start_e=1.0,
        end_e=0.05,
        exploration_fraction=0.5,
        learning_starts=1000,
        train_frequency=10,
        capture_video=True,
        use_wandb=True,
        wandb_project="NeatRL",
        wandb_entity="",
        exp_name="DQN-Atari-Test",
        atari_wrapper=True,
    )

    print(f"Training completed! Model type: {type(model)}")
    print(f"Model device: {next(model.parameters()).device}")

    # Test model inference
    print("Testing model inference...")
    test_obs = torch.randn(1, 4)  # CartPole observation shape
    with torch.no_grad():
        q_values = model(test_obs)
        action = q_values.argmax().item()

    print(f"Q-values shape: {q_values.shape}")
    print(f"Selected action: {action}")
    print("Test completed successfully!")

    return model


if __name__ == "__main__":
    test_dqn_breakout()
