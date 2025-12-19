"""
script for RND-PPO training on CarRacing using neatrl library.
"""

import torch
import torch.nn as nn

from neatrl.rnd import train_rnd


def test_rnd_ppo_carracing():
    """Test RND-PPO training on CarRacing-v3."""
    print("Testing RND-PPO training on CarRacing-v3 with neatrl...")

    # Train RND-PPO on CarRacing
    model = train_rnd(
        env_id="CarRacing-v3",
        total_timesteps=500000,
        seed=42,
        lr=3e-4,
        ext_gamma=0.99,
        int_gamma=0.99,
        num_envs=4,
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
        exp_name="RND-PPO-CarRacing",
        grid_env=False,
        max_grad_norm=1.0,
    )

    print("Training completed!")
    return model


if __name__ == "__main__":
    test_rnd_ppo_carracing()