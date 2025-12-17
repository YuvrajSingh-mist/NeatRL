"""
script for DQN training on Atari-Breakout using neatrl library.
"""

import torch
import torch.nn as nn

from neatrl import train_dqn


class QNet(nn.Module):
    def __init__(self, state_space, action_space):
        print(f"State space: {state_space}, Action space: {action_space}")
        self.conv1 = nn.Conv2d(state_space, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 512)
        self.q_value = nn.Linear(512, action_space)

    def forward(self, x):
        return self.q_value(
            self.fc2(
                torch.relu(
                    self.fc1(
                        self.flatten(
                            torch.relu(
                                self.conv3(
                                    torch.relu(self.conv2(torch.relu(self.conv1(x))))
                                )
                            )
                        )
                    )
                )
            )
        )


def test_dqn_atari():
    """Test DQN training on Breakout."""
    print("Testing DQN training on Breakout with neatrl...")

    # Train DQN on Breakout
    model = train_dqn(
        env_id="BreakoutNoFrameskip-v4",
        total_timesteps=1000000,
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
        train_frequency=4,
        capture_video=True,
        use_wandb=True,
        wandb_project="cleanRL",
        wandb_entity="",
        exp_name="DQN-Atari-Test",
        custom_agent=QNet(
            4, 4
        ),  # Atari state (4 stacked frames) and action dimensions (4 actions for Breakout)
        atari_wrapper=True,
        n_envs=4,
        eval_every=5000,
        device="cpu",
    )

    print(f"Training completed! Model type: {type(model)}")
    print(f"Model device: {next(model.parameters()).device}")

    # Test model inference
    print("Testing model inference...")
    test_obs = torch.randn(1, 4, 84, 84)  # Atari observation shape after preprocessing
    with torch.no_grad():
        q_values = model(test_obs)
        action = q_values.argmax().item()

    print(f"Q-values shape: {q_values.shape}")
    print(f"Selected action: {action}")
    print("Test completed successfully!")

    return model


if __name__ == "__main__":
    test_dqn_atari()
