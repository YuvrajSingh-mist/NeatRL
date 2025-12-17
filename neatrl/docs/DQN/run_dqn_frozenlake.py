"""
script for DQN training on FrozenLake using neatrl library.
"""

import gymnasium as gym
import torch
import torch.nn as nn

from neatrl import train_dqn


class FrozenLakeQNet(nn.Module):
    """Q-network for FrozenLake with one-hot encoded states."""

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.q_value = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.q_value(x)


def test_dqn_frozenlake():
    """Test DQN training on FrozenLake-v1."""
    print("Testing DQN training on FrozenLake-v1 with neatrl...")

    # Note: FrozenLake has discrete states, automatically one-hot encoded in make_env

    # Train DQN on FrozenLake
    model = train_dqn(
        env_id="FrozenLake-v1",
        total_timesteps=50000,
        seed=42,
        learning_rate=1e-3,
        buffer_size=10000,
        gamma=0.99,
        tau=1.0,
        target_network_frequency=100,
        batch_size=64,
        start_e=1.0,
        end_e=0.01,
        exploration_fraction=0.8,
        learning_starts=1000,
        train_frequency=4,
        capture_video=True,  # FrozenLake is text-based
        use_wandb=True,
        wandb_project="cleanRL",
        wandb_entity="",
        exp_name="DQN-FrozenLake-Test",
        custom_agent=FrozenLakeQNet(16, 4),  # 16 one-hot states, 4 actions
        atari_wrapper=False,
        n_envs=1,
        eval_every=5000,
        grid_env=True,
    )

    print(f"Training completed! Model type: {type(model)}")
    print(f"Model device: {next(model.parameters()).device}")

    # Test model inference
    print("Testing model inference...")

    class OneHotWrapper(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            self.observation_space = gym.spaces.Box(0, 1, (16,), dtype=float)

        def observation(self, obs):
            one_hot = torch.zeros(16)
            one_hot[obs] = 1.0
            return one_hot.numpy()

    env = gym.make("FrozenLake-v1", render_mode="rgb_array")
    env = OneHotWrapper(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    obs, _ = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done and steps < 100:
        with torch.no_grad():
            q_values = model(torch.tensor(obs, dtype=torch.float32).unsqueeze(0))
            action = q_values.argmax().item()
        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

    print(f"Test episode reward: {total_reward}, steps: {steps}")
    print("Test completed successfully!")

    env.close()
    return model


if __name__ == "__main__":
    test_dqn_frozenlake()
