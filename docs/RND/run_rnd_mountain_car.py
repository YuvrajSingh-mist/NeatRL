"""
script for RND-PPO training on MountainCar using neatrl library.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neatrl import train_ppo_rnd


# --- Networks ---
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# --- Networks ---
class ActorNet(nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.fc1 = layer_init(nn.Linear(state_space, 512))
        self.fc2 = layer_init(nn.Linear(512, 512))
        self.fc3 = layer_init(nn.Linear(512, 256))
        self.out = layer_init(nn.Linear(256, action_space), std=0.01)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)

    def get_action(self, x, action=None):
        logits = self.forward(x)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)

        return action, log_prob, dist


class CriticNet(nn.Module):
    def __init__(self, state_space):
        super().__init__()
        self.fc1 = layer_init(nn.Linear(state_space, 512))
        self.fc2 = layer_init(nn.Linear(512, 512))
        self.fc3 = layer_init(nn.Linear(512, 256))
        self.value_ext = layer_init(nn.Linear(256, 1), std=1.0)
        self.value_int = layer_init(nn.Linear(256, 1), std=1.0)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.value_ext(x), self.value_int(x)


class PredictorNet(nn.Module):
    def __init__(self, state_space):
        super().__init__()
        self.fc1 = layer_init(nn.Linear(state_space, 512))
        self.fc2 = layer_init(nn.Linear(512, 512))
        self.fc3 = layer_init(nn.Linear(512, 256))
        self.out = layer_init(nn.Linear(256, 256))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)


class TargetNet(nn.Module):
    def __init__(self, state_space):
        super().__init__()
        self.fc1 = layer_init(nn.Linear(state_space, 512))
        self.fc2 = layer_init(nn.Linear(512, 512))
        self.fc3 = layer_init(nn.Linear(512, 256))
        self.out = layer_init(nn.Linear(256, 256))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)


def test_rnd_ppo_mountain_car():
    """Test RND-PPO training on MountainCar-v0."""
    print("Testing RND-PPO training on MountainCar-v0 with neatrl...")

    # Train RND-PPO on MountainCar
    model = train_ppo_rnd(
        env_id="MountainCar-v0",
        total_timesteps=1000000,
        seed=42,
        lr=3e-4,
        ext_gamma=0.999,
        int_gamma=0.99,
        n_envs=8,
        max_steps=128,
        num_minibatches=4,
        PPO_EPOCHS=4,
        clip_value=0.2,
        ENTROPY_COEFF=0.01,
        VALUE_COEFF=0.5,
        EXT_COEFF=1.0,
        INT_COEFF=2.0,
        capture_video=True,
        use_wandb=True,
        wandb_project="cleanRL",
        exp_name="RND-PPO-MountainCar",
        grid_env=False,
        eval_every=100,
        save_every=1000,
        num_eval_episodes=1,
        anneal_lr=True,
        log_gradients=True,
        actor_class=ActorNet,
        critic_class=CriticNet,
        predictor_class=PredictorNet,
        target_class=TargetNet,
    )

    print("Training completed!")
    return model


if __name__ == "__main__":
    test_rnd_ppo_mountain_car()
