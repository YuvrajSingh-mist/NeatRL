import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

# ── config ────────────────────────────────────────────────────────────
ENV_ID = "CartPole-v1"
USE_CNN = False  # True → uses ppo_cnn (CarRacing, Atari, etc.)
# ──────────────────────────────────────────────────────────────────────


class ActorCriticMLP(nn.Module):
    def __init__(self, obs_dim, act_dim, continuous=False):
        super().__init__()
        self.continuous = continuous
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )
        self.actor = nn.Linear(256, act_dim)
        self.critic = nn.Linear(256, 1)
        if continuous:
            self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        h = self.shared(x)
        if self.continuous:
            return Normal(self.actor(h), self.log_std.exp()), self.critic(h)
        return Categorical(logits=self.actor(h)), self.critic(h)


def run(env_id=None, use_cnn=None):
    _cnn = USE_CNN if use_cnn is None else use_cnn
    if _cnn:
        from neatrl.ppo_cnn import train_ppo_cnn

        train_ppo_cnn(env_id=env_id or ENV_ID, capture_video=False)
    else:
        from neatrl.ppo_mlp import train_ppo

        train_ppo(env_id=env_id or ENV_ID, capture_video=False)


if __name__ == "__main__":
    run()
