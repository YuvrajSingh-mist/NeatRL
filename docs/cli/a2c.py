import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

# ── config ────────────────────────────────────────────────────────────
ENV_ID   = "CartPole-v1"
USE_CNN  = False   # True → uses a2c_cnn (CarRacing, Atari, etc.)
GRID_ENV = False   # True for discrete obs (CliffWalking, FrozenLake)
# ──────────────────────────────────────────────────────────────────────


class ActorMLP(nn.Module):
    def __init__(self, obs_dim, act_dim, continuous=False):
        super().__init__()
        self.continuous = continuous
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.Tanh(),
            nn.Linear(256, 256),     nn.Tanh(),
            nn.Linear(256, act_dim),
        )
        if continuous:
            self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        out = self.net(x)
        if self.continuous:
            return Normal(out, self.log_std.exp())
        return Categorical(logits=out)


class CriticMLP(nn.Module):
    def __init__(self, obs_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.Tanh(),
            nn.Linear(256, 256),     nn.Tanh(),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.net(x)


def run(env_id=None, use_cnn=None):
    _cnn = USE_CNN if use_cnn is None else use_cnn
    if _cnn:
        from neatrl.a2c_cnn import train_a2c_cnn
        train_a2c_cnn(env_id=env_id or ENV_ID, capture_video=False)
    else:
        from neatrl.a2c_mlp import train_a2c
        train_a2c(env_id=env_id or ENV_ID, grid_env=GRID_ENV, capture_video=False)


if __name__ == "__main__":
    run()
