import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal

# ── config ────────────────────────────────────────────────────────────
ENV_ID  = "CartPole-v1"
USE_CNN = False   # True → uses reinforce_cnn (CarRacing, Atari, etc.)
# ──────────────────────────────────────────────────────────────────────


class PolicyMLP(nn.Module):
    def __init__(self, obs_dim, act_dim, continuous=False):
        super().__init__()
        self.continuous = continuous
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256),     nn.ReLU(),
            nn.Linear(256, act_dim),
        )
        if continuous:
            self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        out = self.net(x)
        if self.continuous:
            return Normal(out, self.log_std.exp())
        return Categorical(logits=out)


def run(env_id=None, use_cnn=None):
    _cnn = USE_CNN if use_cnn is None else use_cnn
    if _cnn:
        from neatrl.reinforce_cnn import train_reinforce_cnn
        train_reinforce_cnn(env_id=env_id or ENV_ID, capture_video=False)
    else:
        from neatrl.reinforce_mlp import train_reinforce
        train_reinforce(env_id=env_id or ENV_ID, capture_video=False)


if __name__ == "__main__":
    run()
