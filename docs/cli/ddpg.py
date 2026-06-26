import torch
import torch.nn as nn

# ── config ────────────────────────────────────────────────────────────
ENV_ID  = "Pendulum-v1"
USE_CNN = False   # True → uses ddpg_cnn (CarRacing, etc.)
# ──────────────────────────────────────────────────────────────────────


class ActorMLP(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.Mish(),
            nn.Linear(256, 256),     nn.Mish(),
            nn.Linear(256, act_dim), nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class CriticMLP(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256), nn.Mish(),
            nn.Linear(256, 256),               nn.Mish(),
            nn.Linear(256, 1),
        )

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))


def run(env_id=None, use_cnn=None):
    _cnn = USE_CNN if use_cnn is None else use_cnn
    if _cnn:
        from neatrl.ddpg_cnn import train_ddpg_cnn
        train_ddpg_cnn(env_id=env_id or ENV_ID, capture_video=False)
    else:
        from neatrl.ddpg_mlp import train_ddpg
        train_ddpg(env_id=env_id or ENV_ID, capture_video=False)


if __name__ == "__main__":
    run()
