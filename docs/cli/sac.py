import torch
import torch.nn as nn

# ── config ────────────────────────────────────────────────────────────
ENV_ID = "Pendulum-v1"
USE_CNN = False  # True → uses sac_cnn (CarRacing, etc.)
# ──────────────────────────────────────────────────────────────────────


class ActorMLP(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.mean = nn.Linear(256, act_dim)
        self.log_std = nn.Linear(256, act_dim)

    def forward(self, x):
        h = self.shared(x)
        return self.mean(h), self.log_std(h).clamp(-5, 2)


class CriticMLP(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))


def run(env_id=None, use_cnn=None):
    _cnn = USE_CNN if use_cnn is None else use_cnn
    if _cnn:
        from neatrl.sac_cnn import train_sac_cnn

        train_sac_cnn(env_id=env_id or ENV_ID, capture_video=False)
    else:
        from neatrl.sac_mlp import train_sac

        train_sac(env_id=env_id or ENV_ID, capture_video=False)


if __name__ == "__main__":
    run()
