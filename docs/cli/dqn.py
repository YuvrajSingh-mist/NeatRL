import torch.nn as nn

# ── config ────────────────────────────────────────────────────────────
ENV_ID = "CartPole-v1"
USE_CNN = False  # True → Atari preprocessing (e.g. BreakoutNoFrameskip-v4)
# ──────────────────────────────────────────────────────────────────────


class QNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
        )

    def forward(self, x):
        return self.net(x)


class QNetCNN(nn.Module):
    def __init__(self, obs_shape, act_dim):
        super().__init__()
        c = obs_shape[0]
        self.conv = nn.Sequential(
            nn.Conv2d(c, 32, 8, 4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, act_dim),
        )

    def forward(self, x):
        return self.fc(self.conv(x).flatten(1))


def run(env_id=None, use_cnn=None):
    from neatrl.dqn_mlp import train_dqn

    _cnn = USE_CNN if use_cnn is None else use_cnn
    train_dqn(env_id=env_id or ENV_ID, atari_wrapper=_cnn, capture_video=False)


if __name__ == "__main__":
    run()
