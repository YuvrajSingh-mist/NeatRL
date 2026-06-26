import torch.nn as nn

# ── config ────────────────────────────────────────────────────────────
ENV_ID = "CliffWalking-v0"
USE_CNN = False  # True → Atari preprocessing
GRID_ENV = True  # True for discrete obs (CliffWalking, FrozenLake)
# ──────────────────────────────────────────────────────────────────────


class DuelingQNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.value = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.adv = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, act_dim))

    def forward(self, x):
        feat = self.features(x)
        v, a = self.value(feat), self.adv(feat)
        return v + a - a.mean(dim=1, keepdim=True)


def run(env_id=None, use_cnn=None):
    from neatrl.dueling_dqn_mlp import train_dueling_dqn

    _cnn = USE_CNN if use_cnn is None else use_cnn
    train_dueling_dqn(
        env_id=env_id or ENV_ID,
        atari_wrapper=_cnn,
        grid_env=GRID_ENV,
        capture_video=False,
    )


if __name__ == "__main__":
    run()
