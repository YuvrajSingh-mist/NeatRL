import torch.nn as nn

# ── config ────────────────────────────────────────────────────────────
ENV_ID   = "CliffWalking-v0"
USE_CNN  = False   # True → uses rnd_cnn (Atari, etc.)
GRID_ENV = True    # True for discrete obs (CliffWalking, FrozenLake)
# ──────────────────────────────────────────────────────────────────────


class PolicyMLP(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.Tanh(),
            nn.Linear(256, 256),     nn.Tanh(),
        )
        self.actor  = nn.Linear(256, act_dim)
        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)


class PredictorMLP(nn.Module):
    def __init__(self, obs_dim, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, embed_dim),
        )

    def forward(self, x):
        return self.net(x)


def run(env_id=None, use_cnn=None):
    _cnn = USE_CNN if use_cnn is None else use_cnn
    if _cnn:
        from neatrl.rnd_cnn import train_ppo_rnd_cnn
        train_ppo_rnd_cnn(env_id=env_id or ENV_ID, capture_video=False)
    else:
        from neatrl.rnd_mlp import train_ppo_rnd
        train_ppo_rnd(
            env_id=env_id or ENV_ID,
            grid_env=GRID_ENV,
            use_wandb=False,
            capture_video=False,
        )


if __name__ == "__main__":
    run()
