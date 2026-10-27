import torch.nn as nn

# ── config ────────────────────────────────────────────────────────────
ENV_ID = "CartPole-v1"
ATARI_ID = "ALE/Pong-v5"  # used when USE_CNN is True
USE_CNN = False  # True → Atari DQN-CNN (frame stack, clipped rewards)
TOTAL_TIMESTEPS = 1_000_000
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


def run(env_id=None, use_cnn=None):
    _cnn = USE_CNN if use_cnn is None else use_cnn

    if _cnn:
        from neatrl.dqn_cnn import train_dqn_cnn

        train_dqn_cnn(
            env_id=env_id or ATARI_ID,
            total_timesteps=TOTAL_TIMESTEPS,
            capture_video=False,
        )
    else:
        from neatrl.dqn_mlp import train_dqn

        train_dqn(
            env_id=env_id or ENV_ID,
            total_timesteps=TOTAL_TIMESTEPS,
            atari_wrapper=False,
            capture_video=False,
        )


if __name__ == "__main__":
    run()
