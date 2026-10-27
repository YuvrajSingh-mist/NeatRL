# ── config ────────────────────────────────────────────────────────────
ENV_ID = "CartPole-v1"
ATARI_ID = "ALE/Pong-v5"  # used when USE_CNN is True
USE_CNN = False  # True → Atari Double DQN-CNN (frame stack, clipped rewards)
TOTAL_TIMESTEPS = 1_000_000
# ──────────────────────────────────────────────────────────────────────


def run(env_id=None, use_cnn=None):
    _cnn = USE_CNN if use_cnn is None else use_cnn

    if _cnn:
        from neatrl.double_dqn_cnn import train_double_dqn_cnn

        train_double_dqn_cnn(
            env_id=env_id or ATARI_ID,
            total_timesteps=TOTAL_TIMESTEPS,
            capture_video=False,
        )
    else:
        from neatrl.double_dqn_mlp import train_double_dqn

        train_double_dqn(
            env_id=env_id or ENV_ID,
            total_timesteps=TOTAL_TIMESTEPS,
            atari_wrapper=False,
            capture_video=False,
        )


if __name__ == "__main__":
    run()
