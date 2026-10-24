# ── config ────────────────────────────────────────────────────────────
ENV_ID = "FrozenLake-v1"
# ──────────────────────────────────────────────────────────────────────


def run(env_id=None, use_cnn=None):
    from neatrl.q_learning import train_q_learning

    # Tabular Q-learning: Discrete obs + Discrete actions only (no CNN).
    _ = use_cnn
    train_q_learning(
        env_id=env_id or ENV_ID,
        capture_video=False,
        use_wandb=False,
    )


if __name__ == "__main__":
    run()
