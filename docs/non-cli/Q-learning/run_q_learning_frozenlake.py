"""Script for tabular Q-learning on FrozenLake using neatrl."""

from neatrl.q_learning import train_q_learning


def main():
    print("Training tabular Q-learning on FrozenLake-v1…")
    q_table = train_q_learning(
        env_id="FrozenLake-v1",
        total_timesteps=50_000,
        seed=42,
        lr=0.1,
        gamma=0.99,
        start_e=1.0,
        end_e=0.05,
        exploration_fraction=0.5,
        use_wandb=False,
        exp_name="Q-learning-FrozenLake",
        eval_every=5_000,
        save_every=10_000,
    )
    print(f"Done. Q-table shape: {q_table.shape}")


if __name__ == "__main__":
    main()
