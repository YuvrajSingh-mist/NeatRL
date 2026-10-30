"""Train Double DQN on CartPole with the MLP trainer.

The smallest complete run: vector observations, a 4 -> 2 action head, and the
decoupled target. ``use_wandb=False`` keeps it fully local; pass ``True`` to log.
"""

from neatrl.double_dqn_mlp import train_double_dqn


def main():
    print("Training Double DQN on CartPole-v1…")
    model = train_double_dqn(
        env_id="CartPole-v1",
        total_timesteps=200_000,
        seed=42,
        lr=2.5e-4,
        buffer_size=10_000,
        gamma=0.99,
        tau=1.0,
        target_network_frequency=50,
        batch_size=128,
        start_e=1.0,
        end_e=0.05,
        exploration_fraction=0.5,
        learning_starts=1_000,
        train_frequency=10,
        eval_every=5_000,
        save_every=20_000,
        use_wandb=False,
        capture_video=False,
        exp_name="DoubleDQN-CartPole",
    )
    print(f"Done. Model: {type(model).__name__}")


if __name__ == "__main__":
    main()
