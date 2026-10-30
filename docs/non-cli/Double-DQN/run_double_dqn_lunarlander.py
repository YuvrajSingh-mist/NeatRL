"""Train Double DQN on LunarLander with the MLP trainer.

LunarLander's reward is sparse-ish and its Q values are large, so the
overestimation that Double DQN fixes is visible here well before CartPole's
values start to matter. Install the Box2D extra first:

    pip install "neatrl[box2d]"
"""

from neatrl.double_dqn_mlp import train_double_dqn


def main():
    print("Training Double DQN on LunarLander-v3…")
    model = train_double_dqn(
        env_id="LunarLander-v3",
        total_timesteps=300_000,
        seed=42,
        lr=2.5e-4,
        buffer_size=50_000,
        gamma=0.99,
        tau=1.0,
        target_network_frequency=50,
        batch_size=128,
        start_e=1.0,
        end_e=0.05,
        exploration_fraction=0.5,
        learning_starts=1_000,
        train_frequency=10,
        eval_every=10_000,
        save_every=50_000,
        use_wandb=False,
        capture_video=False,
        exp_name="DoubleDQN-LunarLander",
    )
    print(f"Done. Model: {type(model).__name__}")


if __name__ == "__main__":
    main()
