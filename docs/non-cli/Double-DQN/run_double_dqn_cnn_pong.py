"""Train the Atari Double DQN-CNN on Pong.

Same preprocessing and hyperparameters as ``run_dqn_cnn_pong.py``; the only
difference is the target ``R + gamma * Q_target(s', argmax_a Q_online(s', a))``.

Watch ``values/mean_max_q`` against ``values/true_discounted_return``: with the
decoupled target the estimate stays much closer to what the greedy policy
actually earns.
"""

from neatrl.double_dqn_cnn import train_double_dqn_cnn

if __name__ == "__main__":
    model = train_double_dqn_cnn(
        env_id="ALE/Pong-v5",
        total_timesteps=1_000_000,  # paper: 50M agent steps (200M frames)
        seed=42,
        lr=1e-4,
        buffer_size=50_000,
        gamma=0.99,
        tau=1.0,  # hard target copies
        target_network_frequency=10_000,
        batch_size=32,
        start_e=1.0,
        end_e=0.1,
        exploration_fraction=0.1,
        learning_starts=20_000,
        train_frequency=4,
        max_grad_norm=10.0,
        eval_every=50_000,
        save_every=200_000,
        num_eval_eps=3,
        use_wandb=True,
        wandb_project="cleanRL",
        exp_name="DoubleDQN-CNN-Pong",
        capture_video=False,
        device="cpu",
    )
    print(f"Training finished. Model: {type(model).__name__}")
