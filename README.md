# NeatRL

A collection of one-file reinforcement learning algorithm implementations, plus a packaged library for drop-in training.

## Repository Layout

```
NeatRL/
  neatrl/          # Installable Python library
  experiments/     # Standalone single-file scripts (one per algorithm/env)
```

## NeatRL Library

The `neatrl/` directory is a pip-installable package that exposes clean `train_*` functions for each algorithm.

```bash
pip install neatrl

# Optional environment extras
pip install neatrl[atari]    # Atari games
pip install neatrl[box2d]    # BipedalWalker, LunarLander
pip install neatrl[classic]  # CartPole, Pendulum, FrozenLake
pip install neatrl[mujoco]   # HalfCheetah, Ant, Humanoid
```

### Supported Algorithms

| Algorithm | Function | Action Space |
|-----------|----------|--------------|
| DQN | `train_dqn` | Discrete |
| Dueling DQN | `train_dueling_dqn` | Discrete |
| REINFORCE | `train_reinforce`, `train_reinforce_cnn` | Discrete / Continuous |
| A2C | `train_a2c`, `train_a2c_cnn` | Discrete / Continuous |
| PPO | `train_ppo`, `train_ppo_cnn` | Discrete / Continuous |
| PPO-RND | `train_ppo_rnd`, `train_ppo_rnd_cnn` | Discrete / Continuous |
| DDPG | `train_ddpg`, `train_ddpg_cnn` | Continuous |
| TD3 | `train_td3`, `train_td3_cnn` | Continuous |
| SAC | `train_sac`, `train_sac_cnn` | Continuous |

### Quick Start

```python
from neatrl import train_dqn

model = train_dqn(
    env_id="CartPole-v1",
    total_timesteps=10000,
    seed=42
)
```

```python
from neatrl import train_ppo

model = train_ppo(
    env_id="LunarLander-v3",
    total_timesteps=500000,
    n_envs=4,
    use_wandb=True,
    seed=42
)
```

```python
from neatrl import train_sac

model = train_sac(
    env_id="HalfCheetah-v5",
    total_timesteps=1000000,
    use_wandb=True,
    seed=42
)
```

Full documentation: [neatrl/README.md](./neatrl/README.md)

## Experiments

The `experiments/` directory contains standalone scripts -- each file is a self-contained implementation written for a specific algorithm or environment before it was absorbed into the library. Useful as reference or for one-off runs.

```
experiments/
  DQN/              CartPole, LunarLander
  DQN-atari/        Atari with CNN
  DQN-FrozenLake/   FrozenLake with one-hot encoding
  DQN-Lunar/        LunarLander-specific tuning
  DQN-Taxi/         Taxi-v3
  DQN-flappy/       FlappyBird
  Duel-DQN/         Dueling DQN on CliffWalking
  Q-Learning/       Tabular Q-learning
  REINFORCE/        CartPole, Pendulum
  A2C/              Acrobot, LunarLander, CarRacing
  PPO/              LunarLander, BipedalWalker, Atari
  FlappyBird-PPO/   PPO on FlappyBird
  DDPG/             HalfCheetah, BipedalWalker
  TD3/              HalfCheetah, BipedalWalker, Atari
  SAC/              HalfCheetah, BipedalWalker, discrete
  RND/              CliffWalking, CarRacing (curiosity-driven)
  MARL/             Multi-agent RL
  GRPO/             Group Relative Policy Optimization
  Imitation Learning/  Behavioral cloning
  VizDoom-RL/       3D first-person environments
```

## License

MIT
