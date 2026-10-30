# NeatRL Documentation - DQN

This guide shows you how to use NeatRL's DQN implementation.

## Quick Start with DQN

### CLI

```bash
neatrl train dqn CartPole-v1
neatrl train dqn LunarLander-v3
neatrl train dqn MountainCar-v0
neatrl train dqn FrozenLake-v1
neatrl train dqn ALE/Breakout-v5
```

### Python API

```python
from neatrl.dqn_mlp import train_dqn

model = train_dqn(
    env_id="CartPole-v1",
    total_timesteps=100000,
    seed=42,
    use_wandb=False,
)
```

## Function Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `env_id` | str | `"ALE/Breakout-v5"` | Gymnasium environment ID |
| `total_timesteps` | int | `20000` | Total number of environment steps |
| `seed` | int | `42` | Random seed |
| `learning_rate` | float | `2.5e-4` | Learning rate for the Adam optimizer |
| `buffer_size` | int | `10000` | Size of the replay buffer |
| `gamma` | float | `0.99` | Discount factor |
| `tau` | float | `1.0` | Target network update rate (1.0 = hard update) |
| `target_network_frequency` | int | `50` | How often to update the target network |
| `batch_size` | int | `128` | Batch size for training |
| `start_e` | float | `1.0` | Initial epsilon for exploration |
| `end_e` | float | `0.05` | Final epsilon for exploration |
| `exploration_fraction` | float | `0.5` | Fraction of training steps for epsilon decay |
| `learning_starts` | int | `1000` | Steps before training begins |
| `train_frequency` | int | `10` | How often to train the network |
| `max_grad_norm` | float | `0.0` | Maximum gradient norm for clipping (0.0 to disable) |
| `capture_video` | bool | `False` | Whether to record training videos |
| `use_wandb` | bool | `False` | Whether to log to Weights & Biases |
| `wandb_project` | str | `"cleanRL"` | W&B project name |
| `wandb_entity` | str | `""` | W&B username/entity |
| `exp_name` | str | `"DQN"` | Experiment name |
| `eval_every` | int | `1000` | Evaluation frequency (steps) |
| `save_every` | int | `1000` | Model save frequency (steps) |
| `atari_wrapper` | bool | `False` | Whether to apply Atari preprocessing |
| `custom_agent` | nn.Module | `None` | Custom Q-network class/instance |
| `num_eval_eps` | int | `10` | Number of evaluation episodes |
| `n_envs` | int | `4` | Number of parallel environments |
| `device` | str | `"cpu"` | Training device ("cpu", "cuda", "mps") |
| `grid_env` | bool | `False` | Enable one-hot encoding for discrete states |


## Double DQN

Double DQN has its own folder — [`docs/non-cli/Double-DQN/`](../Double-DQN/) —
with the full argument table, example scripts and the CLI reference.

`neatrl.double_dqn_mlp` (vector observations) and `neatrl.double_dqn_cnn` (Atari
frames) are drop-in replacements for `dqn_mlp` / `dqn_cnn` that differ only in
the TD target:

```
DQN        : Y = r + gamma * max_a Q_target(s', a)
Double DQN : Y = r + gamma * Q_target(s', argmax_a Q_online(s', a))
```

```python
from neatrl.double_dqn_mlp import train_double_dqn

model = train_double_dqn(env_id="CartPole-v1", total_timesteps=50000, use_wandb=False)
```

```python
from neatrl.double_dqn_cnn import train_double_dqn_cnn

model = train_double_dqn_cnn(
    env_id="ALE/Pong-v5", total_timesteps=50000, use_wandb=False
)
```

```bash
neatrl train double-dqn CartPole-v1
neatrl train double-dqn-cnn ALE/Pong-v5
```

### Seeing the overestimation gap

`docs/non-cli/DQN/overestimation_demo.py` trains DQN and Double DQN with
identical settings and reproduces the measurement from Figure 3 of
van Hasselt et al. (2016). During greedy evaluation it pairs, for every state
the policy visits:

| quantity | definition |
|---|---|
| predicted | `max_a Q(s_t, a)` |
| realised | `sum_{k >= t} gamma^(k-t) r_k` |

Both are averaged over the *same* states. With an unbiased critic the two
curves coincide; DQN's estimate rides above what the policy earns, while
Double DQN keeps them closer together.

```bash
python docs/non-cli/DQN/overestimation_demo.py                     # CartPole, both
python docs/non-cli/DQN/overestimation_demo.py --algorithms dqn    # DQN only first
python docs/non-cli/DQN/overestimation_demo.py --steps 5000        # quick smoke run
```

`--algorithms` accepts `dqn`, `double_dqn`, or both (run in the order given, so
the default `dqn,double_dqn` shows DQN first and then the Double DQN fix). The
network is picked automatically: Atari ids (`ALE/...`, `...NoFrameskip...`) use
the CNN versions, everything else the MLP ones.

It prints one line per evaluation, writes a CSV, and rewrites
`runs/overestimation/overestimation_<env>_<steps>steps.png` after every
evaluation, so you can watch the curves while training is still running. The
figure has three panels: DQN's estimate vs reality, Double DQN's estimate vs
reality, and both gaps on one axis.

**Pick the right environment.** Overestimation is a bias in the *target*, so it
is only measurable once the agent actually learns; before that, "predicted
minus realised" is dominated by under-fitting. `CartPole-v1` learns to
near-optimal in under a minute and its values are large, so the effect is
obvious. `ALE/Pong-v5` reproduces the paper's setting but needs the paper's
budget (50M frames ≈ 12.5M agent steps, many hours); at 1M steps the agent is
still at random play and the measurement says nothing.


## Installation

```bash
pip install neatrl

pip install neatrl[atari]      # For Atari games
pip install neatrl[box2d]      # For LunarLander
pip install neatrl[classic]    # For CartPole, FrozenLake, MountainCar, Acrobot
```

## PyPI

For installation and more information, visit [NeatRL on PyPI](https://pypi.org/project/neatrl/)
