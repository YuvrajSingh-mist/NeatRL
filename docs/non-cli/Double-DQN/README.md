# Double DQN

Double DQN is DQN with one line of the target changed: the **online** network
picks the next action and the **target** network scores it. Plain DQN asks the
target network to both choose and evaluate the next action, and because both
sides of that `max` come from the same noisy estimator, the target is biased
upward. Decoupling them removes most of the overestimation.

| | |
|---|---|
| CLI aliases | `double-dqn` · `double-dqn-cnn` |
| Default env | `CartPole-v1` · `ALE/Pong-v5` |
| Observation | flat vector · stacked Atari frames |
| Trainer | `neatrl.double_dqn_mlp.train_double_dqn` · `neatrl.double_dqn_cnn.train_double_dqn_cnn` |
| Extra needed | none · `neatrl[atari]` |

## The target

```
DQN        : Y = r + gamma * max_a Q_target(s', a)
Double DQN : Y = r + gamma * Q_target(s', argmax_a Q_online(s', a))
```

Everything else — replay buffer, epsilon schedule, hard target copies — is
identical to DQN, so the two trainers are drop-in replacements for each other.
The derivation and the Atari evidence are in
[van Hasselt, Guez & Silver (2016)](https://www.alphaxiv.org/abs/1509.06461).

## Quick start

### CLI

```bash
neatrl train double-dqn CartPole-v1
neatrl train double-dqn LunarLander-v3 --steps 300000
neatrl train double-dqn-cnn ALE/Pong-v5
```

Omit the environment to use the alias default; `--steps` sets
`total_timesteps`, and `--wandb` turns on logging (off by default). Naming an
image environment on the plain alias switches to the CNN trainer
automatically. The full flag reference is
[`docs/cli/double_dqn.md`](../../cli/double_dqn.md).

### Python API

```python
from neatrl.double_dqn_mlp import train_double_dqn

model = train_double_dqn(
    env_id="CartPole-v1",
    total_timesteps=200_000,
    use_wandb=False,  # the function default is True
)
```

```python
from neatrl.double_dqn_cnn import train_double_dqn_cnn

model = train_double_dqn_cnn(
    env_id="ALE/Pong-v5",
    total_timesteps=1_000_000,
    use_wandb=False,
)
```

## Function arguments — `train_double_dqn`

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `env_id` | str | `None` | Gymnasium environment ID |
| `env` | gym.Env | `None` | Pre-built environment, instead of `env_id` |
| `total_timesteps` | int | `50000` | Environment steps to train for |
| `seed` | int | `42` | Random seed |
| `lr` | float | `2.5e-4` | Adam learning rate |
| `buffer_size` | int | `10000` | Replay buffer capacity |
| `gamma` | float | `0.99` | Discount factor |
| `tau` | float | `1.0` | Target update rate (1.0 = hard copy) |
| `target_network_frequency` | int | `50` | Steps between target copies |
| `batch_size` | int | `128` | Replay batch size |
| `start_e` / `end_e` | float | `1.0` / `0.05` | Epsilon schedule bounds |
| `exploration_fraction` | float | `0.5` | Fraction of training spent decaying epsilon |
| `learning_starts` | int | `1000` | Steps collected before the first update |
| `train_frequency` | int | `10` | Environment steps per gradient step |
| `max_grad_norm` | float | `0.5` | Gradient clipping (0 disables) |
| `capture_video` | bool | `False` | Record evaluation episodes |
| `use_wandb` | bool | `True` | Log to Weights & Biases |
| `eval_every` / `save_every` | int | `1000` / `10000` | Evaluation and checkpoint cadence |
| `n_envs` | int | `1` | Parallel environments |
| `device` | str | `"cpu"` | `"cpu"`, `"cuda"` or `"mps"` |

`train_double_dqn_cnn` takes the same replay and optimiser settings, but its
defaults are the Atari ones (`lr=1e-4`, `buffer_size=100000`,
`target_network_frequency=10000`, `learning_starts=20000`, `batch_size=32`,
`end_e=0.1`, `max_grad_norm=10.0`) and it adds `frame_stack=4`,
`screen_size=84` and `clip_reward=True`.

## Example scripts

| Script | What it trains |
|--------|----------------|
| `run_double_dqn_cartpole.py` | CartPole-v1, MLP |
| `run_double_dqn_lunarlander.py` | LunarLander-v3, MLP (`neatrl[box2d]`) |
| `run_double_dqn_cnn_pong.py` | ALE/Pong-v5, CNN (`neatrl[atari]`) |

```bash
python docs/non-cli/Double-DQN/run_double_dqn_cartpole.py
```

## Seeing the overestimation gap

`docs/non-cli/DQN/overestimation_demo.py` trains DQN and Double DQN with
identical settings and plots `max_a Q(s, a)` against the return the greedy
policy actually earns. With an unbiased critic the curves coincide; DQN's
estimate rides above what the policy collects, and Double DQN keeps the two
closer. Both trainers are selected automatically by environment id.

```bash
python docs/non-cli/DQN/overestimation_demo.py --steps 5000
```

## Installation

```bash
pip install neatrl                 # CartPole and other classic-control envs
pip install neatrl[box2d]          # LunarLander
pip install neatrl[atari]          # Pong, Breakout, any ALE environment
```

## See also

- CLI reference: [`docs/cli/double_dqn.md`](../../cli/double_dqn.md)
- DQN notes and the full argument table: [`docs/non-cli/DQN/`](../DQN/)
- [NeatRL on PyPI](https://pypi.org/project/neatrl/)
