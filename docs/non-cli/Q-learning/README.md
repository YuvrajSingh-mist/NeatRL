# Q-learning

Tabular off-policy TD control: no network, no replay buffer, just a table of
state-action values updated from each transition. The foundation DQN later
approximates with a neural network.

Use it only for environments whose observation is a **single integer with a
modest number of values** — `FrozenLake-v1`, `CliffWalking-v1`, `Taxi-v4`.
Anything with images or continuous observations has no table to index; use DQN
or its CNN sibling instead. There is no `q-learning-cnn` alias for that reason.

| | |
|---|---|
| CLI alias | `q-learning` (no CNN sibling) |
| Default env | `FrozenLake-v1` |
| Observation | discrete scalar, small state space |
| Trainer | `neatrl.q_learning.train_q_learning` |
| Extra needed | none |

## Quick start

### CLI

```bash
neatrl train q-learning                            # FrozenLake-v1
neatrl train q-learning FrozenLake-v1 --steps 20000
neatrl train q-learning Taxi-v4
```

Omit the environment to use the alias default; `--steps` sets
`total_timesteps` — the number of environment steps, which is also the number
of table updates. `--wandb` turns on logging (off by default). The full flag
reference is [`docs/cli/q_learning.md`](../../cli/q_learning.md).

### Python API

```python
from neatrl.q_learning import train_q_learning

q_table = train_q_learning(
    env_id="FrozenLake-v1",
    total_timesteps=20_000,
    seed=42,
    use_wandb=False,  # the function default is True
)
```

Unlike the deep trainers, this returns the **Q-table** (a numpy array of shape
`[n_states, n_actions]`), not an `nn.Module`. The final table is also written to
`runs/<env>__Q-learning__<seed>__<timestamp>/models/q_table_final.npy`.

## Function arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `env_id` | str | `"FrozenLake-v1"` | Gymnasium environment ID |
| `env` | gym.Env | `None` | Pre-built environment, instead of `env_id` |
| `total_timesteps` | int | `50000` | Environment steps (and table updates) |
| `seed` | int | `42` | Random seed |
| `lr` | float | `0.1` | Learning rate for the TD update |
| `gamma` | float | `0.99` | Discount factor |
| `start_e` / `end_e` | float | `1.0` / `0.05` | Epsilon schedule bounds |
| `exploration_fraction` | float | `0.5` | Fraction of training spent decaying epsilon |
| `capture_video` | bool | `False` | Record evaluation episodes |
| `use_wandb` | bool | `True` | Log to Weights & Biases |
| `eval_every` / `save_every` | int | `2000` / `5000` | Evaluation and checkpoint cadence |
| `num_eval_eps` | int | `10` | Evaluation episodes per checkpoint |

## The update

```
Q(S, A) <- Q(S, A) + lr * (R + gamma * max_a Q(S', a) - Q(S, A))
```

The bootstrap term is dropped when the episode ends, and actions come from an
epsilon-greedy policy with `epsilon` annealed over `exploration_fraction` of
training. Because the target uses `max`, this is Sarsamax Q-learning — the same
target DQN approximates.

## Example script

| Script | What it trains |
|--------|----------------|
| `run_q_learning_frozenlake.py` | FrozenLake-v1, 50k steps |

```bash
python docs/non-cli/Q-learning/run_q_learning_frozenlake.py
```

## Installation

```bash
pip install neatrl                 # FrozenLake, CliffWalking, Taxi (classic-control)
```

## See also

- CLI reference: [`docs/cli/q_learning.md`](../../cli/q_learning.md)
- The deep version of the same update: [`docs/non-cli/DQN/`](../DQN/)
- [NeatRL on PyPI](https://pypi.org/project/neatrl/)
