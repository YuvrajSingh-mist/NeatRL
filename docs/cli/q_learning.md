# Q-learning — `neatrl train q-learning`

Tabular Q-learning: no network, no replay buffer — a table of state-action
values updated by the off-policy TD rule, with epsilon-greedy exploration.

| | |
|---|---|
| Alias | `q-learning` (no CNN sibling) |
| Default env | `FrozenLake-v1` |
| Observation | discrete scalar (`int(obs)`), small state space |
| Trainer | `neatrl.q_learning.train_q_learning` |
| Extra needed | none |
| Docs | [`docs/non-cli/Q-learning/`](../non-cli/Q-learning/) |

## Run it

```bash
neatrl train q-learning                      # FrozenLake-v1
neatrl train q-learning FrozenLake-v1 --steps 20000
neatrl train q-learning Taxi-v4
```

The run saves the final table as `runs/<env>/.../q_table_final.npy` instead of
a `.pth` model.

## Why there is no `q-learning-cnn`

The alias only supports environments whose observation is a single integer
with a modest number of distinct values, because the "model" is a dict of
tables. An image environment has nothing to index. Use `dqn-cnn` for pixels.

## What the alias passes

`capture_video=False` and `use_wandb=False`. `--steps` sets
`total_timesteps` — the number of environment steps, which is also the number
of table updates.

## Programmatic equivalent

```python
from neatrl.q_learning import train_q_learning

model = train_q_learning(
    env_id="FrozenLake-v1",
    total_timesteps=20_000,
    capture_video=False,
    use_wandb=False,
)
```

Full reference: [CLI](README.md) · [algorithm](../non-cli/Q-learning/README.md)
