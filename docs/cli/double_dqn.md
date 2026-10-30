# Double DQN — `neatrl train double-dqn`

DQN with decoupled action selection and evaluation: the online network picks
the next action, the target network scores it. Removes the max-operator
overestimation that plain DQN inherits from its own target.

| | |
|---|---|
| Aliases | `double-dqn` · `double-dqn-cnn` |
| Default env | `CartPole-v1` · `ALE/Pong-v5` |
| Observation | flat vector · stacked Atari frames |
| Trainer | `neatrl.double_dqn_mlp.train_double_dqn` · `neatrl.double_dqn_cnn.train_double_dqn_cnn` |
| Extra needed | none · `neatrl[atari]` |
| Docs | [`docs/non-cli/`](../non-cli/) |

## Run it

```bash
neatrl train double-dqn                          # CartPole-v1
neatrl train double-dqn CartPole-v1 --steps 200000
neatrl train double-dqn-cnn                      # ALE/Pong-v5
neatrl train double-dqn-cnn ALE/Breakout-v5 --steps 500000
```

## What the alias passes

`double-dqn`: `atari_wrapper=False`, `capture_video=False`, `use_wandb=False`.
`double-dqn-cnn`: `capture_video=False`, `use_wandb=False`, plus its own Atari
preprocessing. `--steps` sets `total_timesteps` for both.

## Environments

Same reach as DQN: any vector-observation environment on the plain alias, Atari
on the `-cnn` alias. An image environment named on `double-dqn` switches to
`double-dqn-cnn` automatically, with a note on stderr.

## Programmatic equivalent

```python
from neatrl.double_dqn_mlp import train_double_dqn

model = train_double_dqn(
    env_id="CartPole-v1",
    total_timesteps=200_000,
    atari_wrapper=False,
    capture_video=False,
    use_wandb=False,
)
```

Full reference: [CLI](README.md) · [DQN](dqn.md)
