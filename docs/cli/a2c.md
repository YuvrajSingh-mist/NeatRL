# A2C — `neatrl train a2c`

Advantage Actor-Critic: an actor picks actions, a critic scores them, and the
advantage `A = r + γV(s') − V(s)` gives the actor a lower-variance signal than
the raw return.

| | |
|---|---|
| Aliases | `a2c` · `a2c-cnn` |
| Default env | `CartPole-v1` · `ALE/Pong-v5` |
| Observation | flat vector · stacked Atari frames |
| Trainer | `neatrl.a2c_mlp.train_a2c` · `neatrl.a2c_cnn.train_a2c_cnn` |
| Extra needed | none · `neatrl[atari]` |
| Docs | [`docs/non-cli/A2C/`](../non-cli/A2C/) |

## Run it

```bash
neatrl train a2c                        # CartPole-v1
neatrl train a2c CartPole-v1 --steps 200000
neatrl train a2c LunarLander-v3
neatrl train a2c-cnn                    # ALE/Pong-v5
neatrl train a2c-cnn ALE/Breakout-v5 --steps 1000000
```

## What the alias passes

`a2c` calls the trainer with `grid_env=False`, `capture_video=False` and
`use_wandb=False`; the `grid_env` switch exists for the tabular-style grid
observation, and `False` is the ordinary continuous-vector case. `a2c-cnn`
passes `capture_video=False` and `use_wandb=False` with Atari preprocessing.

`--steps` sets `total_timesteps` (a2c counts transitions, not episodes).

## Environments

Discrete-action environments. Atari on `a2c-cnn`; naming `ALE/Pong-v5` on `a2c`
switches to the CNN trainer with a note on stderr.

Full reference: [CLI](README.md) · [algorithm](../non-cli/A2C/README.md)
