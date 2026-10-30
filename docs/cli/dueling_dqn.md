# Dueling DQN — `neatrl train dueling-dqn`

Splits the network head into a state value `V(s)` and an advantage `A(s,a)`, so
the value of a state is learned without evaluating every action from it.

| | |
|---|---|
| Aliases | `dueling-dqn` · `dueling-dqn-cnn` |
| Default env | `CliffWalking-v1` · `ALE/Pong-v5` |
| Observation | flat vector · stacked Atari frames |
| Trainer | `neatrl.dueling_dqn_mlp.train_dueling_dqn` (both aliases) |
| Extra needed | none · `neatrl[atari]` |
| Docs | [`docs/non-cli/Dueling-DQN/`](../non-cli/Dueling-DQN/) |

## Run it

```bash
neatrl train dueling-dqn                        # CliffWalking-v1
neatrl train dueling-dqn CartPole-v1 --steps 100000
neatrl train dueling-dqn-cnn                    # ALE/Pong-v5
neatrl train dueling-dqn-cnn ALE/Breakout-v5
```

## What the alias passes

Unlike the rest of the family, `dueling-dqn` has **one trainer** for both
observation types: the `-cnn` alias is the same function with
`atari_wrapper=True`. Everything else is `capture_video=False` and
`use_wandb=False`; `--steps` sets `total_timesteps`.

If you name an image environment on the plain alias, the routing adds
`atari_wrapper=True` for you.

## Environments

`CliffWalking-v1` (the default) is a sparse-reward grid where dueling heads
earn their keep. Any discrete-action vector environment works too. Atari needs
`pip install "neatrl[atari]"`.

Full reference: [CLI](README.md) · [algorithm](../non-cli/Dueling-DQN/README.md)
