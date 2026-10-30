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

Full reference: [CLI](README.md) · [algorithm](../non-cli/A2C/README.md)
