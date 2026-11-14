# A2C — `neatrl train a2c`

Actor-Critic: an actor picks actions, a critic scores states. By default the
actor is weighted by the Monte Carlo advantage `A = G − V(s)`; the Python API
also accepts `advantage=False` to weight by the raw return `G` instead (no
baseline). The CLI always uses the default (`advantage=True`).

Discrete and continuous action spaces both work: the built-in actor picks a
categorical head for `Discrete` and a Normal head for `Box`, same as PPO.
`a2c-cnn` on `CarRacing-v3` is continuous; `a2c` on `LunarLander-v3` is
discrete (`pip install "neatrl[box2d]"`). Atari needs `pip install "neatrl[atari]"`.

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
pip install neatrl

neatrl train a2c                        # CartPole-v1
neatrl train a2c CartPole-v1 --steps 200000
neatrl train a2c LunarLander-v3
neatrl train a2c-cnn                    # ALE/Pong-v5
neatrl train a2c-cnn ALE/Breakout-v5 --steps 1000000
```

Full reference: [CLI](README.md) · [algorithm](../non-cli/A2C/README.md)
