# REINFORCE — `neatrl train reinforce`

The original policy gradient: run whole episodes, weight each log-probability
by its Monte Carlo return, and step uphill on expected reward.

| | |
|---|---|
| Aliases | `reinforce` · `reinforce-cnn` |
| Default env | `CartPole-v1` · `ALE/Pong-v5` |
| Observation | flat vector · stacked Atari frames |
| Trainer | `neatrl.reinforce_mlp.train_reinforce` · `neatrl.reinforce_cnn.train_reinforce_cnn` |
| Extra needed | none · `neatrl[atari]` |
| Docs | [`docs/non-cli/REINFORCE/`](../non-cli/REINFORCE/) |

## Run it

```bash
neatrl train reinforce                     # CartPole-v1
neatrl train reinforce CartPole-v1 --steps 1000    # 1000 EPISODES
neatrl train reinforce-cnn                 # ALE/Pong-v5
```

## `--steps` counts episodes here

REINFORCE learns from complete rollouts, so its trainer takes
`total_episodes`, not `total_timesteps` — the one alias where `--steps N` means
episodes:

| Command | Meaning |
|---|---|
| `neatrl train reinforce --steps 1000` | 1000 episodes |
| `neatrl train ppo --steps 1000` | 1000 transitions |

So a "small" number that is a smoke test elsewhere still runs a long time here.

Full reference: [CLI](README.md) · [algorithm](../non-cli/REINFORCE/README.md)
