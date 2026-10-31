# RND — `neatrl train rnd`

Random Network Distillation: a frozen random network gives each state a
fingerprint, a predictor tries to reproduce it, and the prediction error becomes
an intrinsic exploration bonus added to the environment reward.

| | |
|---|---|
| Aliases | `rnd` · `rnd-cnn` |
| Default env | `CliffWalking-v1` · `ALE/Pong-v5` |
| Observation | flat vector · stacked Atari frames |
| Trainer | `neatrl.rnd_mlp.train_ppo_rnd` · `neatrl.rnd_cnn.train_ppo_rnd_cnn` |
| Extra needed | none · `neatrl[atari]` |
| Docs | [`docs/non-cli/RND/`](../non-cli/RND/) |

## Run it

```bash
pip install neatrl

neatrl train rnd                           # CliffWalking-v1
neatrl train rnd CliffWalking-v1 --steps 100000
neatrl train rnd CartPole-v1
neatrl train rnd-cnn                       # ALE/Pong-v5
```

Full reference: [CLI](README.md) · [algorithm](../non-cli/RND/README.md)
