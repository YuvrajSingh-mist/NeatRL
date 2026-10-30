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
neatrl train rnd                           # CliffWalking-v1
neatrl train rnd CliffWalking-v1 --steps 100000
neatrl train rnd CartPole-v1
neatrl train rnd-cnn                       # ALE/Pong-v5
```

## What the alias passes

Both aliases pass `capture_video=False` and `use_wandb=False`; `rnd-cnn` adds
Atari preprocessing. `--steps` sets `total_timesteps`.

The trainer is PPO with the RND bonus bolted on — hence `train_ppo_rnd` in the
import path, not `train_rnd`.

## Environments

Best where the environment reward is sparse: `CliffWalking-v1` (the default) is
almost all negative reward, so the bonus is what keeps the agent moving. On a
dense-reward task the bonus mostly adds noise.

Full reference: [CLI](README.md) · [algorithm](../non-cli/RND/README.md)
