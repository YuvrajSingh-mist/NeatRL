# TD3 — `neatrl train td3`

Twin Delayed DDPG: two critics (target = the minimum of the pair), actor
updates delayed behind the critic, and target policy smoothing. The three fixes
for DDPG's overestimation and fragility.

| | |
|---|---|
| Aliases | `td3` · `td3-cnn` |
| Default env | `Pendulum-v1` · `CarRacing-v3` |
| Observation | flat vector · Box2D images |
| Trainer | `neatrl.td3_mlp.train_td3` · `neatrl.td3_cnn.train_td3_cnn` |
| Extra needed | none · `neatrl[box2d]` |
| Docs | [`docs/non-cli/TD3/`](../non-cli/TD3/) |

## Run it

```bash
neatrl train td3                          # Pendulum-v1
neatrl train td3 Pendulum-v1 --steps 50000
neatrl train td3 HalfCheetah-v4           # needs neatrl[mujoco]
neatrl train td3-cnn                      # CarRacing-v3
neatrl train td3-cnn CarRacing-v3 --steps 200000
```

## What the alias passes

Both aliases pass `capture_video=False` and `use_wandb=False`; `td3-cnn`
applies CarRacing image preprocessing (cropped, stacked frames). `--steps`
sets `total_timesteps`.

## Environments

Continuous actions only. TD3 is usually the better default than DDPG on the
same environment — same interface, fewer divergences.

Full reference: [CLI](README.md) · [algorithm](../non-cli/TD3/README.md)
