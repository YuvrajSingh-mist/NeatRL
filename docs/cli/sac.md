# SAC — `neatrl train sac`

Soft Actor-Critic: maximum-entropy RL — maximise reward *plus* policy entropy,
with an automatically tuned temperature. The strong off-policy default for
continuous control.

| | |
|---|---|
| Aliases | `sac` · `sac-cnn` |
| Default env | `Pendulum-v1` · `CarRacing-v3` |
| Observation | flat vector · Box2D images |
| Trainer | `neatrl.sac_mlp.train_sac` · `neatrl.sac_cnn.train_sac_cnn` |
| Extra needed | none · `neatrl[box2d]` |
| Docs | [`docs/non-cli/SAC/`](../non-cli/SAC/) |

## Run it

```bash
neatrl train sac                           # Pendulum-v1
neatrl train sac Pendulum-v1 --steps 50000
neatrl train sac HalfCheetah-v4            # needs neatrl[mujoco]
neatrl train sac-cnn                       # CarRacing-v3
neatrl train sac-cnn CarRacing-v3 --steps 200000
```

## What the alias passes

Both aliases pass `capture_video=False` and `use_wandb=False`; `sac-cnn` applies
CarRacing image preprocessing. `--steps` sets `total_timesteps`.

## Environments

Continuous actions only. The entropy term makes SAC the most forgiving of the
three continuous-control algorithms when the reward scale is unhelpful — a
reasonable first attempt before hand-tuning anything.

## Programmatic equivalent

```python
from neatrl.sac_mlp import train_sac

model = train_sac(
    env_id="Pendulum-v1",
    total_timesteps=50_000,
    capture_video=False,
    use_wandb=False,
)
```

Full reference: [CLI](README.md) · [algorithm](../non-cli/SAC/README.md)
