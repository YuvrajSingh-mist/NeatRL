# DDPG — `neatrl train ddpg`

Deep Deterministic Policy Gradient: a deterministic actor trained by
backpropagating through a critic, with Polyak-averaged target networks. DQN
extended to continuous actions.

| | |
|---|---|
| Aliases | `ddpg` · `ddpg-cnn` |
| Default env | `Pendulum-v1` · `CarRacing-v3` |
| Observation | flat vector · Box2D images |
| Trainer | `neatrl.ddpg_mlp.train_ddpg` · `neatrl.ddpg_cnn.train_ddpg_cnn` |
| Extra needed | none · `neatrl[box2d]` |
| Docs | [`docs/non-cli/DDPG/`](../non-cli/DDPG/) |

## Run it

```bash
neatrl train ddpg                          # Pendulum-v1
neatrl train ddpg Pendulum-v1 --steps 50000
neatrl train ddpg HalfCheetah-v4           # needs neatrl[mujoco]
neatrl train ddpg-cnn                      # CarRacing-v3
neatrl train ddpg-cnn CarRacing-v3 --steps 200000
```

## What the alias passes

Both aliases pass `capture_video=False` and `use_wandb=False`; `ddpg-cnn`
applies the CarRacing-specific image preprocessing (cropped, stacked frames).
`--steps` sets `total_timesteps`.

## Environments

Continuous actions only. `CarRacing-v3` needs `pip install "neatrl[box2d]"`;
the MuJoCo environments need `pip install "neatrl[mujoco]"`.

CarRacing is the one image family for `ddpg-cnn` — do not expect ALE games to
work on it. For pixels in an Atari game use a `-cnn` alias from the DQN family.

## Programmatic equivalent

```python
from neatrl.ddpg_mlp import train_ddpg

model = train_ddpg(
    env_id="Pendulum-v1",
    total_timesteps=50_000,
    capture_video=False,
    use_wandb=False,
)
```

Full reference: [CLI](README.md) · [algorithm](../non-cli/DDPG/README.md)
