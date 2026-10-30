# DQN — `neatrl train dqn`

Deep Q-Network: a neural Q-function trained off-policy from a replay buffer
against a frozen target network.

| | |
|---|---|
| Aliases | `dqn` · `dqn-cnn` |
| Default env | `CartPole-v1` · `ALE/Pong-v5` |
| Observation | flat vector · stacked Atari frames |
| Trainer | `neatrl.dqn_mlp.train_dqn` · `neatrl.dqn_cnn.train_dqn_cnn` |
| Extra needed | none · `neatrl[atari]` |
| Docs | [`docs/non-cli/DQN/`](../non-cli/DQN/) |

## Run it

```bash
neatrl train dqn                        # CartPole-v1, the default
neatrl train dqn CartPole-v1 --steps 200000
neatrl train dqn MountainCar-v0
neatrl train dqn LunarLander-v3
neatrl train dqn FrozenLake-v1          # discrete observations are fine
neatrl train dqn ALE/Pong-v5            # note: switches to dqn-cnn
```

## What the alias passes

The `dqn` recipe calls the trainer with `atari_wrapper=False`,
`capture_video=False` and `use_wandb=False`; `--steps` sets
`total_timesteps`, and `--wandb` flips `use_wandb`.

`dqn-cnn` passes `capture_video=False` and `use_wandb=False`, and already
applies the Atari preprocessing (frame stack, clipped rewards, 84×84 frames).

## Environments

Any vector-observation environment works, continuous or discrete. Naming an
image environment on `dqn` prints a note and runs `dqn-cnn` instead — see
[environment routing](README.md#environment-routing). CarRacing is not
supported by `dqn-cnn`, whose preprocessing is Atari-only; use `ddpg-cnn`,
`td3-cnn` or `sac-cnn` for that.

## Programmatic equivalent

```python
from neatrl.dqn_mlp import train_dqn

model = train_dqn(
    env_id="CartPole-v1",
    total_timesteps=200_000,
    atari_wrapper=False,
    capture_video=False,
    use_wandb=False,
)
```

Full reference: [CLI](README.md) · [algorithm](../non-cli/DQN/README.md)
