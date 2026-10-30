# PPO — `neatrl train ppo`

Proximal Policy Optimization: several epochs of gradient updates per batch of
experience, with the probability ratio clipped so one update cannot wreck the
policy.

| | |
|---|---|
| Aliases | `ppo` · `ppo-cnn` |
| Default env | `CartPole-v1` · `ALE/Pong-v5` |
| Observation | flat vector · stacked Atari frames |
| Trainer | `neatrl.ppo_mlp.train_ppo` · `neatrl.ppo_cnn.train_ppo_cnn` |
| Extra needed | none · `neatrl[atari]` |
| Docs | [`docs/non-cli/PPO/`](../non-cli/PPO/) |

## Run it

```bash
neatrl train ppo                        # CartPole-v1
neatrl train ppo CartPole-v1 --steps 200000
neatrl train ppo LunarLander-v3
neatrl train ppo-cnn                    # ALE/Pong-v5
neatrl train ppo-cnn ALE/Breakout-v5 --steps 2000000
```

## What the alias passes

Both aliases pass `capture_video=False` and `use_wandb=False`; the CNN alias
adds Atari preprocessing. `--steps` sets `total_timesteps`.

## Environments

Discrete-action environments. PPO is the usual first choice for a new
discrete-action problem: it tolerates defaults better than A2C and needs no
replay buffer. The built-in actor is categorical, so naming a continuous
environment raises a `ValueError`; for continuous control use `sac`, `td3` or
`ddpg`, or a custom `actor_class` from Python.

## Programmatic equivalent

```python
from neatrl.ppo_mlp import train_ppo

model = train_ppo(
    env_id="CartPole-v1",
    total_timesteps=200_000,
    capture_video=False,
    use_wandb=False,
)
```

Full reference: [CLI](README.md) · [algorithm](../non-cli/PPO/README.md)
