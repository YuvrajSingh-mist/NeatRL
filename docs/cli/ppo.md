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
neatrl train ppo                        # CartPole-v1 (discrete)
neatrl train ppo CartPole-v1 --steps 200000
neatrl train ppo Pendulum-v1            # continuous, Gaussian head
neatrl train ppo LunarLander-v3
neatrl train ppo-cnn                    # ALE/Pong-v5
neatrl train ppo-cnn ALE/Breakout-v5 --steps 2000000
```

Full reference: [CLI](README.md) · [algorithm](../non-cli/PPO/README.md)
