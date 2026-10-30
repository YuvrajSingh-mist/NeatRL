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

Full reference: [CLI](README.md) · [algorithm](../non-cli/DDPG/README.md)
