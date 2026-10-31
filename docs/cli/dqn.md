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
pip install neatrl

neatrl train dqn                        # CartPole-v1, the default
neatrl train dqn CartPole-v1 --steps 200000
neatrl train dqn MountainCar-v0
neatrl train dqn LunarLander-v3
neatrl train dqn FrozenLake-v1          # discrete observations are fine
neatrl train dqn ALE/Pong-v5            # note: switches to dqn-cnn
```

Full reference: [CLI](README.md) · [algorithm](../non-cli/DQN/README.md)
