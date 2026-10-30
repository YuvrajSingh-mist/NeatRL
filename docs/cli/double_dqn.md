# Double DQN — `neatrl train double-dqn`

DQN with decoupled action selection and evaluation: the online network picks
the next action, the target network scores it. Removes the max-operator
overestimation that plain DQN inherits from its own target.

| | |
|---|---|
| Aliases | `double-dqn` · `double-dqn-cnn` |
| Default env | `CartPole-v1` · `ALE/Pong-v5` |
| Observation | flat vector · stacked Atari frames |
| Trainer | `neatrl.double_dqn_mlp.train_double_dqn` · `neatrl.double_dqn_cnn.train_double_dqn_cnn` |
| Extra needed | none · `neatrl[atari]` |
| Docs | [`docs/non-cli/Double-DQN/`](../non-cli/Double-DQN/) |

## Run it

```bash
neatrl train double-dqn                          # CartPole-v1
neatrl train double-dqn CartPole-v1 --steps 200000
neatrl train double-dqn-cnn                      # ALE/Pong-v5
neatrl train double-dqn-cnn ALE/Breakout-v5 --steps 500000
```

Full reference: [CLI](README.md) · [algorithm](../non-cli/Double-DQN/README.md)
