# Tabular Q-learning

Off-policy TD control with a Q-table. For **simple discrete Gym envs only**
(`FrozenLake-v1`, `CliffWalking-v0`, `Taxi-v3`, …). Continuous observations → use DQN.

```bash
pip install "neatrl[classic]"
python docs/non-cli/Q-learning/run_q_learning_frozenlake.py
# or
neatrl train q-learning FrozenLake-v1
```

```python
from neatrl.q_learning import train_q_learning

q_table = train_q_learning(
    env_id="FrozenLake-v1", total_timesteps=50_000, use_wandb=False
)
```
