"""Tier 3 — integration: every DQN-family trainer on every observation type.

Three observation kinds are covered, because each exercises a different code
path in the trainers:

  * discrete observations  -> needs ``grid_env=True`` (one-hot encoding)
  * continuous observations -> plain MLP Q-network
  * image observations     -> CNN Q-network over stacked Atari frames

Both plain DQN and Double DQN are trained in each case so a regression in the
bootstrap target or in the env plumbing is caught for both.

Full coverage (all 18 env/trainer pairs) lives in ``scripts/check_dqn_family.py``;
this module keeps one representative env per type so the suite stays fast.
"""

import numpy as np
import pytest
import torch

from neatrl import (
    train_double_dqn,
    train_double_dqn_cnn,
    train_dqn,
    train_dqn_cnn,
)

TRAINERS = {
    "dqn": train_dqn,
    "double_dqn": train_double_dqn,
    "dqn_cnn": train_dqn_cnn,
    "double_dqn_cnn": train_double_dqn_cnn,
}

# (trainer, env_id, grid_env)
CASES = [
    pytest.param("dqn", "FrozenLake-v1", True, id="discrete-obs/dqn"),
    pytest.param("double_dqn", "FrozenLake-v1", True, id="discrete-obs/double-dqn"),
    pytest.param("dqn", "CartPole-v1", False, id="continuous-obs/dqn"),
    pytest.param("double_dqn", "CartPole-v1", False, id="continuous-obs/double-dqn"),
    pytest.param("dqn_cnn", "ALE/Pong-v5", False, id="image-obs/dqn"),
    pytest.param("double_dqn_cnn", "ALE/Pong-v5", False, id="image-obs/double-dqn"),
]


def _mlp_kwargs():
    return {
        "total_timesteps": 300,
        "seed": 0,
        "buffer_size": 2000,
        "batch_size": 32,
        "learning_starts": 100,
        "eval_every": 200,
        "save_every": 0,
        "num_eval_eps": 2,
        "use_wandb": False,
        "capture_video": False,
        "device": "cpu",
        "atari_wrapper": False,
        "train_frequency": 4,
        "target_network_frequency": 100,
    }


def _cnn_kwargs():
    return {
        "total_timesteps": 100,
        "seed": 0,
        "buffer_size": 400,
        "batch_size": 8,
        "learning_starts": 50,
        "eval_every": 80,
        "save_every": 0,
        "num_eval_eps": 1,
        "use_wandb": False,
        "capture_video": False,
        "device": "cpu",
        "target_network_frequency": 50,
    }


@pytest.mark.parametrize("trainer_key,env_id,grid_env", CASES)
def test_trainer_runs_on_observation_type(trainer_key, env_id, grid_env):
    """Train briefly, then check the returned net behaves on a real observation."""
    train_fn = TRAINERS[trainer_key]
    is_cnn = trainer_key.endswith("_cnn")

    evals = []
    kwargs = _cnn_kwargs() if is_cnn else _mlp_kwargs()
    kwargs["env_id"] = env_id
    if not is_cnn:
        kwargs["grid_env"] = grid_env

    net = train_fn(**kwargs, eval_callback=lambda step, m: evals.append((step, m)))

    # A network came back and evaluation actually ran.
    assert isinstance(net, torch.nn.Module)
    assert evals, f"{trainer_key} never evaluated on {env_id}"
    step, metrics = evals[0]
    assert isinstance(step, int)
    assert np.isfinite(metrics["avg_return"])
    assert isinstance(metrics["model"], torch.nn.Module)

    # A forward pass on a fresh observation has the environment's action count.
    if is_cnn:
        from neatrl.dqn_cnn import _obs_tensor, make_env

        env = make_env(env_id, 0, 0, atari_wrapper=True)()
    else:
        from neatrl.dqn_mlp import make_env

        env = make_env(env_id, 0, 0, atari_wrapper=False, grid_env=grid_env)()

    try:
        obs, _ = env.reset()
        n_actions = env.action_space.n
        with torch.no_grad():
            if is_cnn:
                q = net(_obs_tensor(obs, "cpu"))
            else:
                q = net(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
    finally:
        env.close()

    assert q.shape == (1, n_actions), f"{trainer_key}/{env_id}: got {q.shape}"
    assert torch.isfinite(q).all(), f"non-finite Q values on {env_id}"


def test_double_target_differs_from_plain_max():
    """Double DQN must bootstrap from the online-selected action, not the max."""
    from neatrl.double_dqn_mlp import QNet, double_dqn_target

    torch.manual_seed(0)
    online = QNet(4, 3)
    target = QNet(4, 3)  # deliberately different weights
    obs = torch.randn(64, 4)
    rewards = torch.zeros(64, 1)
    dones = torch.zeros(64, 1)

    double = double_dqn_target(online, target, obs, rewards, dones, 0.99)
    with torch.no_grad():
        plain = 0.99 * target(obs).max(dim=1)[0]

    assert (double - plain).abs().max() > 1e-6, (
        "double DQN target collapsed to the plain max target"
    )


def test_evaluate_terminates_without_env_timelimit():
    """CliffWalking-v1 has no TimeLimit; evaluate() must still return.

    Regression test: the greedy rollout used to loop forever because nothing
    ever set terminated/truncated.
    """
    from neatrl.dqn_mlp import QNet, evaluate

    net = QNet(48, 4)
    returns, _frames = evaluate(
        "CliffWalking-v1",
        net,
        "cpu",
        seed=0,
        num_eval_eps=1,
        grid_env=True,
    )
    assert len(returns) == 1
    assert np.isfinite(returns[0])
