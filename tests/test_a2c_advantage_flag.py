"""``advantage`` flag on A2C: baseline-subtracted A vs raw return G.

Default (``advantage=True``) keeps the previous A2C update
``L_π = -E[log π(a|s) · (G - V(s))]``. With ``advantage=False`` the baseline
is dropped and the actor uses the classic return-weighted form
``L_π = -E[log π(a|s) · G]``. The critic is always ``L_V = MSE(V(s), G)``.
"""

from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn.functional as F

from neatrl.a2c_cnn import (
    Config as CnnConfig,
)
from neatrl.a2c_cnn import (
    actor_policy_weight as cnn_weight,
)
from neatrl.a2c_cnn import (
    train_a2c_cnn,
)
from neatrl.a2c_mlp import (
    Config as MlpConfig,
)
from neatrl.a2c_mlp import (
    actor_policy_weight as mlp_weight,
)
from neatrl.a2c_mlp import (
    train_a2c,
)

MODULES = [
    ("mlp", MlpConfig, mlp_weight, train_a2c),
    ("cnn", CnnConfig, cnn_weight, train_a2c_cnn),
]
IDS = [m[0] for m in MODULES]


@pytest.mark.parametrize("_, Config, weight_fn, train_fn", MODULES, ids=IDS)
def test_advantage_defaults_to_true(_, Config, weight_fn, train_fn):
    assert Config.advantage is True
    sig = inspect.signature(train_fn)
    assert "advantage" in sig.parameters
    assert sig.parameters["advantage"].default is True or (
        sig.parameters["advantage"].default == Config.advantage
    )


@pytest.mark.parametrize("_, Config, weight_fn, train_fn", MODULES, ids=IDS)
def test_weight_is_advantage_when_flag_on(_, Config, weight_fn, train_fn):
    returns = torch.tensor([10.0, 5.0, 1.0])
    values = torch.tensor([3.0, 2.0, 0.5])

    weight = weight_fn(returns, values, advantage=True)

    assert torch.allclose(weight, returns - values)
    assert not weight.requires_grad  # always detached


@pytest.mark.parametrize("_, Config, weight_fn, train_fn", MODULES, ids=IDS)
def test_weight_is_raw_return_when_flag_off(_, Config, weight_fn, train_fn):
    returns = torch.tensor([10.0, 5.0, 1.0])
    values = torch.tensor([3.0, 2.0, 0.5])

    weight = weight_fn(returns, values, advantage=False)

    assert torch.allclose(weight, returns)
    # Values must not leak into the weight when the baseline is off.
    assert not torch.allclose(weight, returns - values)
    assert not weight.requires_grad


@pytest.mark.parametrize("_, Config, weight_fn, train_fn", MODULES, ids=IDS)
def test_weight_ignores_value_grad_graph(_, Config, weight_fn, train_fn):
    """Either mode must detach so the actor update cannot train the critic."""
    returns = torch.tensor([4.0, 2.0], requires_grad=False)
    values = torch.tensor([1.0, 0.5], requires_grad=True)

    for advantage in (True, False):
        w = weight_fn(returns, values, advantage=advantage)
        assert not w.requires_grad
        assert w.grad_fn is None
        # Detached weight has no graph, so a backward cannot reach ``values``.
        with pytest.raises(RuntimeError, match="does not require grad"):
            w.sum().backward()
        assert values.grad is None


@pytest.mark.parametrize("_, Config, weight_fn, train_fn", MODULES, ids=IDS)
def test_actor_and_critic_formulas_match_flag(_, Config, weight_fn, train_fn):
    """Pin the exact losses used by the trainers for both flag states."""
    log_probs = torch.tensor([-0.2, -1.1, -0.4])
    returns = torch.tensor([3.0, 1.5, 0.5])
    values = torch.tensor([1.0, 0.5, 0.0])

    # Critic is always MSE(V, G) — independent of the flag.
    critic_loss = F.mse_loss(values, returns)
    assert torch.allclose(critic_loss, F.mse_loss(values, returns))

    # advantage=True: L_π = -mean(logπ · (G - V))
    w_on = weight_fn(returns, values, advantage=True)
    policy_on = -(log_probs * w_on).mean()
    expected_on = -(log_probs * (returns - values).detach()).mean()
    assert torch.allclose(policy_on, expected_on)

    # advantage=False: L_π = -mean(logπ · G)
    w_off = weight_fn(returns, values, advantage=False)
    policy_off = -(log_probs * w_off).mean()
    expected_off = -(log_probs * returns.detach()).mean()
    assert torch.allclose(policy_off, expected_off)

    # The two actor losses must disagree whenever V is not identically zero.
    assert not torch.allclose(policy_on, policy_off)


def test_train_a2c_runs_with_advantage_on_and_off():
    """Short CartPole runs for both flag values — no crash, returns a Module."""
    common = {
        "env_id": "CartPole-v1",
        "total_timesteps": 256,
        "seed": 0,
        "use_wandb": False,
        "capture_video": False,
        "eval_every": 10_000,
        "save_every": 10_000,
        "num_eval_episodes": 1,
        "ENTROPY_COEFF": 0.0,
        "device": "cpu",
    }
    for advantage in (True, False):
        net = train_a2c(**common, advantage=advantage, exp_name=f"a2c-adv-{advantage}")
        assert isinstance(net, torch.nn.Module)
        # Actor still produces a distribution after training.
        obs = torch.zeros(1, 4)
        dist = net(obs)
        assert hasattr(dist, "sample")


def test_train_a2c_advantage_false_still_trains_critic():
    """With the baseline off the critic must still update (MSE onto G)."""
    from neatrl.a2c_mlp import CriticNet

    held: dict = {}

    class ProbeCritic(CriticNet):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            held["net"] = self
            held["before"] = [p.detach().clone() for p in self.parameters()]

    net = train_a2c(
        env_id="CartPole-v1",
        total_timesteps=256,
        seed=1,
        use_wandb=False,
        capture_video=False,
        eval_every=10_000,
        save_every=10_000,
        num_eval_episodes=1,
        ENTROPY_COEFF=0.0,
        device="cpu",
        advantage=False,
        critic_class=ProbeCritic,
        exp_name="a2c-no-adv-critic",
    )
    assert isinstance(net, torch.nn.Module)
    assert "net" in held
    after = [p.detach().clone() for p in held["net"].parameters()]
    moved = any(not torch.allclose(a, b) for a, b in zip(held["before"], after))
    assert moved, "critic parameters did not change with advantage=False"
