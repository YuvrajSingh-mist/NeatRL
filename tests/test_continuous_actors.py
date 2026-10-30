"""The built-in PPO / PPO-RND actors read the action space.

These four trainers used to ship a categorical head only, so a Box environment
crashed inside gymnasium with ``IndexError: invalid index to scalar variable``
-- *after* the first rollout, which made it look like an environment bug. The
actor now picks its head from the space it is built for: categorical logits for
``Discrete``, a Normal mean for ``Box``.

The tests below pin both halves: the distribution each head produces, and a
short end-to-end run on Pendulum (continuous) and CartPole (discrete).
"""

import pytest
import torch

BUILTIN_ACTORS = [
    ("neatrl.ppo_mlp", "ActorNet", 4),
    ("neatrl.ppo_cnn", "ActorNet", 4),
    ("neatrl.rnd_mlp", "ActorNet", 4),
    ("neatrl.rnd_cnn", "ActorNet", 4),
]
IDS = [f"{m.split('.')[-1]}.{c}" for m, c, _ in BUILTIN_ACTORS]


def _actor(module, name, obs_dim, act_dim, continuous):
    import importlib

    cls = getattr(importlib.import_module(module), name)
    return cls(obs_dim, act_dim, continuous=continuous)


@pytest.mark.parametrize("module,name,act_dim", BUILTIN_ACTORS, ids=IDS)
def test_continuous_actor_returns_a_normal(module, name, act_dim):
    net = _actor(module, name, obs_dim=4, act_dim=act_dim, continuous=True)
    dist = net(torch.randn(8, 4))

    assert isinstance(dist, torch.distributions.Normal)
    assert dist.mean.shape == (8, act_dim)


@pytest.mark.parametrize("module,name,act_dim", BUILTIN_ACTORS, ids=IDS)
def test_discrete_actor_still_returns_a_categorical(module, name, act_dim):
    net = _actor(module, name, obs_dim=4, act_dim=act_dim, continuous=False)
    dist = net(torch.randn(8, 4))

    assert isinstance(dist, torch.distributions.Categorical)


@pytest.mark.parametrize("module,name,act_dim", BUILTIN_ACTORS, ids=IDS)
def test_continuous_log_prob_is_one_scalar_per_step(module, name, act_dim):
    """The rollout stores ``(max_steps, n_envs)`` log-probs, so a multi-dim
    action must not leak a per-dimension tensor into the PPO ratio."""
    net = _actor(module, name, obs_dim=4, act_dim=act_dim, continuous=True)
    action, log_prob, dist = net.get_action(torch.randn(8, 4))

    assert action.shape == (8, act_dim)
    assert log_prob.shape == (8,)
    # And it has to be the sum over the action dims, not just the first one.
    expected = dist.log_prob(action).sum(dim=-1)
    assert torch.allclose(log_prob, expected)


def test_actor_defaults_to_discrete():
    """``continuous`` is keyword-only-in-spirit: existing 2-arg callers must be
    unaffected, and that includes custom actor classes and the docs scripts."""
    from neatrl.ppo_mlp import ActorNet

    assert isinstance(
        ActorNet(4, 2)(torch.randn(1, 4)), torch.distributions.Categorical
    )


@pytest.mark.parametrize(
    "trainer,continuous,discrete",
    [
        ("train_ppo", "Pendulum-v1", "CartPole-v1"),
        ("train_ppo_rnd", "Pendulum-v1", "CartPole-v1"),
    ],
)
def test_default_actor_trains_both_action_spaces(trainer, continuous, discrete):
    import importlib

    fn = getattr(importlib.import_module("neatrl"), trainer)
    for env_id in (continuous, discrete):
        fn(
            env_id=env_id,
            total_timesteps=1024,
            n_envs=2,
            max_steps=64,
            use_wandb=False,
            capture_video=False,
            eval_every=100000,
            save_every=100000,
        )
