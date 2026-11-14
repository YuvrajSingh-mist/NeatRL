"""End-to-end A2C trains that exercise ``advantage`` on real MLP and CNN envs.

Needs the optional extras the CLI already hints at:

    pip install "neatrl[box2d]"   # LunarLander, CarRacing
    pip install "neatrl[atari]"   # ALE/Pong

Continuous action spaces (Pendulum, CarRacing) use the same built-in
``ActorNet(..., continuous=True)`` path as PPO / PPO-RND.
"""

from __future__ import annotations

import pytest
import torch

from neatrl.a2c_cnn import CriticNet as CnnCritic
from neatrl.a2c_cnn import train_a2c_cnn
from neatrl.a2c_mlp import CriticNet as MlpCritic
from neatrl.a2c_mlp import train_a2c


def _probe_critic(base):
    """Subclass ``base`` and keep a handle + initial weight snapshot."""
    held: dict = {}

    class Probe(base):  # type: ignore[valid-type,misc]
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            held["net"] = self
            held["before"] = [p.detach().clone() for p in self.parameters()]

    return Probe, held


def _assert_finite_forward(net, obs):
    dist = net(obs)
    action, log_prob, _ = net.get_action(obs)
    assert torch.isfinite(log_prob).all()
    assert action.shape[0] == obs.shape[0]
    assert hasattr(dist, "sample")


def _critic_moved(held) -> bool:
    after = [p.detach().clone() for p in held["net"].parameters()]
    return any(not torch.allclose(a, b) for a, b in zip(held["before"], after))


@pytest.mark.parametrize("advantage", [True, False], ids=["adv-on", "adv-off"])
@pytest.mark.parametrize(
    "env_id,obs_dim",
    [
        ("CartPole-v1", 4),
        ("Acrobot-v1", 6),
        ("LunarLander-v3", 8),  # neatrl[box2d]
        ("Pendulum-v1", 3),  # continuous Box — Normal head
    ],
    ids=["cartpole", "acrobot", "lunarlander", "pendulum"],
)
def test_mlp_trains_on_real_envs(env_id, obs_dim, advantage):
    Probe, held = _probe_critic(MlpCritic)
    net = train_a2c(
        env_id=env_id,
        total_timesteps=2_000,
        seed=0,
        use_wandb=False,
        capture_video=False,
        eval_every=10_000,
        save_every=10_000,
        num_eval_episodes=1,
        ENTROPY_COEFF=0.01,
        device="cpu",
        advantage=advantage,
        critic_class=Probe,
        exp_name=f"a2c-mlp-{env_id}-{advantage}",
    )
    assert isinstance(net, torch.nn.Module)
    _assert_finite_forward(net, torch.zeros(2, obs_dim))
    assert _critic_moved(held), (
        f"critic did not update on {env_id} advantage={advantage}"
    )


@pytest.mark.parametrize("advantage", [True, False], ids=["adv-on", "adv-off"])
@pytest.mark.parametrize(
    "env_id,obs_shape,kwargs",
    [
        # continuous image env — Normal head via continuous=True
        (
            "CarRacing-v3",
            (3, 96, 96),
            {"atari_wrapper": False},
        ),
        # discrete Atari — Categorical head + ALE preprocessing
        (
            "ALE/Pong-v5",
            (4, 84, 84),
            {"atari_wrapper": True},
        ),
    ],
    ids=["carracing", "pong"],
)
def test_cnn_trains_on_real_envs(env_id, obs_shape, kwargs, advantage):
    Probe, held = _probe_critic(CnnCritic)
    net = train_a2c_cnn(
        env_id=env_id,
        total_timesteps=800,
        seed=0,
        use_wandb=False,
        capture_video=False,
        eval_every=10_000,
        save_every=10_000,
        num_eval_episodes=1,
        ENTROPY_COEFF=0.01,
        device="cpu",
        advantage=advantage,
        critic_class=Probe,
        exp_name=f"a2c-cnn-{env_id.split('/')[-1]}-{advantage}",
        n_envs=1,
        **kwargs,
    )
    assert isinstance(net, torch.nn.Module)
    # CarRacing is channel-last from gym; the trainer's observation_input_layer
    # accepts either layout. Feed the shape the net was built for.
    # After AtariPreprocessing + FrameStack the obs is (4, 84, 84).
    # CarRacing raw is (96, 96, 3) — ImageFeatures accepts channel-last too.
    if env_id.startswith("ALE/"):
        obs = torch.zeros(2, *obs_shape)
    else:
        obs = torch.zeros(2, 96, 96, 3)
    _assert_finite_forward(net, obs)
    assert _critic_moved(held), (
        f"CNN critic did not update on {env_id} advantage={advantage}"
    )
