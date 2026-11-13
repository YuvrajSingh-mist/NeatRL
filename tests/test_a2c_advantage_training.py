"""End-to-end A2C trains that exercise ``advantage`` on MLP and CNN paths.

The unit tests in ``test_a2c_advantage_flag.py`` pin the formulas. These go
further: actually run the trainers for a few thousand steps on real (or
synthetic image) environments and assert the run finishes, losses stay finite,
and the critic still moves when the baseline is off.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium import Env, spaces

from neatrl.a2c_cnn import CriticNet as CnnCritic
from neatrl.a2c_cnn import train_a2c_cnn
from neatrl.a2c_mlp import CriticNet as MlpCritic
from neatrl.a2c_mlp import train_a2c


class TinyImageDiscreteEnv(Env):
    """Minimal image + discrete env so the CNN trainer can run without ALE.

    Observation is a ``(3, 42, 42)`` float32 frame; action is ``Discrete(3)``.
    Reward is ``+1`` until a short horizon, which is enough to drive a few
    A2C updates without needing Atari deps.
    """

    metadata = {"render_modes": []}

    def __init__(self, horizon: int = 32):
        super().__init__()
        self.horizon = horizon
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(3, 42, 42), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)
        self._t = 0
        self._spec = type("S", (), {"id": "TinyImageDiscrete-v0"})()

    @property
    def spec(self):
        return self._spec

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        return self.observation_space.sample(), {}

    def step(self, action):
        self._t += 1
        terminated = self._t >= self.horizon
        return self.observation_space.sample(), 1.0, terminated, False, {}


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


@pytest.mark.parametrize("advantage", [True, False], ids=["adv-on", "adv-off"])
@pytest.mark.parametrize(
    "env_id,obs_dim",
    [
        ("CartPole-v1", 4),
        ("Acrobot-v1", 6),  # classic-control; avoids the Box2D LunarLander dep
        ("MountainCar-v0", 2),
    ],
    ids=["cartpole", "acrobot", "mountaincar"],
)
def test_mlp_trains_on_vector_envs(env_id, obs_dim, advantage):
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

    after = [p.detach().clone() for p in held["net"].parameters()]
    moved = any(not torch.allclose(a, b) for a, b in zip(held["before"], after))
    assert moved, f"critic did not update on {env_id} with advantage={advantage}"


@pytest.mark.parametrize("advantage", [True, False], ids=["adv-on", "adv-off"])
def test_cnn_trains_on_image_env(advantage):
    Probe, held = _probe_critic(CnnCritic)
    env = TinyImageDiscreteEnv(horizon=24)
    net = train_a2c_cnn(
        env=env,
        total_timesteps=1_500,
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
        atari_wrapper=False,
        exp_name=f"a2c-cnn-tiny-{advantage}",
    )
    assert isinstance(net, torch.nn.Module)
    _assert_finite_forward(net, torch.zeros(2, 3, 42, 42))

    after = [p.detach().clone() for p in held["net"].parameters()]
    moved = any(not torch.allclose(a, b) for a, b in zip(held["before"], after))
    assert moved, f"CNN critic did not update with advantage={advantage}"
