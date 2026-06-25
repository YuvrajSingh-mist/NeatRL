"""Tier 3 — make_env thunk and evaluate functional tests."""

import gymnasium as gym
import torch

from neatrl.ddpg_mlp import ActorNet as DDPGActor
from neatrl.ddpg_mlp import evaluate as ddpg_evaluate
from neatrl.ddpg_mlp import make_env as ddpg_make_env
from neatrl.dqn_mlp import QNet
from neatrl.dqn_mlp import evaluate as dqn_evaluate
from neatrl.dqn_mlp import make_env as dqn_make_env
from neatrl.ppo_mlp import make_env as ppo_make_env

# ─── make_env thunk ──────────────────────────────────────────────────────────


class TestMakeEnv:
    def test_dqn_thunk_returns_callable(self):
        thunk = dqn_make_env("CartPole-v1", seed=0, idx=0)
        assert callable(thunk)

    def test_dqn_thunk_creates_env(self):
        thunk = dqn_make_env("CartPole-v1", seed=0, idx=0)
        env = thunk()
        assert isinstance(env, gym.Env)
        env.close()

    def test_dqn_thunk_obs_act_space(self):
        thunk = dqn_make_env("CartPole-v1", seed=0, idx=0)
        env = thunk()
        assert env.observation_space.shape == (4,)
        assert env.action_space.n == 2
        env.close()

    def test_ppo_thunk_returns_callable(self):
        thunk = ppo_make_env("CartPole-v1", seed=0, idx=0)
        assert callable(thunk)

    def test_ppo_thunk_creates_env(self):
        env = ppo_make_env("CartPole-v1", seed=0, idx=0)()
        assert isinstance(env, gym.Env)
        env.close()

    def test_ddpg_thunk_creates_continuous_env(self):
        env = ddpg_make_env("Pendulum-v1", seed=0, idx=0)()
        assert isinstance(env, gym.Env)
        assert env.observation_space.shape == (3,)
        env.close()

    def test_seed_offset_by_idx(self):
        # Two envs with different idx should produce different initial observations
        env0 = dqn_make_env("CartPole-v1", seed=42, idx=0)()
        env1 = dqn_make_env("CartPole-v1", seed=42, idx=1)()
        obs0, _ = env0.reset()
        obs1, _ = env1.reset()
        # Seeds differ → observations should differ (very likely)
        assert not (obs0 == obs1).all()
        env0.close()
        env1.close()


# ─── evaluate ────────────────────────────────────────────────────────────────


class TestEvaluateDQN:
    def test_returns_list_of_floats(self):
        net = QNet(4, 2)
        returns, frames = dqn_evaluate(
            env_id="CartPole-v1",
            model=net,
            device=torch.device("cpu"),
            seed=0,
            num_eval_eps=2,
            capture_video=False,
        )
        assert isinstance(returns, list)
        assert len(returns) == 2
        assert all(isinstance(r, float) for r in returns)

    def test_no_video_frames_when_not_requested(self):
        net = QNet(4, 2)
        _, frames = dqn_evaluate(
            env_id="CartPole-v1",
            model=net,
            device=torch.device("cpu"),
            seed=0,
            num_eval_eps=1,
            capture_video=False,
        )
        assert frames == []

    def test_returns_are_positive(self):
        net = QNet(4, 2)
        returns, _ = dqn_evaluate(
            env_id="CartPole-v1",
            model=net,
            device=torch.device("cpu"),
            seed=0,
            num_eval_eps=2,
            capture_video=False,
        )
        assert all(r > 0 for r in returns)


class TestEvaluateDDPG:
    def test_returns_list_of_floats(self):
        actor = DDPGActor(3, 1)
        returns, frames = ddpg_evaluate(
            model=actor,
            device=torch.device("cpu"),
            env_id="Pendulum-v1",
            seed=0,
            num_eval_eps=2,
            record=False,
        )
        assert isinstance(returns, list)
        assert len(returns) == 2
        assert all(isinstance(r, float) for r in returns)

    def test_no_video_frames_when_not_requested(self):
        actor = DDPGActor(3, 1)
        _, frames = ddpg_evaluate(
            model=actor,
            device=torch.device("cpu"),
            env_id="Pendulum-v1",
            seed=0,
            num_eval_eps=1,
            record=False,
        )
        assert frames == []
