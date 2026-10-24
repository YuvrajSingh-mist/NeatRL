"""Unit + smoke tests for tabular Q-learning (discrete Gym envs only)."""

import dataclasses

import gymnasium as gym
import numpy as np
import pytest

from neatrl.q_learning import (
    Config,
    _require_discrete,
    epsilon_by_step,
    evaluate,
    make_env,
    select_action,
    train_q_learning,
)


class TestConfig:
    def test_is_dataclass(self):
        assert dataclasses.is_dataclass(Config)

    def test_defaults_instantiate(self):
        cfg = Config()
        assert cfg.env_id == "FrozenLake-v1"
        assert cfg.lr > 0
        assert 0 <= cfg.gamma <= 1
        assert cfg.start_e >= cfg.end_e

    def test_equality(self):
        assert Config() == Config()
        other = Config()
        other.seed = 7
        assert Config() != other


class TestRequireDiscrete:
    def test_accepts_frozenlake(self):
        env = gym.make("FrozenLake-v1")
        n_s, n_a = _require_discrete(env)
        assert n_s == 16
        assert n_a == 4
        env.close()

    def test_rejects_box_observation(self):
        env = gym.make("CartPole-v1")
        with pytest.raises(ValueError, match="Discrete observation"):
            _require_discrete(env)
        env.close()


class TestMakeEnv:
    def test_thunk_returns_callable(self):
        assert callable(make_env("FrozenLake-v1", seed=0, idx=0))

    def test_thunk_creates_discrete_env(self):
        env = make_env("FrozenLake-v1", seed=0, idx=0)()
        assert isinstance(env.observation_space, gym.spaces.Discrete)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        env.close()


class TestEpsilonDecay:
    def test_starts_near_start_e(self):
        eps = epsilon_by_step(0, 1000, start_e=1.0, end_e=0.05, exploration_fraction=0.5)
        assert eps == pytest.approx(1.0)

    def test_ends_at_end_e(self):
        eps = epsilon_by_step(
            1000, 1000, start_e=1.0, end_e=0.05, exploration_fraction=0.5
        )
        assert eps == pytest.approx(0.05)

    def test_monotone_decrease(self):
        values = [
            epsilon_by_step(s, 1000, 1.0, 0.05, 0.5) for s in range(0, 500, 50)
        ]
        assert values == sorted(values, reverse=True)


class TestSelectAction:
    def test_greedy_when_eps_zero(self):
        q = np.array([[0.0, 2.0, 1.0]], dtype=np.float64)
        assert select_action(q, state=0, eps=0.0, n_actions=3) == 1

    def test_explores_when_eps_one(self):
        q = np.zeros((1, 4))
        actions = {select_action(q, 0, eps=1.0, n_actions=4) for _ in range(40)}
        assert len(actions) > 1


class TestEvaluate:
    def test_returns_list_of_floats(self):
        q = np.zeros((16, 4))
        returns, frames = evaluate(
            "FrozenLake-v1", q, seed=0, num_eval_eps=2, capture_video=False
        )
        assert len(returns) == 2
        assert all(isinstance(r, float) for r in returns)
        assert frames == []


class TestTrainQLearning:
    def test_short_run_returns_q_table(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        q = train_q_learning(
            env_id="FrozenLake-v1",
            total_timesteps=200,
            seed=0,
            use_wandb=False,
            eval_every=500,
            save_every=10_000,
            exp_name="test-ql",
            num_eval_eps=1,
        )
        assert isinstance(q, np.ndarray)
        assert q.shape == (16, 4)
        assert np.isfinite(q).all()

    def test_rejects_continuous_env(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="Discrete observation"):
            train_q_learning(
                env_id="CartPole-v1",
                total_timesteps=10,
                use_wandb=False,
                exp_name="test-ql-bad",
            )

    def test_td_update_moves_q(self, tmp_path, monkeypatch):
        """CliffWalking gives a reward every step, so the Q-table must leave zeros."""
        monkeypatch.chdir(tmp_path)
        q = train_q_learning(
            env_id="CliffWalking-v0",
            total_timesteps=200,
            seed=0,
            lr=0.5,
            start_e=1.0,
            end_e=0.1,
            exploration_fraction=0.5,
            use_wandb=False,
            eval_every=10_000,
            save_every=10_000,
            exp_name="test-ql-td",
            num_eval_eps=1,
        )
        assert q.shape[1] == 4
        assert np.any(q != 0.0)
        assert np.isfinite(q).all()
