"""Tests for custom_agent / actor_class / critic_class / predictor_class and env flags.

Each test exercises the dispatch or validation logic WITHOUT running a full training
loop — we test thunks, forward passes, evaluate(), and the validate_* functions that
guard custom networks at the boundary where user code enters.
"""

import pytest
import torch
import torch.nn as nn

# ─── helpers: minimal compatible custom networks ──────────────────────────────


class CustomQNet(nn.Module):
    """Drop-in replacement for dqn_mlp.QNet with identical interface."""

    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc = nn.Linear(obs_dim, act_dim)

    def forward(self, x):
        return self.fc(x)


class CustomPolicyNet(nn.Module):
    """Minimal policy net compatible with reinforce_mlp.PolicyNet interface."""

    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc = nn.Linear(obs_dim, act_dim)

    def forward(self, x):
        return self.fc(x)

    def get_action(self, x):
        logits = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action)


class CustomActorNet(nn.Module):
    """Minimal stochastic actor compatible with ppo_mlp / a2c_mlp interface."""

    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc = nn.Linear(obs_dim, act_dim)

    def forward(self, x):
        return self.fc(x)

    def get_action(self, x):
        logits = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist


class CustomCriticNet(nn.Module):
    """Minimal critic compatible with ppo_mlp / a2c_mlp interface (output dim=1)."""

    def __init__(self, obs_dim):
        super().__init__()
        self.fc = nn.Linear(obs_dim, 1)

    def forward(self, x):
        return self.fc(x)


class CustomContinuousActor(nn.Module):
    """Minimal deterministic actor for DDPG/TD3 (continuous action)."""

    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc = nn.Linear(obs_dim, act_dim)

    def forward(self, x):
        return torch.tanh(self.fc(x))


class CustomContinuousQNet(nn.Module):
    """Minimal Q-network for DDPG/TD3 (obs + act → 1)."""

    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.fc_obs = nn.Linear(obs_dim, 32)
        self.fc_act = nn.Linear(act_dim, 32)
        self.out = nn.Linear(64, 1)

    def forward(self, state, act):
        return self.out(torch.cat([self.fc_obs(state), self.fc_act(act)], dim=-1))


class CustomRNDPredictor(nn.Module):
    """Minimal RND predictor/target net (obs → feature_dim)."""

    def __init__(self, obs_dim):
        super().__init__()
        self.fc = nn.Linear(obs_dim, 32)

    def forward(self, x):
        return self.fc(x)


# ─── make_env flags ──────────────────────────────────────────────────────────


class TestMakeEnvFlags:
    def test_grid_env_applies_one_hot_wrapper(self):
        """FrozenLake has Discrete obs; grid_env=True must wrap it to Box."""
        from neatrl.dqn_mlp import make_env

        env = make_env("FrozenLake-v1", seed=0, idx=0, grid_env=True)()
        import gymnasium as gym

        assert isinstance(env.observation_space, gym.spaces.Box)
        assert env.observation_space.shape == (16,)
        env.close()

    def test_grid_env_false_keeps_discrete_obs(self):
        """Without grid_env, FrozenLake obs remains Discrete."""
        import gymnasium as gym

        from neatrl.dqn_mlp import make_env

        env = make_env("FrozenLake-v1", seed=0, idx=0, grid_env=False)()
        assert isinstance(env.observation_space, gym.spaces.Discrete)
        env.close()

    def test_grid_env_obs_is_float32_one_hot(self):
        """OneHotWrapper should produce a float32 vector with exactly one hot bit."""
        import numpy as np

        from neatrl.dqn_mlp import make_env

        env = make_env("FrozenLake-v1", seed=0, idx=0, grid_env=True)()
        obs, _ = env.reset()
        assert obs.dtype == np.float32
        assert obs.sum() == 1.0
        assert obs.shape == (16,)
        env.close()

    def test_ppo_make_env_grid_env(self):
        import gymnasium as gym

        from neatrl.ppo_mlp import make_env

        env = make_env("FrozenLake-v1", seed=0, idx=0, grid_env=True)()
        assert isinstance(env.observation_space, gym.spaces.Box)
        env.close()

    @pytest.mark.skipif(
        not __import__("importlib").util.find_spec("ale_py"),
        reason="ale_py not installed",
    )
    def test_atari_wrapper_changes_obs_shape(self):
        from neatrl.dqn_mlp import make_env

        env = make_env("ALE/Pong-v5", seed=0, idx=0, atari_wrapper=True)()
        # AtariPreprocessing + FrameStack → (4, 84, 84)
        assert env.observation_space.shape == (4, 84, 84)
        env.close()


# ─── custom_agent (DQN) ──────────────────────────────────────────────────────


class TestCustomAgentDQN:
    def test_compatible_instance_passes_validation(self):
        from neatrl.utils import validate_q_network_dimensions

        net = CustomQNet(4, 2)
        validate_q_network_dimensions(net, obs_dim=4, action_dim=2)  # must not raise

    def test_incompatible_instance_fails_validation(self):
        from neatrl.utils import validate_q_network_dimensions

        net = CustomQNet(8, 5)  # wrong dims for CartPole
        with pytest.raises(ValueError):
            validate_q_network_dimensions(net, obs_dim=4, action_dim=2)

    def test_custom_agent_works_with_evaluate(self):
        from neatrl.dqn_mlp import evaluate

        net = CustomQNet(4, 2)
        returns, frames = evaluate(
            env_id="CartPole-v1",
            model=net,
            device=torch.device("cpu"),
            seed=0,
            num_eval_eps=2,
            capture_video=False,
        )
        assert isinstance(returns, list) and len(returns) == 2
        assert all(r > 0 for r in returns)

    def test_custom_qnet_forward_shape(self):
        net = CustomQNet(4, 2)
        out = net(torch.randn(8, 4))
        assert out.shape == (8, 2)


# ─── custom actor_class as class vs instance (PPO / A2C style) ───────────────


class TestCustomActorClass:
    def test_custom_class_instantiates_with_obs_act_dims(self):
        actor = CustomActorNet(4, 2)
        assert isinstance(actor, nn.Module)

    def test_custom_class_get_action_shapes(self):
        actor = CustomActorNet(4, 2)
        x = torch.randn(4, 4)
        action, log_prob, dist = actor.get_action(x)
        assert action.shape == (4,)
        assert log_prob.shape == (4,)

    def test_custom_instance_passes_policy_validation(self):
        from neatrl.utils import validate_policy_network_dimensions

        actor = CustomActorNet(4, 2)
        validate_policy_network_dimensions(actor, obs_dim=4, action_dim=2)

    def test_incompatible_custom_actor_fails_policy_validation(self):
        from neatrl.utils import validate_policy_network_dimensions

        actor = CustomActorNet(8, 5)
        with pytest.raises(ValueError):
            validate_policy_network_dimensions(actor, obs_dim=4, action_dim=2)

    def test_custom_critic_instance_passes_validation(self):
        from neatrl.utils import validate_critic_network_dimensions

        critic = CustomCriticNet(4)
        validate_critic_network_dimensions(critic, obs_dim=4)

    def test_incompatible_custom_critic_fails_validation(self):
        from neatrl.utils import validate_critic_network_dimensions

        critic = CustomCriticNet(8)
        with pytest.raises(ValueError):
            validate_critic_network_dimensions(critic, obs_dim=4)


# ─── continuous actor/Q-net (DDPG / TD3 style) ───────────────────────────────


class TestCustomContinuousNetworks:
    def test_continuous_actor_forward_shape(self):
        actor = CustomContinuousActor(3, 1)
        out = actor(torch.randn(4, 3))
        assert out.shape == (4, 1)

    def test_continuous_actor_output_bounded(self):
        actor = CustomContinuousActor(3, 1)
        out = actor(torch.randn(64, 3))
        assert out.abs().max().item() <= 1.0 + 1e-6  # tanh output

    def test_continuous_qnet_forward_shape(self):
        qnet = CustomContinuousQNet(3, 1)
        out = qnet(torch.randn(4, 3), torch.randn(4, 1))
        assert out.shape == (4, 1)

    def test_continuous_actor_passes_policy_validation(self):
        from neatrl.utils import validate_policy_network_dimensions

        actor = CustomContinuousActor(3, 1)
        validate_policy_network_dimensions(actor, obs_dim=3, action_dim=1)

    def test_custom_actor_works_with_ddpg_evaluate(self):
        from neatrl.ddpg_mlp import evaluate

        actor = CustomContinuousActor(3, 1)
        returns, _ = evaluate(
            model=actor,
            device=torch.device("cpu"),
            env_id="Pendulum-v1",
            seed=0,
            num_eval_eps=2,
            record=False,
        )
        assert isinstance(returns, list) and len(returns) == 2


# ─── custom RND predictor / target ───────────────────────────────────────────


class TestCustomRNDNetworks:
    def test_predictor_forward_shape(self):
        net = CustomRNDPredictor(4)
        out = net(torch.randn(4, 4))
        assert out.shape == (4, 32)

    def test_predictor_passes_feature_validation(self):
        from neatrl.utils import validate_feature_network_dimensions

        net = CustomRNDPredictor(4)
        validate_feature_network_dimensions(net, obs_dim=4, feature_dim=32)

    def test_incompatible_predictor_fails_feature_validation(self):
        from neatrl.utils import validate_feature_network_dimensions

        net = CustomRNDPredictor(4)
        with pytest.raises(ValueError):
            validate_feature_network_dimensions(net, obs_dim=4, feature_dim=64)


# ─── OneHotWrapper unit tests ─────────────────────────────────────────────────


class TestOneHotWrapper:
    def _make_wrapped_env(self):
        import gymnasium as gym

        from neatrl.dqn_mlp import OneHotWrapper

        env = gym.make("FrozenLake-v1")
        return OneHotWrapper(env, obs_shape=env.observation_space.n)

    def test_obs_space_is_box(self):
        import gymnasium as gym

        env = self._make_wrapped_env()
        assert isinstance(env.observation_space, gym.spaces.Box)
        env.close()

    def test_reset_obs_is_one_hot(self):
        import numpy as np

        env = self._make_wrapped_env()
        obs, _ = env.reset()
        assert obs.sum() == pytest.approx(1.0)
        assert obs.dtype == np.float32
        env.close()

    def test_obs_shape_matches_n(self):
        env = self._make_wrapped_env()
        obs, _ = env.reset()
        assert obs.shape == (16,)
        env.close()
