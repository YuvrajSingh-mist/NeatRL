"""Tier 1 unit tests for neatrl.utils — all fast, no training."""

import logging

import gymnasium as gym
import pytest
import torch
import torch.nn as nn

from neatrl.utils import (
    calculate_param_norm,
    configure_logging,
    get_logger,
    get_space_dims,
    setup_device,
    validate_critic_network_dimensions,
    validate_dueling_q_network_dimensions,
    validate_feature_network_dimensions,
    validate_policy_network_dimensions,
    validate_q_network_dimensions,
)

# ─── helpers ─────────────────────────────────────────────────────────────────


def _seq(in_f, out_f):
    """Minimal Sequential with one hidden layer."""
    return nn.Sequential(nn.Linear(in_f, 64), nn.ReLU(), nn.Linear(64, out_f))


# ─── get_space_dims ──────────────────────────────────────────────────────────


class TestGetSpaceDims:
    def test_box_obs_discrete_act(self):
        env = gym.make("CartPole-v1")
        obs_dim, act_dim = get_space_dims(env)
        assert obs_dim == 4
        assert act_dim == 2
        env.close()

    def test_box_obs_box_act(self):
        env = gym.make("Pendulum-v1")
        obs_dim, act_dim = get_space_dims(env)
        assert obs_dim == 3
        assert act_dim == 1
        env.close()

    def test_sync_vector_env(self):
        envs = gym.make_vec("CartPole-v1", num_envs=2)
        obs_dim, act_dim = get_space_dims(envs)
        assert obs_dim == 4
        assert act_dim == 2
        envs.close()

    def test_returns_plain_ints(self):
        env = gym.make("CartPole-v1")
        obs_dim, act_dim = get_space_dims(env)
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        env.close()

    def test_continuous_vector_env(self):
        envs = gym.make_vec("Pendulum-v1", num_envs=4)
        obs_dim, act_dim = get_space_dims(envs)
        assert obs_dim == 3
        assert act_dim == 1
        envs.close()


# ─── configure_logging ───────────────────────────────────────────────────────


class TestConfigureLogging:
    def test_idempotent_handler_count(self):
        """Calling configure_logging multiple times must not accumulate handlers."""
        root = logging.getLogger()
        configure_logging()
        count_after_one = len(root.handlers)
        configure_logging()
        configure_logging()
        assert len(root.handlers) == count_after_one

    def test_creates_log_file(self, tmp_path, monkeypatch):
        root = logging.getLogger()
        # Temporarily clear handlers so configure_logging adds a fresh file handler
        saved = list(root.handlers)
        monkeypatch.setattr(root, "handlers", [])
        try:
            configure_logging(log_dir=str(tmp_path))
            assert (tmp_path / "training.log").exists()
        finally:
            for h in root.handlers:
                h.close()
            monkeypatch.setattr(root, "handlers", saved)

    def test_at_least_one_handler_after_call(self):
        root = logging.getLogger()
        configure_logging()
        assert len(root.handlers) >= 1


# ─── setup_device ────────────────────────────────────────────────────────────


class TestSetupDevice:
    def test_cpu_string(self):
        dev = setup_device("cpu")
        assert isinstance(dev, torch.device)
        assert dev.type == "cpu"

    def test_accepts_torch_device_object(self):
        dev = setup_device(torch.device("cpu"))
        assert dev.type == "cpu"

    def test_cuda_falls_back_when_unavailable(self):
        if torch.cuda.is_available():
            pytest.skip("CUDA present — fallback not triggered")
        dev = setup_device("cuda")
        assert dev.type == "cpu"

    def test_returns_torch_device_type(self):
        dev = setup_device("cpu")
        assert isinstance(dev, torch.device)


# ─── calculate_param_norm ────────────────────────────────────────────────────


class TestCalculateParamNorm:
    def test_known_exact_value(self):
        # weight=[[3.0]], bias=[4.0] → sqrt(3² + 4²) = 5.0
        layer = nn.Linear(1, 1)
        with torch.no_grad():
            layer.weight.fill_(3.0)
            layer.bias.fill_(4.0)
        assert abs(calculate_param_norm(layer) - 5.0) < 1e-5

    def test_all_zeros_gives_zero(self):
        layer = nn.Linear(4, 4)
        with torch.no_grad():
            for p in layer.parameters():
                p.fill_(0.0)
        assert calculate_param_norm(layer) == 0.0

    def test_returns_float(self):
        assert isinstance(calculate_param_norm(nn.Linear(4, 2)), float)

    def test_positive_for_random_init(self):
        assert calculate_param_norm(nn.Linear(8, 4)) > 0.0


# ─── validate_q_network_dimensions ──────────────────────────────────────────


class TestValidateQNetworkDimensions:
    def test_correct_dims_no_error(self):
        validate_q_network_dimensions(_seq(4, 2), obs_dim=4, action_dim=2)

    def test_wrong_obs_dim_raises(self):
        with pytest.raises(ValueError, match="input dimension"):
            validate_q_network_dimensions(_seq(4, 2), obs_dim=8, action_dim=2)

    def test_wrong_action_dim_raises(self):
        with pytest.raises(ValueError, match="output dimension"):
            validate_q_network_dimensions(_seq(4, 2), obs_dim=4, action_dim=5)


# ─── validate_dueling_q_network_dimensions ───────────────────────────────────


class TestValidateDuelingQNetworkDimensions:
    def _make_dueling(self, obs_dim, act_dim):
        from neatrl.dueling_dqn_mlp import DuelingQNet

        return DuelingQNet(obs_dim, act_dim)

    def test_correct_dims_no_error(self):
        net = self._make_dueling(4, 2)
        validate_dueling_q_network_dimensions(net, obs_dim=4, action_dim=2)

    def test_wrong_obs_dim_raises(self):
        net = self._make_dueling(4, 2)
        with pytest.raises(ValueError):
            validate_dueling_q_network_dimensions(net, obs_dim=8, action_dim=2)

    def test_wrong_action_dim_raises(self):
        net = self._make_dueling(4, 2)
        with pytest.raises(ValueError):
            validate_dueling_q_network_dimensions(net, obs_dim=4, action_dim=5)


# ─── validate_policy_network_dimensions ──────────────────────────────────────


class TestValidatePolicyNetworkDimensions:
    def test_correct_dims_no_error(self):
        validate_policy_network_dimensions(_seq(4, 2), obs_dim=4, action_dim=2)

    def test_wrong_obs_dim_raises(self):
        with pytest.raises(ValueError):
            validate_policy_network_dimensions(_seq(4, 2), obs_dim=8, action_dim=2)

    def test_wrong_action_dim_raises(self):
        with pytest.raises(ValueError):
            validate_policy_network_dimensions(_seq(4, 2), obs_dim=4, action_dim=5)

    def test_tuple_obs_dim_no_error_for_mlp(self):
        # tuple obs_dim triggers the conv-layer check path; no ValueError for missing conv
        validate_policy_network_dimensions(_seq(4, 2), obs_dim=(4,), action_dim=2)


# ─── validate_critic_network_dimensions ──────────────────────────────────────


class TestValidateCriticNetworkDimensions:
    def test_correct_dims_no_error(self):
        # critic: obs → hidden → 1
        net = _seq(4, 1)
        validate_critic_network_dimensions(net, obs_dim=4)

    def test_wrong_obs_dim_raises(self):
        net = _seq(4, 1)
        with pytest.raises(ValueError):
            validate_critic_network_dimensions(net, obs_dim=8)

    def test_wrong_output_dim_raises(self):
        # critic output must be 1
        net = _seq(4, 2)
        with pytest.raises(ValueError):
            validate_critic_network_dimensions(net, obs_dim=4)


# ─── validate_feature_network_dimensions ─────────────────────────────────────


class TestValidateFeatureNetworkDimensions:
    def test_correct_dims_no_error(self):
        validate_feature_network_dimensions(_seq(4, 32), obs_dim=4, feature_dim=32)

    def test_wrong_obs_dim_raises(self):
        with pytest.raises(ValueError):
            validate_feature_network_dimensions(_seq(4, 32), obs_dim=8, feature_dim=32)

    def test_wrong_feature_dim_raises(self):
        with pytest.raises(ValueError):
            validate_feature_network_dimensions(_seq(4, 32), obs_dim=4, feature_dim=64)


# ─── get_logger ──────────────────────────────────────────────────────────────


def test_get_logger_returns_logger_instance():
    logger = get_logger("test.neatrl")
    assert isinstance(logger, logging.Logger)
    assert logger.name == "test.neatrl"


def test_get_logger_same_name_same_instance():
    assert get_logger("same") is get_logger("same")


# ─── layer_init ──────────────────────────────────────────────────────────────


class TestLayerInit:
    def test_returns_same_layer(self):
        from neatrl.ppo_mlp import layer_init

        layer = nn.Linear(4, 2)
        result = layer_init(layer)
        assert result is layer

    def test_default_bias_is_zero(self):
        from neatrl.ppo_mlp import layer_init

        layer = layer_init(nn.Linear(4, 2))
        assert torch.all(layer.bias == 0.0)

    def test_custom_bias_const(self):
        from neatrl.ppo_mlp import layer_init

        layer = layer_init(nn.Linear(4, 2), bias_const=1.0)
        assert torch.all(layer.bias == 1.0)

    def test_weight_is_not_default_init(self):
        from neatrl.ppo_mlp import layer_init

        torch.manual_seed(0)
        default_layer = nn.Linear(8, 4)
        default_weight = default_layer.weight.data.clone()

        torch.manual_seed(0)
        ortho_layer = layer_init(nn.Linear(8, 4))
        # Orthogonal init should differ from default Kaiming uniform
        assert not torch.allclose(ortho_layer.weight.data, default_weight)
