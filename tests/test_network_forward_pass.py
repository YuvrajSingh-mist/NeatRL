"""Tier 2 — network forward-pass shape and sanity tests. No training required."""

import torch

DEVICE = torch.device("cpu")
BATCH = 4

# CartPole-v1 dims (discrete control)
OBS_DISC = 4  # Box(4,)
ACT_DISC = 2  # Discrete(2)

# Pendulum-v1 dims (continuous control)
OBS_CONT = 3  # Box(3,)
ACT_CONT = 1  # Box(1,)


def _rand(obs_dim: int) -> torch.Tensor:
    return torch.randn(BATCH, obs_dim)


def _no_nan(t: torch.Tensor) -> bool:
    return not (torch.isnan(t).any() or torch.isinf(t).any())


# ─── DQN ─────────────────────────────────────────────────────────────────────


class TestDQNNetworks:
    def test_qnet_output_shape(self):
        from neatrl.dqn_mlp import QNet

        net = QNet(OBS_DISC, ACT_DISC)
        out = net(torch.randn(BATCH, OBS_DISC))
        assert out.shape == (BATCH, ACT_DISC)

    def test_qnet_no_nan(self):
        from neatrl.dqn_mlp import QNet

        net = QNet(OBS_DISC, ACT_DISC)
        assert _no_nan(net(torch.randn(BATCH, OBS_DISC)))


# ─── Dueling DQN ─────────────────────────────────────────────────────────────


class TestDuelingDQNNetworks:
    def test_dueling_qnet_output_shape(self):
        from neatrl.dueling_dqn_mlp import DuelingQNet

        net = DuelingQNet(OBS_DISC, ACT_DISC)
        q_values, values, adv, feat = net(torch.randn(BATCH, OBS_DISC))
        assert q_values.shape == (BATCH, ACT_DISC)
        assert values.shape == (BATCH, 1)
        assert adv.shape == (BATCH, ACT_DISC)

    def test_dueling_qnet_no_nan(self):
        from neatrl.dueling_dqn_mlp import DuelingQNet

        net = DuelingQNet(OBS_DISC, ACT_DISC)
        q_values, _, _, _ = net(torch.randn(BATCH, OBS_DISC))
        assert _no_nan(q_values)

    def test_dueling_qnet_has_adv_values_attrs(self):
        import torch.nn as nn

        from neatrl.dueling_dqn_mlp import DuelingQNet

        net = DuelingQNet(OBS_DISC, ACT_DISC)
        assert hasattr(net, "adv") and isinstance(net.adv, nn.Module)
        assert hasattr(net, "values") and isinstance(net.values, nn.Module)


# ─── PPO ─────────────────────────────────────────────────────────────────────


class TestPPONetworks:
    def test_actor_get_action_shapes(self):
        from neatrl.ppo_mlp import ActorNet

        net = ActorNet(OBS_DISC, ACT_DISC)
        action, log_prob, dist = net.get_action(torch.randn(BATCH, OBS_DISC))
        assert action.shape == (BATCH,)
        assert log_prob.shape == (BATCH,)

    def test_critic_output_shape(self):
        from neatrl.ppo_mlp import CriticNet

        net = CriticNet(OBS_DISC)
        out = net(torch.randn(BATCH, OBS_DISC))
        assert out.shape == (BATCH, 1)

    def test_actor_no_nan(self):
        from neatrl.ppo_mlp import ActorNet

        net = ActorNet(OBS_DISC, ACT_DISC)
        action, log_prob, _ = net.get_action(torch.randn(BATCH, OBS_DISC))
        assert _no_nan(action) and _no_nan(log_prob)

    def test_critic_no_nan(self):
        from neatrl.ppo_mlp import CriticNet

        net = CriticNet(OBS_DISC)
        assert _no_nan(net(torch.randn(BATCH, OBS_DISC)))


# ─── A2C ─────────────────────────────────────────────────────────────────────


class TestA2CNetworks:
    def test_actor_get_action_shapes(self):
        from neatrl.a2c_mlp import ActorNet

        net = ActorNet(OBS_DISC, ACT_DISC)
        action, log_prob, dist = net.get_action(torch.randn(BATCH, OBS_DISC))
        assert action.shape == (BATCH,)
        assert log_prob.shape == (BATCH,)

    def test_critic_output_shape(self):
        from neatrl.a2c_mlp import CriticNet

        net = CriticNet(OBS_DISC)
        out = net(torch.randn(BATCH, OBS_DISC))
        assert out.shape == (BATCH, 1)


# ─── REINFORCE ───────────────────────────────────────────────────────────────


class TestREINFORCENetworks:
    def test_policy_output_shape(self):
        from neatrl.reinforce_mlp import PolicyNet

        net = PolicyNet(OBS_DISC, ACT_DISC)
        logits = net(torch.randn(BATCH, OBS_DISC))
        assert logits.shape == (BATCH, ACT_DISC)

    def test_policy_no_nan(self):
        from neatrl.reinforce_mlp import PolicyNet

        net = PolicyNet(OBS_DISC, ACT_DISC)
        assert _no_nan(net(torch.randn(BATCH, OBS_DISC)))


# ─── DDPG ────────────────────────────────────────────────────────────────────


class TestDDPGNetworks:
    def test_actor_output_shape(self):
        from neatrl.ddpg_mlp import ActorNet

        net = ActorNet(OBS_CONT, ACT_CONT)
        out = net(torch.randn(BATCH, OBS_CONT))
        assert out.shape == (BATCH, ACT_CONT)

    def test_qnet_output_shape(self):
        from neatrl.ddpg_mlp import QNet

        net = QNet(OBS_CONT, ACT_CONT)
        out = net(torch.randn(BATCH, OBS_CONT), torch.randn(BATCH, ACT_CONT))
        assert out.shape == (BATCH, 1)

    def test_actor_no_nan(self):
        from neatrl.ddpg_mlp import ActorNet

        net = ActorNet(OBS_CONT, ACT_CONT)
        assert _no_nan(net(torch.randn(BATCH, OBS_CONT)))

    def test_qnet_no_nan(self):
        from neatrl.ddpg_mlp import QNet

        net = QNet(OBS_CONT, ACT_CONT)
        assert _no_nan(net(torch.randn(BATCH, OBS_CONT), torch.randn(BATCH, ACT_CONT)))


# ─── TD3 ─────────────────────────────────────────────────────────────────────


class TestTD3Networks:
    def test_actor_output_shape(self):
        from neatrl.td3_mlp import ActorNet

        net = ActorNet(OBS_CONT, ACT_CONT)
        out = net(torch.randn(BATCH, OBS_CONT))
        assert out.shape == (BATCH, ACT_CONT)

    def test_qnet_output_shape(self):
        from neatrl.td3_mlp import QNet

        net = QNet(OBS_CONT, ACT_CONT)
        out = net(torch.randn(BATCH, OBS_CONT), torch.randn(BATCH, ACT_CONT))
        assert out.shape == (BATCH, 1)


# ─── SAC ─────────────────────────────────────────────────────────────────────


class TestSACNetworks:
    def test_actor_get_action_shapes(self):
        from neatrl.sac_mlp import ActorNet

        net = ActorNet(OBS_CONT, ACT_CONT)
        action, log_prob = net.get_action(torch.randn(BATCH, OBS_CONT))
        assert action.shape == (BATCH, ACT_CONT)
        assert log_prob.shape == (BATCH, 1)

    def test_qnet_output_shape(self):
        from neatrl.sac_mlp import QNet

        net = QNet(OBS_CONT, ACT_CONT)
        out = net(torch.randn(BATCH, OBS_CONT), torch.randn(BATCH, ACT_CONT))
        assert out.shape == (BATCH, 1)

    def test_actor_no_nan(self):
        from neatrl.sac_mlp import ActorNet

        net = ActorNet(OBS_CONT, ACT_CONT)
        action, log_prob = net.get_action(torch.randn(BATCH, OBS_CONT))
        assert _no_nan(action) and _no_nan(log_prob)


# ─── RND ─────────────────────────────────────────────────────────────────────


class TestRNDNetworks:
    def test_actor_get_action_shapes(self):
        from neatrl.rnd_mlp import ActorNet

        net = ActorNet(OBS_DISC, ACT_DISC)
        action, log_prob, dist = net.get_action(torch.randn(BATCH, OBS_DISC))
        assert action.shape == (BATCH,)
        assert log_prob.shape == (BATCH,)

    def test_critic_output_shapes(self):
        from neatrl.rnd_mlp import CriticNet

        net = CriticNet(OBS_DISC)
        ext_val, int_val = net(torch.randn(BATCH, OBS_DISC))
        assert ext_val.shape == (BATCH, 1)
        assert int_val.shape == (BATCH, 1)

    def test_predictor_output_shape(self):
        from neatrl.rnd_mlp import PredictorNet

        net = PredictorNet(OBS_DISC)
        out = net(torch.randn(BATCH, OBS_DISC))
        assert out.shape == (BATCH, 32)

    def test_target_output_shape(self):
        from neatrl.rnd_mlp import TargetNet

        net = TargetNet(OBS_DISC)
        out = net(torch.randn(BATCH, OBS_DISC))
        assert out.shape == (BATCH, 32)

    def test_predictor_target_same_output_dim(self):
        from neatrl.rnd_mlp import PredictorNet, TargetNet

        obs = torch.randn(BATCH, OBS_DISC)
        pred_out = PredictorNet(OBS_DISC)(obs)
        tgt_out = TargetNet(OBS_DISC)(obs)
        assert pred_out.shape == tgt_out.shape


# ─── RunningMeanStd (RND) ────────────────────────────────────────────────────


class TestRunningMeanStd:
    def test_update_changes_mean(self):
        from neatrl.rnd_mlp import RunningMeanStd

        rms = RunningMeanStd()
        import numpy as np

        rms.update(np.array([1.0, 2.0, 3.0]))
        assert rms.mean != 0.0

    def test_update_from_moments(self):
        from neatrl.rnd_mlp import RunningMeanStd

        rms = RunningMeanStd()
        rms.update_from_moments(batch_mean=5.0, batch_var=1.0, batch_count=10)
        assert rms.count > 1e-4  # count was updated
