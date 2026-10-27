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


# ─── DQN (CNN / Atari) ───────────────────────────────────────────────────────

OBS_IMAGE = (4, 84, 84)  # 4 stacked 84x84 grayscale frames


class TestDQNCNNNetworks:
    def test_qnet_cnn_output_shape_uint8(self):
        from neatrl.dqn_cnn import QNetCNN

        net = QNetCNN(OBS_IMAGE, ACT_DISC)
        obs = torch.zeros((BATCH, *OBS_IMAGE), dtype=torch.uint8)
        assert net(obs).shape == (BATCH, ACT_DISC)

    def test_qnet_cnn_accepts_float_input(self):
        from neatrl.dqn_cnn import QNetCNN

        net = QNetCNN(OBS_IMAGE, ACT_DISC)
        out = net(torch.rand(BATCH, *OBS_IMAGE))
        assert out.shape == (BATCH, ACT_DISC)
        assert _no_nan(out)

    def test_qnet_cnn_infers_other_screen_sizes(self):
        from neatrl.dqn_cnn import QNetCNN

        net = QNetCNN((4, 96, 96), ACT_DISC)
        out = net(torch.zeros((BATCH, 4, 96, 96), dtype=torch.uint8))
        assert out.shape == (BATCH, ACT_DISC)

    def test_qnet_cnn_no_nan(self):
        from neatrl.dqn_cnn import QNetCNN

        net = QNetCNN(OBS_IMAGE, ACT_DISC)
        obs = torch.randint(0, 255, (BATCH, *OBS_IMAGE), dtype=torch.uint8)
        assert _no_nan(net(obs))

    def test_qnet_cnn_uint8_and_float_match_after_scaling(self):
        from neatrl.dqn_cnn import QNetCNN

        net = QNetCNN(OBS_IMAGE, ACT_DISC).eval()
        obs = torch.randint(0, 255, (1, *OBS_IMAGE), dtype=torch.uint8)
        with torch.no_grad():
            from_uint8 = net(obs)
            from_float = net(obs.float() / 255.0)
        assert torch.allclose(from_uint8, from_float, atol=1e-5)


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


# ─── Double DQN target ───────────────────────────────────────────────────────


def _zeroed_qnet(obs_dim: int, act_dim: int, q_values: list[float]):
    """Build a QNet whose Q-values are pinned to ``q_values`` for any input.

    All weights are zero and the second hidden layer is biased to one, so the
    head sees an all-ones feature vector and action ``i`` scores ``q_values[i]``.
    """
    from neatrl.double_dqn_mlp import QNet

    net = QNet(obs_dim, act_dim)
    for param in net.parameters():
        param.data.zero_()
    net.fc2.bias.data.fill_(1.0)
    for action, value in enumerate(q_values):
        net.q_value.weight.data[action, 0] = value
    return net


class TestDoubleDQNTarget:
    def test_matches_manual_double_target(self):
        from neatrl.double_dqn_mlp import QNet, double_dqn_target

        torch.manual_seed(0)
        online = QNet(OBS_DISC, ACT_DISC)
        torch.manual_seed(1)
        target = QNet(OBS_DISC, ACT_DISC)

        next_obs = torch.randn(BATCH, OBS_DISC)
        rewards = torch.randn(BATCH, 1)
        dones = torch.zeros(BATCH, 1)

        out = double_dqn_target(online, target, next_obs, rewards, dones, 0.99)
        best = online(next_obs).argmax(dim=1, keepdim=True)
        expected = rewards.flatten() + 0.99 * target(next_obs).gather(1, best).squeeze(
            1
        )

        assert torch.allclose(out, expected)
        assert out.shape == (BATCH,)
        assert not out.requires_grad

    def test_selects_with_online_net_and_evaluates_with_target_net(self):
        # Online prefers action 0, target scores action 1 much higher, so the
        # double target must ignore the target's max.
        from neatrl.double_dqn_mlp import double_dqn_target

        online = _zeroed_qnet(1, 2, [1.0, 0.0])
        target = _zeroed_qnet(1, 2, [0.0, 5.0])

        next_obs = torch.tensor([[1.0]])
        rewards = torch.zeros(1, 1)
        dones = torch.zeros(1, 1)

        double_target = double_dqn_target(online, target, next_obs, rewards, dones, 0.5)
        plain_target = 0.5 * target(next_obs).max(dim=1)[0]

        assert torch.allclose(double_target, torch.zeros(1))
        assert double_target.item() < plain_target.item()

    def test_terminal_transitions_do_not_bootstrap(self):
        from neatrl.double_dqn_mlp import QNet, double_dqn_target

        online = QNet(OBS_DISC, ACT_DISC)
        target = QNet(OBS_DISC, ACT_DISC)
        rewards = torch.randn(BATCH, 1)
        dones = torch.ones(BATCH, 1)

        out = double_dqn_target(
            online, target, torch.randn(BATCH, OBS_DISC), rewards, dones, 0.99
        )
        assert torch.allclose(out, rewards.flatten())

    def test_cnn_variant_shapes(self):
        from neatrl.double_dqn_cnn import QNetCNN, double_dqn_target

        online = QNetCNN(OBS_IMAGE, ACT_DISC)
        target = QNetCNN(OBS_IMAGE, ACT_DISC)
        obs = torch.zeros((BATCH, *OBS_IMAGE), dtype=torch.uint8)
        rewards = torch.ones(BATCH, 1)
        dones = torch.zeros(BATCH, 1)

        out = double_dqn_target(online, target, obs, rewards, dones, 0.99)
        assert out.shape == (BATCH,)
        assert _no_nan(out)
