"""Tier 2 — TUI (Dashboard) tests: construction, rendering, teardown.

These guard the terminal UI that trainers show when stdout is a TTY. It must
construct and render without a real terminal too, because trainers may be driven
from pytest, notebooks or CI.
"""

import io

import pytest


class TestDashboardConstruction:
    def test_constructs_without_tty(self):
        from neatrl.cli.dashboard import Dashboard

        dash = Dashboard("DQN", "CartPole-v1", 1000)
        try:
            assert dash.algo == "DQN"
            assert dash.envId == "CartPole-v1"
            assert dash.totalTimesteps == 1000
        finally:
            dash.close("done")

    def test_no_fd_when_stdin_is_not_a_tty(self, monkeypatch):
        """stdin under pytest has no real descriptor; construction must cope."""
        from neatrl.cli.dashboard import Dashboard

        class NoFileno(io.StringIO):
            def fileno(self):
                raise io.UnsupportedOperation("no fileno")

        monkeypatch.setattr("sys.stdin", NoFileno())
        dash = Dashboard("DQN", "CartPole-v1", 100)
        try:
            assert dash.fd is None
            assert dash.oldTerm is None
        finally:
            dash.close("done")

    def test_accepts_config_object(self):
        from neatrl.cli.dashboard import Dashboard
        from neatrl.dqn_cnn import Config

        dash = Dashboard("DQN-CNN", "ALE/Pong-v5", 1000, config=Config)
        try:
            assert dash.configObj is Config
        finally:
            dash.close("done")


class TestDashboardRendering:
    def test_update_renders_without_crashing(self, capsys):
        from neatrl.cli.dashboard import Dashboard

        dash = Dashboard("DQN", "CartPole-v1", 1000)
        try:
            for step in (0, 100, 500):
                dash.push_metrics(epsilon=0.5, total_loss=0.25, ep_return=12.0)
                dash.update(
                    agent_steps=step,
                    epoch=step,
                    losses={"td_loss": 0.25},
                    eval_stats={"avg_return": 10.0, "last_return": 12.0},
                    message="Q-Net: 1,000 params",
                )
        finally:
            dash.close("done")

        out = capsys.readouterr().out
        assert out, "dashboard produced no output"
        # A frame was actually drawn (ANSI cursor control is used).
        assert "\033[" in out

    def test_each_view_renders(self, capsys):
        """The l / c / v toggles must all render a frame."""
        from neatrl.cli.dashboard import Dashboard

        dash = Dashboard("DQN", "CartPole-v1", 1000)
        try:
            for view in ("showLogs", "showConfig", "showCharts"):
                setattr(dash, view, True)
                dash.isFirst = True
                rendered = dash.renderDash(10, 10, {"td_loss": 0.1}, None, "msg")
                assert isinstance(rendered, str) and rendered
                setattr(dash, view, False)
        finally:
            dash.close("done")

    def test_push_metrics_records_history(self):
        from neatrl.cli.dashboard import Dashboard

        dash = Dashboard("DQN", "CartPole-v1", 1000)
        try:
            dash.push_metrics(ep_return=1.0, total_loss=0.5)
            dash.push_metrics(ep_return=2.0, total_loss=0.4)
            history = dash.history
            assert history.get("ep_return") == [1.0, 2.0]
            assert history.get("total_loss") == [0.5, 0.4]
            assert history.latest("ep_return") == 2.0
            assert history.has("ep_return")
        finally:
            dash.close("done")

    def test_close_is_idempotent(self, capsys):
        from neatrl.cli.dashboard import Dashboard

        dash = Dashboard("DQN", "CartPole-v1", 100)
        dash.close("first")
        # A second close must not raise even though handlers were restored.
        dash.close("second")
        capsys.readouterr()


class TestTrainersSkipDashboardWithoutTty:
    """Trainers must not flood stdout with ANSI frames when stdin/stdout is piped."""

    @pytest.mark.parametrize("module", ["neatrl.dqn_mlp", "neatrl.dqn_cnn"])
    def test_no_dashboard_when_not_a_tty(self, module, monkeypatch, capsys):
        import importlib

        mod = importlib.import_module(module)
        monkeypatch.setattr("sys.stdout.isatty", lambda: False)

        created = []
        original = mod.Dashboard

        def spy(*args, **kwargs):
            created.append(args)
            return original(*args, **kwargs)

        monkeypatch.setattr(mod, "Dashboard", spy)

        if module == "neatrl.dqn_mlp":
            mod.train_dqn(
                env_id="CartPole-v1",
                total_timesteps=120,
                seed=0,
                learning_starts=20,
                buffer_size=200,
                batch_size=16,
                eval_every=0,
                save_every=0,
                use_wandb=False,
                capture_video=False,
                device="cpu",
            )
        else:
            mod.train_dqn_cnn(
                env_id="ALE/Pong-v5",
                total_timesteps=60,
                seed=0,
                learning_starts=10,
                buffer_size=100,
                batch_size=8,
                eval_every=0,
                save_every=0,
                use_wandb=False,
                capture_video=False,
                device="cpu",
            )

        assert created == [], "trainer built a Dashboard with stdout piped"
        out = capsys.readouterr().out
        assert "\033[" not in out, "ANSI dashboard frames leaked into a piped stdout"
