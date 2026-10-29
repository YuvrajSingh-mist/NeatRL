"""CLI tests -- `neatrl train` must work from an installed wheel.

The command used to load ``docs/cli/<algo>.py`` by path, which only exists in
a repository checkout, so every installed release died with "Script not
found". Dispatch now comes from ``neatrl.cli.recipes``; these tests pin that
the recipe table matches the real trainer signatures and that nothing needs
the filesystem.
"""

import inspect
import os
import pathlib
import subprocess
import sys

import pytest

from neatrl.cli import recipes
from neatrl.cli.main import main
from neatrl.cli.recipes import ALIASES, RECIPES, _resolve, default_env, run_train

DOCUMENTED_ALIASES = {
    "a2c",
    "a2c-cnn",
    "ddpg",
    "ddpg-cnn",
    "dqn",
    "dqn-cnn",
    "double-dqn",
    "double-dqn-cnn",
    "dueling-dqn",
    "ppo",
    "ppo-cnn",
    "q-learning",
    "reinforce",
    "reinforce-cnn",
    "rnd",
    "rnd-cnn",
    "sac",
    "sac-cnn",
    "td3",
    "td3-cnn",
}


def _calls():
    for key, recipe in RECIPES.items():
        yield key, "mlp", recipe.mlp
        if recipe.cnn is not None:
            yield key, "cnn", recipe.cnn


class TestAliases:
    def test_alias_set_is_unchanged(self):
        assert set(ALIASES) == DOCUMENTED_ALIASES

    def test_every_alias_maps_to_a_recipe(self):
        for key, use_cnn in ALIASES.values():
            assert key in RECIPES
            if use_cnn:
                assert RECIPES[key].cnn is not None

    def test_cnn_aliases_exist_only_where_a_cnn_trainer_does(self):
        for key, recipe in RECIPES.items():
            alias = key.replace("_", "-")
            assert f"{alias}-cnn" in ALIASES if recipe.cnn else True

    def test_algorithms_without_a_cnn_have_no_cnn_alias(self):
        for key, recipe in RECIPES.items():
            if recipe.cnn is None:
                assert f"{key.replace('_', '-')}-cnn" not in ALIASES


class TestRecipesMatchTrainers:
    def test_every_recipe_targets_a_real_callable(self):
        for key, kind, call in _calls():
            assert callable(_resolve(call.func)), f"{key}/{kind}: {call.func}"

    def test_recipe_kwargs_are_accepted_by_the_trainer(self):
        for key, kind, call in _calls():
            params = set(inspect.signature(_resolve(call.func)).parameters)
            unexpected = set(call.kwargs) - params
            assert not unexpected, f"{key}/{kind} passes {sorted(unexpected)}"

    def test_steps_keyword_exists_on_every_trainer(self):
        for key, kind, call in _calls():
            params = set(inspect.signature(_resolve(call.func)).parameters)
            assert RECIPES[key].steps_kw in params, f"{key}/{kind}"

    def test_cnn_switch_keyword_exists(self):
        for key, recipe in RECIPES.items():
            if recipe.cnn_switch:
                params = set(inspect.signature(_resolve(recipe.mlp.func)).parameters)
                assert recipe.cnn_switch in params, key

    def test_default_env_is_never_empty(self):
        for key, recipe in RECIPES.items():
            assert recipe.mlp.env
            assert default_env(key)
            if recipe.cnn is not None:
                assert default_env(key, use_cnn=True)


class TestDispatch:
    def test_mlp_alias_calls_the_mlp_trainer(self, monkeypatch):
        seen = {}

        def fake_dispatch(dotted):
            seen["func"] = dotted
            return lambda **kw: kw

        monkeypatch.setattr(recipes, "_resolve", fake_dispatch)
        run_train("dqn", "CartPole-v1")

        assert seen["func"] == RECIPES["dqn"].mlp.func

    def test_cnn_alias_calls_the_cnn_trainer(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            recipes,
            "_resolve",
            lambda dotted: lambda **kw: seen.setdefault("func", dotted) or kw,
        )
        run_train("dqn", "ALE/Pong-v5", use_cnn=True)

        assert seen["func"] == RECIPES["dqn"].cnn.func

    def test_explicit_env_overrides_the_default(self, monkeypatch):
        monkeypatch.setattr(recipes, "_resolve", lambda dotted: lambda **kw: kw)
        out = run_train("dqn", "LunarLander-v3")
        assert out["env_id"] == "LunarLander-v3"

    def test_missing_env_falls_back_to_the_recipe_default(self, monkeypatch):
        monkeypatch.setattr(recipes, "_resolve", lambda dotted: lambda **kw: kw)
        out = run_train("dqn", None)
        assert out["env_id"] == RECIPES["dqn"].mlp.env

    def test_steps_use_total_timesteps_by_default(self, monkeypatch):
        monkeypatch.setattr(recipes, "_resolve", lambda dotted: lambda **kw: kw)
        out = run_train("dqn", "CartPole-v1", steps=500)
        assert out["total_timesteps"] == 500

    def test_steps_use_total_episodes_for_reinforce(self, monkeypatch):
        monkeypatch.setattr(recipes, "_resolve", lambda dotted: lambda **kw: kw)
        out = run_train("reinforce", "CartPole-v1", steps=5)
        assert out["total_episodes"] == 5
        assert "total_timesteps" not in out

    def test_cnn_switch_turns_on_atari_wrapper(self, monkeypatch):
        monkeypatch.setattr(recipes, "_resolve", lambda dotted: lambda **kw: kw)
        out = run_train("dueling_dqn", "ALE/Pong-v5", use_cnn=True)
        assert out["atari_wrapper"] is True


class TestOptionalRichDependency:
    """`rich` is the optional `cli` extra, so a bare install must still import."""

    def test_import_neatrl_succeeds_without_rich(self, tmp_path):
        # A stub `rich` earlier on sys.path makes the import fail even on a
        # machine that has the extra installed.
        (tmp_path / "rich.py").write_text("raise ImportError('rich is not installed')")
        pkg_root = str(pathlib.Path(__file__).resolve().parent.parent / "src")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp_path), pkg_root]))
        code = (
            "import neatrl, sys\n"
            "from neatrl.cli import Dashboard\n"
            "assert Dashboard is None, Dashboard\n"
            "assert 'rich' not in sys.modules\n"
            "print('ok', neatrl.__version__)\n"
        )
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            env=env,
        )
        assert out.returncode == 0, out.stderr
        assert out.stdout.startswith("ok")

    def test_cli_help_works_without_rich(self, tmp_path):
        (tmp_path / "rich.py").write_text("raise ImportError('rich is not installed')")
        pkg_root = str(pathlib.Path(__file__).resolve().parent.parent / "src")
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(tmp_path), pkg_root]))
        out = subprocess.run(
            [sys.executable, "-m", "neatrl.cli", "--help"],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            env=env,
        )
        assert out.returncode == 0, out.stderr
        assert "train" in out.stdout


class TestCommandLine:
    def test_train_with_only_an_algo_uses_the_default_env(
        self, monkeypatch, tmp_path, capsys
    ):
        captured = {}
        monkeypatch.setattr(
            "neatrl.cli.main.run_train",
            lambda key, env, **kw: captured.update(key=key, env=env, **kw),
        )
        monkeypatch.chdir(tmp_path)  # no docs/ here: the old loader would fail
        main(["train", "q-learning"])

        assert captured["key"] == "q_learning"
        assert captured["env"] == default_env("q_learning")
        assert captured["use_cnn"] is False

    def test_explicit_env_is_forwarded(self, monkeypatch, tmp_path):
        captured = {}
        monkeypatch.setattr(
            "neatrl.cli.main.run_train",
            lambda key, env, **kw: captured.update(key=key, env=env, **kw),
        )
        monkeypatch.chdir(tmp_path)
        main(["train", "ppo", "LunarLander-v3", "--steps", "1000"])

        assert captured["env"] == "LunarLander-v3"
        assert captured["steps"] == 1000

    def test_cnn_alias_sets_use_cnn(self, monkeypatch, tmp_path):
        captured = {}
        monkeypatch.setattr(
            "neatrl.cli.main.run_train",
            lambda key, env, **kw: captured.update(key=key, env=env, **kw),
        )
        monkeypatch.chdir(tmp_path)
        main(["train", "td3-cnn", "CarRacing-v3"])

        assert captured["use_cnn"] is True

    def test_no_subcommand_prints_help_and_exits_cleanly(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            main([])
        assert excinfo.value.code == 0
        assert "train" in capsys.readouterr().out

    def test_unknown_algorithm_is_rejected(self):
        with pytest.raises(SystemExit) as excinfo:
            main(["train", "not-an-algorithm"])
        assert excinfo.value.code == 2

    def test_module_entry_point_runs_outside_the_repo(self, tmp_path):
        # No docs/ above the cwd, so the old path-based loader would fail here.
        pkg_root = str(pathlib.Path(__file__).resolve().parent.parent / "src")
        env = dict(os.environ, PYTHONPATH=pkg_root)
        out = subprocess.run(
            [sys.executable, "-m", "neatrl.cli", "--help"],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            env=env,
        )
        assert out.returncode == 0, out.stderr
        assert "train" in out.stdout
