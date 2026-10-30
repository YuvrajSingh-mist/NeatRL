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
    "dueling-dqn-cnn",
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


class TestEnvRouting:
    """Image envs and MLP aliases are a mismatch that used to fail in torch."""

    def test_image_env_routes_to_the_cnn_trainer(self):
        call, used_cnn = recipes.resolve_call("dqn", "ALE/Pong-v5", False)
        assert used_cnn is True
        assert call.func == "neatrl.dqn_cnn.train_dqn_cnn"

    def test_carracing_routes_to_a_carracing_cnn_trainer(self):
        call, used_cnn = recipes.resolve_call("td3", "CarRacing-v3", False)
        assert used_cnn is True
        assert call.func == "neatrl.td3_cnn.train_td3_cnn"

    def test_atari_env_is_not_sent_to_a_carracing_trainer(self):
        with pytest.raises(ValueError, match="preprocessing"):
            recipes.resolve_call("td3", "ALE/Pong-v5", False)

    def test_familyless_algorithm_rejects_image_env(self):
        with pytest.raises(ValueError, match="no CNN trainer"):
            recipes.resolve_call("q_learning", "ALE/Pong-v5", False)

    def test_cnn_alias_rejects_a_vector_env(self):
        with pytest.raises(ValueError, match="expects atari image"):
            recipes.resolve_call("dqn", "CartPole-v1", True)

    def test_every_call_declares_a_family(self):
        for key, recipe in RECIPES.items():
            assert recipe.mlp.family == "vector", key
            if recipe.cnn is not None:
                assert recipe.cnn.family in ("atari", "carracing"), key

    def test_cli_reports_a_mismatch_without_a_traceback(self, capsys):
        with pytest.raises(SystemExit) as excinfo:
            main(["train", "q-learning", "ALE/Pong-v5"])
        assert excinfo.value.code == 2
        assert "no CNN trainer" in capsys.readouterr().err


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


def test_version_flag_reports_the_installed_version(capsys):
    """`neatrl --version` is how an upgrade is checked; it must not train."""
    import pytest

    from neatrl import __version__
    from neatrl.cli.main import main

    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert __version__ in capsys.readouterr().out


# ─── the CLI is documented in markdown, not in example scripts ───────────────


def test_the_cli_docs_exist():
    """`docs/cli/` used to hold 11 example .py files that no longer run.

    The command loads its recipe table from the package, so the folder is
    documentation and should say so -- with the usage that used to be implicit
    in those scripts.
    """
    from pathlib import Path

    docs = Path(__file__).resolve().parents[1] / "docs" / "cli"
    assert docs.is_dir(), "docs/cli/ is missing"
    readme = docs / "README.md"
    assert readme.is_file(), "docs/cli/ has no README.md"

    text = readme.read_text(encoding="utf-8")
    assert "neatrl train" in text
    assert "--steps" in text and "--wandb" in text


def test_the_cli_docs_list_every_alias():
    """A new alias must be documented, or the pages silently go stale."""
    from pathlib import Path

    from neatrl.cli.recipes import ALIASES

    docs = Path(__file__).resolve().parents[1] / "docs" / "cli"
    text = "\n".join(p.read_text(encoding="utf-8") for p in docs.glob("*.md"))

    missing = [alias for alias in ALIASES if f"`{alias}`" not in text]
    assert not missing, f"undocumented aliases: {missing}"


def test_every_algorithm_has_its_own_cli_page():
    """One markdown page per recipe, named after the recipe key."""
    from pathlib import Path

    from neatrl.cli.recipes import RECIPES

    docs = Path(__file__).resolve().parents[1] / "docs" / "cli"
    missing = [key for key in RECIPES if not (docs / f"{key}.md").is_file()]
    assert not missing, f"no CLI page for: {missing}"


def test_each_algorithm_page_states_its_trainer_and_default_env():
    """These are the two things a reader comes to the page for."""
    from pathlib import Path

    from neatrl.cli.recipes import RECIPES

    docs = Path(__file__).resolve().parents[1] / "docs" / "cli"
    for key, recipe in RECIPES.items():
        page = (docs / f"{key}.md").read_text(encoding="utf-8")
        assert recipe.mlp.env in page, f"{key}.md never names {recipe.mlp.env}"
        assert recipe.mlp.func in page, f"{key}.md never names {recipe.mlp.func}"
        if recipe.cnn is not None and recipe.cnn.func != recipe.mlp.func:
            assert recipe.cnn.func in page, f"{key}.md never names {recipe.cnn.func}"


def test_no_stale_example_scripts_in_the_cli_docs():
    """The folder documents the CLI; runnable examples belong in docs/non-cli/."""
    from pathlib import Path

    docs = Path(__file__).resolve().parents[1] / "docs" / "cli"
    scripts = sorted(p.name for p in docs.glob("*.py"))
    assert not scripts, (
        f"{scripts} in docs/cli/: the CLI does not load these, and programmatic "
        "examples live in docs/non-cli/"
    )


def test_the_cli_doc_links_resolve():
    """A dead relative link is invisible until someone clicks it on GitHub."""
    import re
    from pathlib import Path

    docs = Path(__file__).resolve().parents[1] / "docs" / "cli"
    broken: list[str] = []

    for page in docs.glob("*.md"):
        for target in re.findall(r"\]\(([^)]+)\)", page.read_text(encoding="utf-8")):
            link = target.split("#", 1)[0]
            if not link or "://" in link or link.startswith("mailto:"):
                continue
            if not (page.parent / link).exists():
                broken.append(f"{page.name} -> {target}")

    assert not broken, f"broken links: {broken}"
