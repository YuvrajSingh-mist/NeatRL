"""The CLI must turn a missing environment dependency into an install hint.

``neatrl train dqn-cnn`` defaults to ``ALE/Pong-v5``; without ``ale-py``
gymnasium raises ``NamespaceNotFound``, which reads like a bug in this package
rather than a missing extra. The original traceback is kept and one install
command is added.
"""

from __future__ import annotations

import pytest

from neatrl.cli.extras import ExtraHint, format_hint, missing_extra


def _namespace_error(namespace: str):
    """Build the exception gymnasium raises for an unregistered namespace."""
    gymnasium = pytest.importorskip("gymnasium")
    error = getattr(gymnasium.error, "NamespaceNotFound", None)
    if error is None:  # older gymnasium
        pytest.skip("gymnasium has no NamespaceNotFound")
    return error(
        f"Namespace {namespace} not found. Have you installed the "
        f"proper package for {namespace}?"
    )


def test_ale_namespace_error_points_at_the_atari_extra():
    hint = missing_extra(_namespace_error("ALE"))
    assert hint is not None
    assert hint.extra == "atari"


def test_missing_ale_py_import_points_at_the_atari_extra():
    hint = missing_extra(ModuleNotFoundError("No module named 'ale_py'"))
    assert hint.extra == "atari"


def test_missing_box2d_points_at_the_box2d_extra():
    hint = missing_extra(ModuleNotFoundError("No module named 'Box2D'"))
    assert hint.extra == "box2d"


def test_missing_mujoco_points_at_the_mujoco_extra():
    hint = missing_extra(ModuleNotFoundError("No module named 'mujoco'"))
    assert hint.extra == "mujoco"


def test_hint_is_found_through_the_exception_chain():
    """gymnasium wraps the import error, so the cause chain must be searched."""
    try:
        try:
            raise ModuleNotFoundError("No module named 'ale_py'")
        except ModuleNotFoundError as inner:
            raise RuntimeError("while registering envs") from inner
    except RuntimeError as outer:
        hint = missing_extra(outer)
    assert hint is not None and hint.extra == "atari"


def test_unrelated_errors_are_not_claimed():
    assert missing_extra(RuntimeError("CUDA out of memory")) is None
    assert missing_extra(ValueError("bad shape")) is None


def test_hint_message_names_the_command_to_run():
    text = format_hint(
        ExtraHint(extra="atari", package="ale-py", unlocks="Atari games")
    )
    assert 'pip install "neatrl[atari]"' in text
    assert "ale-py" in text
    # uv users get the equivalent, since the tool install is the documented path.
    assert 'uv tool install --force "neatrl[atari]"' in text


def test_cli_prints_the_traceback_then_the_hint(monkeypatch, capsys):
    """Full error first (it is the ground truth), hint afterwards."""
    from neatrl.cli import main as cli_main

    def boom(*args, **kwargs):
        raise _namespace_error("ALE")

    monkeypatch.setattr(cli_main, "run_train", boom)
    with pytest.raises(SystemExit) as excinfo:
        cli_main.main(["train", "dqn-cnn"])
    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "NamespaceNotFound" in err  # the original error, unedited
    assert 'pip install "neatrl[atari]"' in err


def test_cli_still_reraises_undiagnosed_failures(monkeypatch):
    from neatrl.cli import main as cli_main

    def boom(*args, **kwargs):
        raise RuntimeError("something else entirely")

    monkeypatch.setattr(cli_main, "run_train", boom)
    with pytest.raises(RuntimeError, match="something else"):
        cli_main.main(["train", "dqn"])
