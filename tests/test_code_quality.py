"""Tier 5 — regression and quality guards (fast, grep/AST-based)."""

import ast
import dataclasses
import importlib
import pathlib

import pytest

SRC = pathlib.Path(__file__).parent.parent / "src" / "neatrl"

ALGO_MODULES = [
    "neatrl.a2c_mlp",
    "neatrl.a2c_cnn",
    "neatrl.ddpg_mlp",
    "neatrl.ddpg_cnn",
    "neatrl.dqn_mlp",
    "neatrl.dueling_dqn_mlp",
    "neatrl.ppo_mlp",
    "neatrl.ppo_cnn",
    "neatrl.reinforce_mlp",
    "neatrl.reinforce_cnn",
    "neatrl.rnd_mlp",
    "neatrl.rnd_cnn",
    "neatrl.sac_mlp",
    "neatrl.sac_cnn",
    "neatrl.td3_mlp",
    "neatrl.td3_cnn",
]


# ─── no print() in algo files ────────────────────────────────────────────────


def _collect_print_calls(path: pathlib.Path) -> list[int]:
    """Return line numbers of print() calls found via AST."""
    tree = ast.parse(path.read_text())
    hits = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        ):
            hits.append(node.lineno)
    return hits


@pytest.mark.parametrize("py_file", sorted(SRC.glob("*.py")))
def test_no_print_calls_in_algo_files(py_file):
    hits = _collect_print_calls(py_file)
    assert hits == [], (
        f"{py_file.name} has print() calls at lines {hits}. Use logger instead."
    )


@pytest.mark.parametrize("py_file", sorted((SRC / "utils").glob("*.py")))
def test_no_print_calls_in_utils(py_file):
    hits = _collect_print_calls(py_file)
    assert hits == [], (
        f"utils/{py_file.name} has print() calls at lines {hits}. Use logger instead."
    )


# ─── module docstring present in every .py ───────────────────────────────────


def _has_module_docstring(path: pathlib.Path) -> bool:
    tree = ast.parse(path.read_text())
    return (
        (
            isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)
        )
        if tree.body
        else False
    )


@pytest.mark.parametrize("py_file", sorted(SRC.glob("*.py")))
def test_module_docstring_present_algo(py_file):
    assert _has_module_docstring(py_file), (
        f"{py_file.name} is missing a module-level docstring."
    )


@pytest.mark.parametrize("py_file", sorted((SRC / "utils").glob("*.py")))
def test_module_docstring_present_utils(py_file):
    assert _has_module_docstring(py_file), (
        f"utils/{py_file.name} is missing a module-level docstring."
    )


# ─── __all__ completeness in utils ───────────────────────────────────────────


def test_utils_all_are_importable():
    from neatrl import utils

    for name in utils.__all__:
        assert hasattr(utils, name), (
            f"neatrl.utils.__all__ lists '{name}' but it is not importable."
        )


# ─── top-level neatrl exports ────────────────────────────────────────────────


EXPECTED_EXPORTS = [
    "train_a2c",
    "train_a2c_cnn",
    "train_ddpg",
    "train_ddpg_cnn",
    "train_dqn",
    "train_dueling_dqn",
    "train_ppo",
    "train_ppo_cnn",
    "train_ppo_rnd",
    "train_ppo_rnd_cnn",
    "train_reinforce",
    "train_reinforce_cnn",
    "train_sac",
    "train_sac_cnn",
    "train_td3",
    "train_td3_cnn",
]


@pytest.mark.parametrize("fn_name", EXPECTED_EXPORTS)
def test_top_level_export_callable(fn_name):
    import neatrl

    fn = getattr(neatrl, fn_name, None)
    assert fn is not None, f"neatrl.{fn_name} is not exported."
    assert callable(fn), f"neatrl.{fn_name} is not callable."


# ─── Config fields all have non-None defaults ────────────────────────────────


@pytest.mark.parametrize("module_name", ALGO_MODULES)
def test_config_fields_have_defaults(module_name):
    mod = importlib.import_module(module_name)
    for field in dataclasses.fields(mod.Config):
        has_default = field.default is not dataclasses.MISSING
        has_factory = field.default_factory is not dataclasses.MISSING  # type: ignore[misc]
        assert has_default or has_factory, (
            f"{module_name}.Config.{field.name} has no default value — every field must have a default."
        )


# ─── Config default instance is reproducible ─────────────────────────────────


@pytest.mark.parametrize("module_name", ALGO_MODULES)
def test_config_default_reproducible(module_name):
    mod = importlib.import_module(module_name)
    assert mod.Config() == mod.Config()
