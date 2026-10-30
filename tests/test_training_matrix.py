"""Regression tests for the cross-algorithm bugs found by the CLI matrix.

Each test here corresponds to a failure that made ``neatrl train`` break for a
whole algorithm, so they are deliberately cheap: they inspect source or build a
single network instead of training.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

from neatrl.cli.recipes import ALIASES, RECIPES
from neatrl.utils.nn_utils import ImageFeatures, observation_input_layer

SRC = Path(__file__).resolve().parents[1] / "src" / "neatrl"
TRAINERS = sorted(p.name for p in SRC.glob("*.py") if p.name != "__init__.py")

IMAGE_PREFIXES = ("ALE/", "CarRacing")


def _cnn_recipes():
    for key, recipe in RECIPES.items():
        if recipe.cnn is not None:
            yield key, recipe


def test_cnn_aliases_default_to_image_environments():
    """A ``-cnn`` alias must not default to a vector-observation env."""
    for key, recipe in _cnn_recipes():
        env = recipe.cnn.env
        assert env.startswith(IMAGE_PREFIXES), (
            f"recipe {key}-cnn defaults to {env}, which has no image observation"
        )


def test_every_cnn_trainer_is_passed_the_full_shape():
    """CNN trainers must hand networks the (C, H, W) shape, not one dimension."""
    for name in ("a2c_cnn", "ppo_cnn", "reinforce_cnn", "rnd_cnn"):
        src = (SRC / f"{name}.py").read_text()
        assert "observation_input_layer" in src, f"{name} builds a bare nn.Linear"


def test_dueling_defaults_to_cnn_classes():
    """The CNN trainers used to default to their MLP siblings' networks."""
    for name in ("ddpg_cnn", "td3_cnn"):
        src = (SRC / f"{name}.py").read_text()
        assert "actor_class: Any = ActorNetCNN" in src, name
        assert "q_network_class: Any = QNetCNN" in src, name


@pytest.mark.parametrize(
    "shape",
    [(4, 84, 84), (3, 96, 96), (96, 96, 3), (1, 42, 42)],
)
def test_image_features_accepts_any_geometry(shape):
    """Hardcoded 64*7*7 broke every non-84x84 env (CarRacing is 96x96)."""
    encoder = ImageFeatures(shape)
    out = encoder(torch.zeros(2, *shape))
    assert out.shape == (2, encoder.out_dim)
    assert encoder.out_dim > 0


def test_observation_input_layer_handles_vectors_and_images():
    flat = observation_input_layer(4, 32)
    assert flat(torch.zeros(2, 4)).shape == (2, 32)

    image = observation_input_layer((4, 84, 84), 32)
    assert image(torch.zeros(2, 4, 84, 84)).shape == (2, 32)


def test_eval_loops_are_step_capped():
    """CliffWalking-v1 has no TimeLimit, so an uncapped eval never returns."""
    offenders = []
    for name in TRAINERS:
        src = (SRC / name).read_text()
        if "def evaluate(" not in src:
            continue
        if "while not done:" in src:
            offenders.append(name)
    assert not offenders, f"uncapped evaluation loop in: {offenders}"


def test_env_close_is_guarded():
    """``env`` is None on the default CLI path, so close() must be guarded."""
    offenders = []
    for name in TRAINERS:
        src = (SRC / name).read_text()
        if re.search(r"^    env\.close\(\)", src, flags=re.M):
            offenders.append(name)
    assert not offenders, f"unguarded env.close() in: {offenders}"


def test_no_stale_get_action_calls():
    """1.0.0 dropped get_action's action argument; internal callers must match."""
    offenders = []
    for name in TRAINERS:
        src = (SRC / name).read_text()
        if re.search(r"get_action\(\s*\w+\[[^\]]*\],\s*\w+\[[^\]]*\]\s*\)", src):
            offenders.append(name)
    assert not offenders, f"stale get_action(x, action) call in: {offenders}"


def test_all_trainers_import():
    """Every module must import cleanly, which the matrix run depends on."""
    for name in TRAINERS:
        __import__(f"neatrl.{name[:-3]}")


def test_every_recipe_default_env_is_registered():
    """A default env that gymnasium has retired fails on the first step.

    ``CliffWalking-v0`` and ``Taxi-v3`` both moved to ``-v1``/``-v4``, so
    two aliases used to die with ``DeprecatedEnv`` before training began.
    """
    import gymnasium as gym

    for key, recipe in RECIPES.items():
        for kind, call in (("", recipe.mlp), ("-cnn", recipe.cnn)):
            if call is None:
                continue
            try:
                gym.spec(call.env)
            except Exception as exc:  # noqa: BLE001 - report the env id
                raise AssertionError(
                    f"recipe {key}{kind} defaults to {call.env}: {exc}"
                ) from exc


def test_every_cnn_capable_recipe_has_an_alias():
    """Bugs hid behind aliases that silently did not exist."""
    for key, recipe in RECIPES.items():
        alias = key.replace("_", "-")
        assert (key, False) == ALIASES[alias]
        if recipe.cnn is not None or recipe.cnn_switch:
            assert ALIASES[f"{alias}-cnn"] == (key, True), alias


def test_cli_training_does_not_log_to_wandb_by_default(monkeypatch):
    """A terminal command must not upload runs to Weights & Biases."""
    import neatrl.cli.recipes as recipes

    seen = {}

    def fake(dotted):
        def call(**kwargs):
            seen.update(kwargs)
            return None

        return call

    monkeypatch.setattr(recipes, "_resolve", fake)
    recipes.run_train("dqn", "CartPole-v1", steps=10)
    assert seen["use_wandb"] is False

    recipes.run_train("dqn", "CartPole-v1", steps=10, use_wandb=True)
    assert seen["use_wandb"] is True


def test_dueling_q_net_accepts_image_observations():
    """The dueling trainer serves the `-cnn` alias through `atari_wrapper`."""
    from neatrl.dueling_dqn_mlp import DuelingQNet

    net = DuelingQNet((4, 84, 84), 3)
    q, values, adv, _feat = net(torch.zeros(2, 4 * 84 * 84))
    assert q.shape == (2, 3)
    assert values.shape == (2, 1)
    assert adv.shape == (2, 3)


def _pyproject_block(name: str) -> str:
    """Return the contents of a top-level ``name = [...]`` array in pyproject."""
    import re
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    match = re.search(rf"^{name} = \[(.*?)^\]", text, re.S | re.M)
    assert match, f"{name} not found in pyproject.toml"
    return match.group(1)


def test_extra_bundles_every_env_family():
    """`neatrl[all]` is what makes the aliases runnable from one install."""
    all_extra = _pyproject_block("all")
    for token in ("ale-py", "box2d", "classic-control", "atari", "mujoco", "rich"):
        assert token in all_extra, f"{token} missing from the `all` extra"


def test_base_install_ships_the_dashboard_and_classic_control():
    """`pip install neatrl` alone must give the TUI and classic-control envs.

    Only the native environment families (Atari, Box2D, MuJoCo) stay opt-in:
    they need compiled packages, and every alias that uses them fails with an
    install hint when they are absent.
    """
    base = _pyproject_block("dependencies")
    for token in ("rich", "psutil", "nvidia-ml-py", "classic-control", "pygame-ce"):
        assert token in base, f"{token} missing from the base dependencies"
    assert "ale-py" not in base, "Atari should stay an opt-in extra"
