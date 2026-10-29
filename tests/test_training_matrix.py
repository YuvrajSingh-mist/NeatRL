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

from neatrl.cli.recipes import RECIPES
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
    """CliffWalking-v0 has no TimeLimit, so an uncapped eval never returns."""
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
