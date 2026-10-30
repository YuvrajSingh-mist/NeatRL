"""Training recipes behind ``neatrl train``.

The console command has to work from an installed wheel, where the
repository's ``docs/cli/`` examples are not on disk, so the trainer for
each algorithm -- and the small set of keyword arguments those examples
pass -- is declared here instead of being loaded from a file path.

Every entry names a public ``neatrl`` trainer by its import path; the
import happens only when the recipe runs, so listing the recipes (for
``--help``) does not pull in torch.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any


@dataclass(frozen=True)
class Call:
    """One trainer invocation: ``func(env_id=env, **kwargs)``."""

    func: str  # "neatrl.dqn_mlp.train_dqn"
    env: str  # default environment when the user names none
    kwargs: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Recipe:
    """The MLP trainer for an algorithm and, when one exists, its CNN sibling."""

    mlp: Call
    cnn: Call | None = None
    # Trainer keyword that enables Atari preprocessing, for algorithms whose
    # single trainer handles both (Dueling DQN) instead of a separate module.
    cnn_switch: str | None = None
    # DQN family counts steps; REINFORCE counts episodes.
    steps_kw: str = "total_timesteps"


RECIPES: dict[str, Recipe] = {
    "a2c": Recipe(
        mlp=Call(
            "neatrl.a2c_mlp.train_a2c",
            "CartPole-v1",
            {"grid_env": False, "capture_video": False, "use_wandb": False},
        ),
        cnn=Call(
            "neatrl.a2c_cnn.train_a2c_cnn",
            "ALE/Pong-v5",
            {"capture_video": False, "use_wandb": False},
        ),
    ),
    "ddpg": Recipe(
        mlp=Call(
            "neatrl.ddpg_mlp.train_ddpg",
            "Pendulum-v1",
            {"capture_video": False, "use_wandb": False},
        ),
        cnn=Call(
            "neatrl.ddpg_cnn.train_ddpg_cnn",
            "CarRacing-v3",
            {"capture_video": False, "use_wandb": False},
        ),
    ),
    "dqn": Recipe(
        mlp=Call(
            "neatrl.dqn_mlp.train_dqn",
            "CartPole-v1",
            {"atari_wrapper": False, "capture_video": False, "use_wandb": False},
        ),
        cnn=Call(
            "neatrl.dqn_cnn.train_dqn_cnn",
            "ALE/Pong-v5",
            {"capture_video": False, "use_wandb": False},
        ),
    ),
    "double_dqn": Recipe(
        mlp=Call(
            "neatrl.double_dqn_mlp.train_double_dqn",
            "CartPole-v1",
            {"atari_wrapper": False, "capture_video": False, "use_wandb": False},
        ),
        cnn=Call(
            "neatrl.double_dqn_cnn.train_double_dqn_cnn",
            "ALE/Pong-v5",
            {"capture_video": False, "use_wandb": False},
        ),
    ),
    "dueling_dqn": Recipe(
        mlp=Call(
            "neatrl.dueling_dqn_mlp.train_dueling_dqn",
            "CliffWalking-v1",
            {"capture_video": False, "use_wandb": False},
        ),
        # One trainer handles both: `atari_wrapper` switches it to the
        # Atari preprocessing + conv trunk that the `-cnn` alias expects.
        cnn=Call(
            "neatrl.dueling_dqn_mlp.train_dueling_dqn",
            "ALE/Pong-v5",
            {"capture_video": False, "use_wandb": False, "atari_wrapper": True},
        ),
        cnn_switch="atari_wrapper",
    ),
    "ppo": Recipe(
        mlp=Call(
            "neatrl.ppo_mlp.train_ppo",
            "CartPole-v1",
            {"capture_video": False, "use_wandb": False},
        ),
        cnn=Call(
            "neatrl.ppo_cnn.train_ppo_cnn",
            "ALE/Pong-v5",
            {"capture_video": False, "use_wandb": False},
        ),
    ),
    "q_learning": Recipe(
        mlp=Call(
            "neatrl.q_learning.train_q_learning",
            "FrozenLake-v1",
            {"capture_video": False, "use_wandb": False},
        )
    ),
    "reinforce": Recipe(
        mlp=Call(
            "neatrl.reinforce_mlp.train_reinforce",
            "CartPole-v1",
            {"capture_video": False, "use_wandb": False},
        ),
        cnn=Call(
            "neatrl.reinforce_cnn.train_reinforce_cnn",
            "ALE/Pong-v5",
            {"capture_video": False, "use_wandb": False},
        ),
        steps_kw="total_episodes",
    ),
    "rnd": Recipe(
        mlp=Call(
            "neatrl.rnd_mlp.train_ppo_rnd",
            "CliffWalking-v1",
            {"capture_video": False, "use_wandb": False},
        ),
        cnn=Call(
            "neatrl.rnd_cnn.train_ppo_rnd_cnn",
            "ALE/Pong-v5",
            {"capture_video": False, "use_wandb": False},
        ),
    ),
    "sac": Recipe(
        mlp=Call(
            "neatrl.sac_mlp.train_sac",
            "Pendulum-v1",
            {"capture_video": False, "use_wandb": False},
        ),
        cnn=Call(
            "neatrl.sac_cnn.train_sac_cnn",
            "CarRacing-v3",
            {"capture_video": False, "use_wandb": False},
        ),
    ),
    "td3": Recipe(
        mlp=Call(
            "neatrl.td3_mlp.train_td3",
            "Pendulum-v1",
            {"capture_video": False, "use_wandb": False},
        ),
        cnn=Call(
            "neatrl.td3_cnn.train_td3_cnn",
            "CarRacing-v3",
            {"capture_video": False, "use_wandb": False},
        ),
    ),
}


def _aliases() -> dict[str, tuple[str, bool]]:
    """Map each ``neatrl train`` alias to (recipe key, use_cnn).

    ``<algo>`` runs the MLP trainer; ``<algo>-cnn`` runs the CNN one,
    which only exists for algorithms that have one.
    """
    out: dict[str, tuple[str, bool]] = {}
    for key, recipe in RECIPES.items():
        alias = key.replace("_", "-")
        out[alias] = (key, False)
        if recipe.cnn is not None:
            out[f"{alias}-cnn"] = (key, True)
    return out


ALIASES: dict[str, tuple[str, bool]] = _aliases()


def _resolve(dotted: str):
    module_path, _, attr = dotted.rpartition(".")
    return getattr(import_module(module_path), attr)


def run_train(
    algo: str,
    env_id: str | None = None,
    *,
    use_cnn: bool = False,
    steps: int | None = None,
    use_wandb: bool = False,
) -> Any:
    """Run the trainer named by ``algo`` and return whatever it returns."""
    recipe = RECIPES[algo]
    call = recipe.cnn if (use_cnn and recipe.cnn is not None) else recipe.mlp

    kwargs: dict[str, Any] = dict(call.kwargs)
    if use_cnn and recipe.cnn_switch:
        kwargs[recipe.cnn_switch] = True
    if steps is not None:
        kwargs[recipe.steps_kw] = steps
    # The trainers default to logging to wandb; a terminal command should
    # not start an upload unless the user asks for one.
    kwargs["use_wandb"] = use_wandb

    return _resolve(call.func)(env_id=env_id or call.env, **kwargs)


def default_env(algo: str, use_cnn: bool = False) -> str:
    recipe = RECIPES[algo]
    call = recipe.cnn if (use_cnn and recipe.cnn is not None) else recipe.mlp
    return call.env
