"""Translate "this environment's dependency is missing" into an install hint.

``neatrl train dqn-cnn`` defaults to ``ALE/Pong-v5``, and gymnasium raises
``NamespaceNotFound: Namespace ALE not found. Have you installed the proper
package for ALE?`` when ``ale-py`` is absent. That message is accurate but does
not say which pip command fixes it, and the traceback alone makes it look like
a bug in this package.

The original traceback is still printed -- it is the ground truth, and hiding
it would make real failures harder to diagnose -- and this module supplies the
one line to run afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExtraHint:
    """A missing optional dependency and the command that installs it."""

    extra: str  # "atari"
    package: str  # "ale-py"
    unlocks: str  # human description of what starts working


# Checked in order against the whole exception chain, lowercased.
_HINTS: tuple[tuple[tuple[str, ...], ExtraHint], ...] = (
    (
        (
            "namespace ale",
            "no module named 'ale_py'",
            'no module named "ale_py"',
            "ale_py",
        ),
        ExtraHint(
            extra="atari",
            package="ale-py",
            unlocks="Atari games such as ALE/Pong-v5 and ALE/Breakout-v5",
        ),
    ),
    (
        ("no module named 'box2d'", "box2d is not installed", "box2d-py"),
        ExtraHint(
            extra="box2d",
            package="Box2D",
            unlocks="Box2D environments such as CarRacing-v3, LunarLander-v3",
        ),
    ),
    (
        ("no module named 'mujoco'", "mujoco is not installed"),
        ExtraHint(
            extra="mujoco",
            package="mujoco",
            unlocks="MuJoCo environments such as HalfCheetah-v4, Ant-v5",
        ),
    ),
)


def _chain_text(exc: BaseException) -> str:
    """Flatten an exception and its causes into one lowercase string."""
    parts: list[str] = []
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        parts.append(f"{type(cur).__name__}: {cur}")
        nxt = cur.__cause__ or cur.__context__
        cur = nxt if isinstance(nxt, BaseException) else None
    return " ".join(parts).lower()


def missing_extra(exc: BaseException) -> ExtraHint | None:
    """Return the extra that would fix ``exc``, or None if it is unrelated."""
    text = _chain_text(exc)
    for needles, hint in _HINTS:
        if any(needle in text for needle in needles):
            return hint
    return None


def format_hint(hint: ExtraHint) -> str:
    """The message shown under the traceback."""
    return (
        f"neatrl: this environment needs the '{hint.package}' package, which "
        f"is not installed.\n"
        f"neatrl: install the '{hint.extra}' extra to enable {hint.unlocks}:\n"
        f'\n    pip install "neatrl[{hint.extra}]"\n'
        f'\nneatrl: with uv tool: uv tool install --force "neatrl[{hint.extra}]"'
    )


__all__ = ["ExtraHint", "missing_extra", "format_hint"]
