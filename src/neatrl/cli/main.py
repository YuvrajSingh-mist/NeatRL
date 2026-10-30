"""NeatRL CLI -- ``neatrl train <algo> [env_id]``.

Dispatch comes from ``neatrl.cli.recipes``, so the command works the same
from an installed wheel as it does from a repository checkout.
"""

import argparse
import sys
import traceback

from .extras import format_hint, missing_extra
from .recipes import ALIASES, RECIPES, RecipeError, default_env, run_train


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neatrl",
        description="NeatRL -- train a reinforcement learning agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  neatrl train dqn CartPole-v1\n"
            "  neatrl train dqn-cnn ALE/Pong-v5\n"
            "  neatrl train double-dqn-cnn ALE/Pong-v5\n"
            "  neatrl train q-learning FrozenLake-v1\n"
            "  neatrl train ppo LunarLander-v3\n"
            "  neatrl train sac Pendulum-v1\n"
            "  neatrl train td3-cnn CarRacing-v3\n"
            "  neatrl train rnd CliffWalking-v1\n"
            "\n"
            "Omit env_id to use the algorithm's default environment.\n"
        ),
    )
    subs = parser.add_subparsers(dest="command", metavar="command")

    train_p = subs.add_parser("train", help="Train an agent")
    train_p.add_argument(
        "algo",
        choices=sorted(ALIASES),
        metavar="algo",
        help=f"Algorithm ({', '.join(sorted(ALIASES))})",
    )
    train_p.add_argument(
        "env",
        metavar="env_id",
        nargs="?",
        help="Gymnasium environment ID (e.g. CartPole-v1, Pendulum-v1)",
    )
    train_p.add_argument(
        "--steps",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Training steps, or episodes for reinforce. "
            "Defaults to the trainer's own setting."
        ),
    )
    train_p.add_argument(
        "--wandb",
        action="store_true",
        help="Log the run to Weights & Biases (off by default).",
    )
    return parser


def _warn_if_dashboard_disabled() -> None:
    """Say why the TUI is missing, instead of silently training without it.

    ``rich`` ships in the base install now, so this only fires for an old
    release or an environment where the import genuinely failed -- which used
    to be indistinguishable from a bug, since the dashboard just never
    appeared.
    """
    from . import Dashboard

    if Dashboard is not None or not sys.stdout.isatty():
        return

    print(
        "note: the terminal dashboard is disabled because 'rich' could not be "
        "imported (it is part of the base install). Reinstall with: "
        "pip install --force-reinstall neatrl  "
        "(uv tool: uv tool install --force neatrl)",
        file=sys.stderr,
    )


def main(argv=None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "train":
        key, use_cnn = ALIASES[args.algo]
        env_id = args.env or default_env(key, use_cnn)
        _warn_if_dashboard_disabled()
        try:
            run_train(
                key,
                env_id,
                use_cnn=use_cnn,
                steps=args.steps,
                use_wandb=args.wandb,
            )
        except RecipeError as exc:
            # Fixture mismatches (an image env handed to an MLP alias, and the
            # like) are user errors, not crashes: report them without a
            # traceback.
            print(f"neatrl: error: {exc}", file=sys.stderr)
            sys.exit(2)
        except Exception as exc:  # noqa: BLE001 - re-raised unless diagnosed
            hint = missing_extra(exc)
            if hint is None:
                raise
            # Keep gymnasium's own error: it names the missing package, and
            # rewriting it would make real failures harder to diagnose.
            traceback.print_exc()
            print("", file=sys.stderr)
            print(format_hint(hint), file=sys.stderr)
            sys.exit(1)


__all__ = ["main", "RECIPES"]
