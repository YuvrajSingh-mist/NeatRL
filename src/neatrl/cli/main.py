"""NeatRL CLI -- ``neatrl train <algo> [env_id]``.

Dispatch comes from ``neatrl.cli.recipes``, so the command works the same
from an installed wheel as it does from a repository checkout.
"""

import argparse
import sys

from .recipes import ALIASES, RECIPES, default_env, run_train


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
            "  neatrl train rnd CliffWalking-v0\n"
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
    return parser


def main(argv=None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "train":
        key, use_cnn = ALIASES[args.algo]
        env_id = args.env or default_env(key, use_cnn)
        run_train(key, env_id, use_cnn=use_cnn, steps=args.steps)


__all__ = ["main", "RECIPES"]
