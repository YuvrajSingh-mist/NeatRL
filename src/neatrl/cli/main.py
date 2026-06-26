"""NeatRL CLI — neatrl train <algo> <env>"""

import argparse
import importlib.util
import pathlib
import sys

_DOCS_CLI = pathlib.Path(__file__).resolve().parent.parent.parent.parent / "docs" / "cli"

_SCRIPTS = {
    "a2c":           "a2c",
    "a2c-cnn":       "a2c",
    "ddpg":          "ddpg",
    "ddpg-cnn":      "ddpg",
    "dqn":           "dqn",
    "dueling-dqn":   "dueling_dqn",
    "ppo":           "ppo",
    "ppo-cnn":       "ppo",
    "reinforce":     "reinforce",
    "reinforce-cnn": "reinforce",
    "rnd":           "rnd",
    "rnd-cnn":       "rnd",
    "sac":           "sac",
    "sac-cnn":       "sac",
    "td3":           "td3",
    "td3-cnn":       "td3",
}


def _load_and_run(algo: str, env_id: str) -> None:
    script = _DOCS_CLI / f"{_SCRIPTS[algo]}.py"
    if not script.exists():
        print(f"Script not found: {script}", file=sys.stderr)
        sys.exit(1)
    spec = importlib.util.spec_from_file_location("_neatrl_script", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.run(env_id=env_id, use_cnn=algo.endswith("-cnn"))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="neatrl",
        description="NeatRL — edit docs/cli/<algo>.py then run from here",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  neatrl train dqn CartPole-v1\n"
            "  neatrl train ppo LunarLander-v3\n"
            "  neatrl train sac Pendulum-v1\n"
            "  neatrl train td3-cnn CarRacing-v3\n"
            "  neatrl train rnd CliffWalking-v0\n"
        ),
    )
    subs = parser.add_subparsers(dest="command", metavar="command")

    train_p = subs.add_parser("train", help="Run a docs/cli/<algo>.py script")
    train_p.add_argument(
        "algo",
        choices=sorted(_SCRIPTS),
        metavar="algo",
        help=f"Algorithm ({', '.join(sorted(_SCRIPTS))})",
    )
    train_p.add_argument(
        "env",
        metavar="env_id",
        help="Gymnasium environment ID (e.g. CartPole-v1, Pendulum-v1)",
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "train":
        _load_and_run(args.algo, args.env)


if __name__ == "__main__":
    main()
