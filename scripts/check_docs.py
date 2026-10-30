#!/usr/bin/env python3
"""Check that the documentation's links and per-algorithm structure hold.

Two failure modes have bitten this repo:

* a CLI page's ``Docs`` row pointing at ``../non-cli/`` (the folder root)
  because the algorithm's own subfolder was never created, so the reader lands
  on an index instead of the guide;
* a relative link whose target was renamed, moved, or never added.

Run with ``--env-ids`` to also check that every environment id named in the
docs is registered by gymnasium. Ids that need an optional extra (Atari,
Box2D, MuJoCo) are reported and skipped when that extra is not installed, so
the check is meaningful without installing all of them.

Usage::

    python scripts/check_docs.py
    python scripts/check_docs.py --env-ids
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
CLI_DOCS = DOCS / "cli"
NON_CLI_DOCS = DOCS / "non-cli"

LINK_RE = re.compile(r"(?<!!)\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
DOCS_ROW_RE = re.compile(r"^\|\s*Docs\s*\|\s*\[[^\]]*\]\(([^)]+)\)", re.M)
# A gymnasium id looks like `CartPole-v1` or `ALE/Pong-v5`. Names start with a
# capital so that lowercase exp_name strings ("reinforce-cartpole-v1") are not
# mistaken for environments.
ENV_ID_RE = re.compile(r"[A-Z][A-Za-z0-9]*(?:/[A-Za-z0-9]+)?-v\d+")
ENV_ID_EXACT_RE = re.compile(rf"^{ENV_ID_RE.pattern}$")
FENCE_RE = re.compile(r"```(?:bash|console|sh)\n(.*?)```", re.S)
ENV_KWARG_RE = re.compile(r"""env_id\s*=\s*["']([^"']+)["']""")
INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")

# Environment families whose registration needs an optional dependency.
OPTIONAL_ENV_PATTERNS = (
    r"^ALE/",
    r"^CarRacing-",
    r"^LunarLander",
    r"^(Ant|HalfCheetah|Hopper|Humanoid|Walker2d|Reacher|Swimmer|Pusher|"
    r"Thrower|Striker|InvertedPendulum|InvertedDoublePendulum)-",
    # Placeholder ids in copy-paste examples, not real registrations.
    r"^Your[A-Za-z]*Env-",
)


def markdown_files() -> list[pathlib.Path]:
    files = [ROOT / "README.md"]
    files.extend(sorted(DOCS.rglob("*.md")))
    return [p for p in files if p.is_file()]


def relative_links(path: pathlib.Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    targets = []
    for raw in LINK_RE.findall(text):
        target = raw.split("#", 1)[0].split("?", 1)[0].strip()
        if not target or "://" in target or target.startswith(("mailto:", "#")):
            continue
        targets.append(target)
    return targets


def check_links() -> list[str]:
    problems = []
    for md in markdown_files():
        for target in relative_links(md):
            resolved = (md.parent / target).resolve()
            if not resolved.exists():
                problems.append(f"{md.relative_to(ROOT)}: broken link -> {target}")
    return problems


def check_structure() -> list[str]:
    """Every CLI algorithm page must point at its own non-cli subfolder."""
    problems = []
    for folder in sorted(NON_CLI_DOCS.iterdir()):
        if not folder.is_dir():
            continue
        if not (folder / "README.md").is_file():
            problems.append(f"docs/non-cli/{folder.name}/: missing README.md")

    for cli_page in sorted(CLI_DOCS.glob("*.md")):
        if cli_page.name == "README.md":
            continue
        text = cli_page.read_text(encoding="utf-8")
        match = DOCS_ROW_RE.search(text)
        if not match:
            problems.append(f"{cli_page.relative_to(ROOT)}: no `| Docs |` row")
            continue
        target = match.group(1).strip()
        if target.rstrip("/") == "../non-cli":
            problems.append(
                f"{cli_page.relative_to(ROOT)}: Docs points at the non-cli root, "
                "not an algorithm subfolder"
            )
            continue
        resolved = (cli_page.parent / target).resolve()
        if not (resolved / "README.md").is_file():
            problems.append(
                f"{cli_page.relative_to(ROOT)}: Docs -> {target} "
                "has no README.md (or does not exist)"
            )
    return problems


def env_ids_named(text: str) -> set[str]:
    """Environment ids the docs actually point a reader at.

    Only structured places count — shell commands, ``env_id=...`` values and
    inline code — so training-run names (``exp_name="..."``) are not checked.
    """
    ids = set(ENV_KWARG_RE.findall(text))
    for block in FENCE_RE.findall(text):
        for line in block.splitlines():
            for token in line.split():
                token = token.strip("\"'")
                if ENV_ID_EXACT_RE.match(token):
                    ids.add(token)
    for inline in INLINE_CODE_RE.findall(text):
        inline = inline.strip()
        if ENV_ID_EXACT_RE.match(inline):
            ids.add(inline)
    return ids


def check_env_ids() -> list[str]:
    try:
        import gymnasium as gym
    except ImportError:
        print("gymnasium is not installed: skipping environment-id check")
        return []

    try:  # Atari registers its namespace on import.
        import ale_py

        gym.register_envs(ale_py)
    except Exception:  # noqa: BLE001 - the extra is optional
        pass

    registered = set(gym.registry)
    optional = [re.compile(p) for p in OPTIONAL_ENV_PATTERNS]

    problems = []
    for md in markdown_files():
        for env_id in sorted(env_ids_named(md.read_text(encoding="utf-8"))):
            if env_id in registered:
                continue
            if any(p.match(env_id) for p in optional):
                continue
            problems.append(f"{md.relative_to(ROOT)}: {env_id} is not a registered env")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env-ids",
        action="store_true",
        help="also verify gymnasium environment ids named in the docs",
    )
    args = parser.parse_args()

    problems = check_links() + check_structure()
    if args.env_ids:
        problems += check_env_ids()

    if problems:
        print(f"{len(problems)} documentation problem(s):\n")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("Documentation checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
