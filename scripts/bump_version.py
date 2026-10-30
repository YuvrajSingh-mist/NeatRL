#!/usr/bin/env python3
"""Bump the version in pyproject.toml and open a matching CHANGELOG entry.

    python scripts/bump_version.py 2.6.0

Kept deliberately small: one source of truth (``pyproject.toml``), and the
CHANGELOG heading is inserted so a release never ships undocumented.
"""

from __future__ import annotations

import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
CHANGELOG = ROOT / "CHANGELOG.md"
INIT = ROOT / "src" / "neatrl" / "__init__.py"


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    new = argv[1].lstrip("v")
    if not re.fullmatch(r"\d+\.\d+\.\d+", new):
        print(f"not a PEP 440 release version: {new!r}", file=sys.stderr)
        return 2

    text = PYPROJECT.read_text()
    old = re.search(r'^version = "([^"]+)"$', text, re.M)
    if old is None:
        print("no version line in pyproject.toml", file=sys.stderr)
        return 1
    if old.group(1) == new:
        print(f"already at {new}")
    else:
        PYPROJECT.write_text(
            text.replace(f'version = "{old.group(1)}"', f'version = "{new}"', 1)
        )
        print(f"pyproject.toml: {old.group(1)} -> {new}")

    changelog = CHANGELOG.read_text()
    if f"## [{new}]" not in changelog:
        today = date.today().isoformat()
        entry = f"## [{new}] - {today}\n\n### Changed\n- \n\n"
        changelog = changelog.replace(
            "## [Unreleased]\n\n", "## [Unreleased]\n\n" + entry, 1
        )
        CHANGELOG.write_text(changelog)
        print(f"CHANGELOG.md: opened a {new} section -- fill it in before publishing")

    # A hardcoded release version in __init__ would drift from pyproject.toml;
    # the "+unknown" fallback for source checkouts is expected.
    if INIT.exists():
        init = INIT.read_text()
        hardcoded = re.search(r'__version__\s*=\s*"(\d+\.\d+\.\d+)"', init)
        if hardcoded:
            print(
                f"warning: src/neatrl/__init__.py hardcodes {hardcoded.group(1)}; "
                "it should read the installed metadata instead",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
