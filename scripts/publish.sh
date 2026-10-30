#!/usr/bin/env bash
# Publish a NeatRL wheel to PyPI.
#
#   ./scripts/publish.sh              # test, build, upload the current version
#   ./scripts/publish.sh --dry-run    # build only, then `twine check`
#
# Credentials live in .env at the repo root (gitignored):
#
#   UV_PUBLISH_TOKEN=pypi-...
#
# The token is a project-scoped PyPI API token, so it is also usable for
# `uv publish --token` by hand. Nothing here echoes it.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

PY="${PYTHON:-python3}"
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

VERSION=$("$PY" -c "import tomllib,pathlib;print(tomllib.loads(pathlib.Path('pyproject.toml').read_text())['project']['version'])")
echo "==> neatrl $VERSION"

if git rev-parse --git-dir >/dev/null 2>&1 && [[ -n "$(git status --porcelain)" ]]; then
    echo "!! uncommitted changes: the published sdist will not match the tag" >&2
fi

echo "==> tests"
"$PY" -m pytest tests -q

echo "==> build"
rm -rf dist
"$PY" -m build --outdir dist -q

if (( DRY_RUN )); then
    "$PY" -m twine check dist/* 2>/dev/null || true
    echo "==> dry run: built $(ls dist | tr '\n' ' ')"
    exit 0
fi

if [[ -z "${UV_PUBLISH_TOKEN:-}" ]]; then
    echo "!! UV_PUBLISH_TOKEN is not set. Put it in .env (see this script's header)." >&2
    exit 1
fi

echo "==> upload"
# uv reads UV_PUBLISH_TOKEN from the environment; never pass it on argv, where
# it would land in the shell history and in `ps`.
uv publish dist/*

echo "==> done: https://pypi.org/project/neatrl/$VERSION/"
echo "    users get it with: pip install --upgrade neatrl"
