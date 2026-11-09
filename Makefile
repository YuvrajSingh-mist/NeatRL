.PHONY: venv install install-dev lint format test docs build upload publish release upgrade clean

PYTHON ?= python3

venv:
	@if [ ! -d ".venv" ]; then \
		uv venv; \
	fi

install: venv
	uv pip install -e .

install-dev: venv
	uv pip install -e .[dev]

lint: venv
	uv run ruff check --fix .

format:
	uv run ruff format .

test: venv
	uv run pytest -q

# Links, the CLI → non-cli folder mapping, and gymnasium env ids in the docs.
docs: venv
	uv run python scripts/check_docs.py --env-ids

build: install-dev lint format
	uv run python -m build

# Publishing is not wired into the Makefile any more: scripts/publish.sh and
# scripts/bump_version.py are local-only (see .gitignore). Build with
# `make build`, then upload with `uv run twine upload dist/*`.
upload:
	@echo "scripts/publish.sh is not tracked; run 'make build' then 'twine upload dist/*'." >&2
	@exit 1

publish: upload

# Bump the version in pyproject.toml by hand, then build and upload. The
# bump helper (scripts/bump_version.py) is local-only.
release:
	@echo "scripts/bump_version.py is not tracked; edit pyproject.toml, then 'make build'." >&2
	@exit 1

# Upgrade the tools installed on this machine from PyPI, which is what users
# are told to run: `pip install --upgrade neatrl`.
upgrade:
	uv tool upgrade neatrl
	@neatrl --version

clean:
	rm -rf dist/ build/ *.egg-info/ .venv/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

all: install-dev lint format build upload
