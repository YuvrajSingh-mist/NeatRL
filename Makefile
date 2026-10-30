.PHONY: venv install install-dev lint format test build upload publish release upgrade clean

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

build: install-dev lint format
	uv run python -m build

# Publish to PyPI. Credentials come from .env (gitignored); see
# scripts/publish.sh, which also runs the tests first.
upload:
	./scripts/publish.sh

publish: upload

# Bump, then publish: `make release V=2.6.0`
release:
	@test -n "$(V)" || (echo "usage: make release V=2.6.0" >&2; exit 1)
	@$(PYTHON) scripts/bump_version.py $(V)
	$(MAKE) publish

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
