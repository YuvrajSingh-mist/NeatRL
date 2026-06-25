.PHONY: venv install install-dev lint format build upload clean

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

build: install-dev lint format
	uv run python -m build

upload: build lint format
	uv run twine upload dist/*

clean:
	rm -rf dist/ build/ *.egg-info/ .venv/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

all: install-dev lint format build upload

