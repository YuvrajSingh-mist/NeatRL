.PHONY: venv install install-dev lint format build upload clean

venv:
	uv venv

install: venv
	. .venv/bin/activate && uv pip install -e .

install-dev: venv
	. .venv/bin/activate && uv pip install -e .[dev]

lint:
	. .venv/bin/activate && ruff check .

format:
	. .venv/bin/activate && ruff format .

build: install-dev
	. .venv/bin/activate && python -m build

upload: build
	. .venv/bin/activate && twine upload dist/*

clean:
	rm -rf dist/ build/ *.egg-info/ .venv/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

all: install-dev lint format build upload

