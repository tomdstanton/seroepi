# SeroEpi Project Justfile
# Run `just` to see all available commands

set shell := ["bash", "-c"]

# Show available commands
default:
    @just --list

# Sync python dependencies and create the virtual environment using `uv`
sync:
    uv sync

test: sync
    uvx karva test

# Update dependencies and lockfile
update:
    uv lock --upgrade
    uv sync

# Clean caches and temporary files
clean:
    rm -rf .cache .ruff_cache .karva_cache build dist site
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type f -name "*.pyc" -delete

# Deep clean including the virtual environment
clean-all: clean
    rm -rf .venv

# Format all Python code
fmt:
    uvx ruff format .

# Check if code is formatted without modifying files
fmt-check:
    uvx ruff format --check .

# Lint Python code and auto-fix safe errors
lint:
    uvx ruff check --fix .

# Static type-check Python code
type-check:
    uv sync --all-groups
    uvx ty check src/

# Run all quality checks at once (ideal for local pre-commit testing)
check-all: fmt-check lint type-check

# Run the full CI pipeline locally
ci: check-all test

# Run the local Shiny server
serve: sync
    uv run shiny run app:app --reload --host 127.0.0.1 --port 8000

# Build the standalone Docker image
docker-build:
    docker build -t seroepi-app:latest .

# Build the Python package
build: clean
    uv build

# Publish the Python package to PyPI
publish: build
    uv publish

# Build the documentation locally
docs-build: sync
    uv run --group docs zensical build

# Test if documentation can be built without warnings or errors
docs-test: sync
    uv run --group docs zensical build -s
