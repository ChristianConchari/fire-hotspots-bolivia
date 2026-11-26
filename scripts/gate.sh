#!/usr/bin/env bash
# Quality gate: the same checks run locally and in CI.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

uv sync --frozen
uv run ruff check
uv run pytest -q
