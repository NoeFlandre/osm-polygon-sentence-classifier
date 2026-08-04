set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

format:
    uv run ruff format .
format-check:
    uv run ruff format --check .
lint:
    uv run ruff check .
typecheck:
    uv run ty check
test:
    uv run pytest -q
docs:
    uv run mkdocs build --strict --site-dir site
check: format-check lint typecheck test
