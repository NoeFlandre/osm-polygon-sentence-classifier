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

docker-build image="osm-polygon-sentence-classifier:local":
    docker build --platform linux/amd64 --tag "{{image}}" .

docker-smoke image="osm-polygon-sentence-classifier:local":
    docker run --rm --platform linux/amd64 --network none --read-only --tmpfs /tmp:rw,noexec,nosuid "{{image}}"

check: format-check lint typecheck test
