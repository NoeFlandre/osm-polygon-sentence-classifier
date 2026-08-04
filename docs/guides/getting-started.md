# Getting started

## Prerequisites

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- Git
- just

The Seagate volume is not required for this foundation. None of the commands
below creates project datasets, checkpoints, models, Trackio run state, or
other project data.

## Local quality gates

Install the locked development environment and run the complete local gate:

```bash
uv sync --locked --all-extras --dev
just check
just docs
```

The equivalent direct commands are:

```bash
uv run pytest -q
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run mkdocs build --strict --site-dir site
```

To run the local hooks against the repository, use:

```bash
uv run pre-commit run --all-files
```

No command in this foundation downloads the training dataset, authenticates,
or submits a Grid5000 job. Dependency installation may use the normal Python
package index, but it does not access project data or remote training systems.
