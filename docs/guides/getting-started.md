# Getting started

## Prerequisites

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- Git
- just

The Seagate volume is required only when running the explicit audit command.
The quality commands below do not create project datasets, checkpoints, models,
Trackio run state, or other project data.

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

The explicit `audit-landuse-dataset` command is the only command that streams
the training dataset; it writes only its approved cache, report, and split
manifest beneath the Seagate root. It does not authenticate, train, or submit a
Grid5000 job. Dependency installation may use the normal Python package index,
but quality commands do not access project data or remote training systems.
