# Contributing

## Required local gate

Use the locked environment and run:

```bash
uv sync --locked --all-extras --dev
just check
uv run mkdocs build --strict --site-dir site
```

New Python behavior follows RED -> GREEN -> REFACTOR: add a focused failing
test, observe the expected failure, implement the smallest passing behavior,
then run the complete gate.

## Scope boundaries

Do not place datasets, checkpoints, models, experiment logs, credentials, or
temporary job state in the repository. Local project data belongs beneath
`/Volumes/Seagate M3/projects/osm-polygon-sentence-classifier`.

Do not download the Hugging Face dataset, authenticate, upload, submit a
Grid'5000 job, or publish a model without an explicit task-level authorization
and a documented verification plan.
