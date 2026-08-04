# OSM Polygon Sentence Classifier

Train a sentence classifier for OSM polygon descriptions, starting with the
`landuse` task.

## Status

The repository currently contains the safe, testable project foundation. It
does not download data, train a model, submit Grid'5000 jobs, authenticate to
Hugging Face, or publish remote artifacts.

## Data and model repositories

- Training source: [NoeFlandre/osm-polygon-wikidata-sentence-relevance](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-sentence-relevance)
- Eventual model repository: [NoeFlandre/osm-polygon-sentence-classifier](https://huggingface.co/NoeFlandre/osm-polygon-sentence-classifier)
- Local project-data root: `/Volumes/Seagate M3/projects/osm-polygon-sentence-classifier`

All local datasets, checkpoints, models, and experiment logs must be kept
beneath the external data root. Nothing in this repository downloads or
creates them yet.

## Development

```bash
uv sync --locked --all-extras --dev
uv run pytest -q
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run mkdocs build --strict --site-dir site
```

The equivalent Just recipes are documented in
[`docs/guides/getting-started.md`](docs/guides/getting-started.md).

## Grid'5000

Computation will run on Grid'5000 in a later milestone. The current safety
boundary is documented in
[`docs/reference/grid5000-boundary.md`](docs/reference/grid5000-boundary.md).
