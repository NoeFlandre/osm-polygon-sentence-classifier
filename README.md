# OSM Polygon Sentence Classifier

Train a sentence classifier for OSM polygon descriptions, starting with the
`landuse` task.

## Status

The repository contains a safe, testable project foundation, one explicit
review command (`audit-landuse-dataset`), a typed training module, and a
plan-first Grid'5000 operator.
The audit consumes the pinned dataset in streaming mode. Its loader may
populate the approved Hugging Face cache; the explicit artifact writer creates
only the JSON report and polygon split manifest beneath the approved external
data root. The training module consumes the clean iterator, uses a Hugging
Face Trainer with local Trackio reporting, and saves model artifacts beneath
the same root. It does not upload or publish artifacts or submit Grid'5000
jobs.

The clean iterator `iter_clean_training_examples` is the only permitted
training-input boundary. The public iterator remains lazy until consumed, and
it is driven by two fresh streams supplied through a factory. The first fresh
stream is fully consumed to discover contradictory sentence-content-hash
groups; the second fresh stream is then consumed to emit clean
representatives, keeping only the first trainable occurrence of each
remaining usable hash. Rows are processed incrementally as they arrive from
each stream rather than materialized into an intermediate list, and no
cleaned dataset is written.

`train_landuse_classifier` is an execution boundary, not a dataset publishing
command. It should be invoked only by an explicitly authorized training
workflow after reviewing the landuse audit. The default model and training
settings are configurable through `TrainingConfig`; Grid'5000 runs must
additionally pin a model revision.

## Data and model repositories

- Training source: [NoeFlandre/osm-polygon-wikidata-sentence-relevance](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-sentence-relevance)
- Eventual model repository: [NoeFlandre/osm-polygon-sentence-classifier](https://huggingface.co/NoeFlandre/osm-polygon-sentence-classifier)
- Local project-data root: `/Volumes/Seagate M3/projects/osm-polygon-sentence-classifier`

All local datasets, checkpoints, models, and experiment logs must be kept
beneath the external data root. Quality commands do not download data or
create training outputs. An authorized training call stores model caches,
checkpoints, models, and Trackio state beneath that root.

Run the audit only when the pinned dataset review is explicitly authorized:

```bash
audit-landuse-dataset
```

The command writes `audit_report.json` and `split_manifest.json` beneath
`/Volumes/Seagate M3/projects/osm-polygon-sentence-classifier/audit/landuse`.
Review-required results are written before the command exits with status 2.
Readiness is sentence-level: `mixed_label_polygons` and
`cross_polygon_duplicate_groups` are diagnostics, while duplicate content
hashes crossing the polygon split or carrying both `no` and `yes` labels are
removed by the clean training boundary. Split-level missing-label reasons
remain blockers for the audited source.

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

Grid'5000 execution is plan-first and explicitly guarded. For a side-effect-
free plan, provide the source and model commits:

```bash
uv run grid5000-landuse plan \
  --site nancy \
  --source-commit aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  --model-revision bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
```

Only `submit --execute` can run policy checks, read the remote soft quota, or
submit one bounded one-GPU night allocation. It records local intent before
OAR and fails closed on duplicate or ambiguous state. No live Grid'5000 job,
Hugging Face upload, or publication is performed by the implementation pass.
See [`docs/reference/grid5000-operator.md`](docs/reference/grid5000-operator.md)
for the complete contract.
