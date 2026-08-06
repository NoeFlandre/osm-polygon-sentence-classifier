# OSM Polygon Sentence Classifier

Train a sentence classifier for OSM polygon descriptions, starting with the
`landuse` task.

## Status

The repository contains a safe, testable project foundation, one explicit
review command (`audit-landuse-dataset`), a typed training module, and an
autonomous Grid'5000 operator.
The audit consumes the pinned dataset in streaming mode. Its loader may
populate the approved Hugging Face cache; the explicit artifact writer creates
only the JSON report and polygon split manifest beneath the approved external
data root. The training module consumes the clean iterator, uses a Hugging
Face Trainer with local Trackio reporting, and saves model artifacts beneath
the same root. It retains five complete, identity-bound checkpoints so an
incomplete allocation can continue safely. An authorized Grid'5000 run may
explicitly publish the completed model to the dedicated model repository,
publish each complete checkpoint under its run-scoped
`experiments/<experiment>/run-<run-id>/checkpoints/step-N/` directory, and publish Trackio metric snapshots to the free static
[Trackio Space](https://huggingface.co/spaces/NoeFlandre/osm-polygon-sentence-classifier-trackio)
and [metric Bucket](https://huggingface.co/buckets/NoeFlandre/osm-polygon-sentence-classifier-trackio-data).
The static Space is read-only between snapshots and requires no paid HF
compute.
The model repository README is generated from the pinned dataset/model
identity, safe training configuration, checkpoint progress, and scalar
metrics at each published checkpoint and at final publication. Evaluation
metrics include accuracy, precision, recall, and F1.

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
workflow after reviewing the landuse audit. The default model is
[`jhu-clsp/mmBERT-small`](https://huggingface.co/jhu-clsp/mmBERT-small), a
140M-parameter multilingual encoder covering 1,800+ languages. Following the
[FineWeb-Edu classifier recipe](https://github.com/huggingface/cosmopedia/tree/main/classification),
the encoder is frozen and only its binary `no`/`yes` classification head is
trained. Training settings remain configurable through `TrainingConfig`;
Grid'5000 runs must additionally pin a model revision.

The choice is guided by the model authors' [reported gains](https://huggingface.co/blog/mmbert)
over older multilingual encoders on classification and multilingual retrieval
benchmarks; the held-out landuse evaluation remains the decision criterion for
this project.

## Data and model repositories

- Training source: [NoeFlandre/osm-polygon-wikidata-sentence-relevance](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-sentence-relevance)
- Dedicated model repository: [NoeFlandre/osm-polygon-sentence-classifier](https://huggingface.co/NoeFlandre/osm-polygon-sentence-classifier)
- Local project-data root: `/Volumes/Seagate M3/projects/osm-polygon-sentence-classifier`

All local datasets, checkpoints, models, and experiment logs must be kept
beneath the external data root. Quality commands do not download data or
create training outputs. An authorized training call stores model caches,
checkpoints, models, and Trackio state beneath that root. Remote publication is
opt-in: the repository root contains documentation only; final model files are
published beneath `experiments/<experiment>/run-<run-id>/final/`, and every
complete checkpoint is published beneath that run's permanent
`checkpoints/step-N/` directory.
The model repository and free static Trackio Space/Bucket are provisioned as
public destinations. The checkpoint upload is queued in order and drained
before final publication; older checkpoint snapshots remain available for
inspection. On network-backed Grid'5000 storage, Trackio JSONL fragments are
imported before each static snapshot so the public dashboard contains the
recorded metrics. Short Grid'5000 continuations receive names containing the
experiment, immutable run ID, starting checkpoint, and OAR job ID.

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

The autonomous command probes all configured Grid'5000 sites concurrently,
selects a factually compatible x86_64 GPU (including CUDA capability `>= 7.5`),
derives the correct OAR queue/resource type from `oarnodes`, stages the exact
checkout, submits one short job, and monitors it. The worker rechecks the
assigned GPU before training and retains five complete, identity-bound
checkpoints. If a
job ends before producing a verified model, the controller continues only from
the newest complete checkpoint, on the same site, with a bounded successor
job; it never restarts from scratch. The default limit is three continuation
jobs and can be changed with `--max-continuations`. If OAR forecasts the
fallback too late, it probes every configured site, tries one replacement at
a time, and repeats the bounded probe after a ten-minute cooldown, for at most
three rounds. The same search starts immediately when OAR provides no start
prediction. It adopts only a replacement that is visibly running; if an
unpredicted fallback still has not started after those rounds, it is canceled
and the run fails explicitly instead of waiting forever. Queue depth is
recorded for diagnostics, never used as an ETA, and speculative multi-site
submissions are not made.

If a run fails specifically after exhausting its continuation limit, a retained
complete checkpoint can be extended safely with a larger bound:

```bash
uv run grid5000-landuse resume \
  --run-id RUN_ID \
  --max-continuations 6 \
  --execute
```

The extension refuses arbitrary failed states and refuses to submit while the
previous recorded Grid'5000 job is still active. With `--execute`, it requires
a clean checkout and uses that checkout's pinned commit for resumed worker code
while preserving the original run identity and checkpoint contract.

Run the complete lifecycle with a pinned source/model pair:

```bash
uv run grid5000-landuse run \
  --source-commit aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  --model-revision bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb \
  --publish \
  --sync-trackio \
  --execute
```

The default is every configured site, Europe/Paris policy selection, a
20-minute allocation, and cleanup of the managed per-run remote data after
the model commit and Trackio Space are verified. Use repeated `--site` flags
to narrow discovery, `--keep-remote` to retain verified remote evidence, or
`status --run-id RUN_ID` to inspect durable local state. The command also
reconciles the old ambiguous submission format only after read-only user-job
checks across the configured sites; an active job always blocks resubmission.
During execution, human-readable progress is flushed to stderr while the
final state JSON remains on stdout.

For a side-effect-free autonomous plan, omit `--execute`:

```bash
uv run grid5000-landuse plan \
  --site nancy \
  --source-commit aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  --model-revision bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
```

The `run --execute` gate is the only path that performs SSH, policy, quota,
OAR, Hub provisioning, publication, or cleanup. It creates the dedicated
model repository and free static Trackio Space/Bucket idempotently. HF credentials
are sent to the selected frontend only through SSH stdin and never enter a
command or durable state. The worker writes a credential-free completion
manifest before the controller verifies Hub facts and cleans its exact marked
run root. The legacy `submit` command remains available as a lower-level,
single-site compatibility boundary.
See [`docs/reference/grid5000-operator.md`](docs/reference/grid5000-operator.md)
for the complete contract.
