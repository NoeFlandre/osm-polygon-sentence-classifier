# OSM Polygon Sentence Classifier

Train sentence classifiers for OSM polygon descriptions. The original
Afghanistan `landuse` study and the worldwide V2 `place-relevance` experiment
use separate identities and public study namespaces.

The public [MkDocs documentation](https://noeflandre.github.io/osm-polygon-sentence-classifier/)
contains the guides, architecture notes, and operational reference.

## Status

The first landuse experiment is complete: the reproducible `landuse-v1`
ablation study finished all 13 planned runs, and the original single-run
baseline remains available as a separate experiment. The repository contains
the review command (`audit-landuse-dataset`), typed training boundary, and
autonomous Grid'5000 operator used to produce those artifacts.
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
metrics include accuracy, precision, recall, positive-class F1, macro-F1,
balanced accuracy, and class support.
Trainer checkpoints are saved locally every 100 steps for safe continuation;
only every 1,000-step checkpoint is published to the model repository and
static Trackio by default. If the Hub rate limit is reached, local checkpoint
writing continues and the next allocation can resume from the newest verified
checkpoint.

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

The completed single-run baseline and the completed landuse ablation study share
the public model repository and Trackio dashboard but use separate namespaces
and run names. The idempotent command that runs or resumes the ablation study is:

```bash
uv run grid5000-landuse ablations --execute
```

See the [ablation study guide](docs/guides/ablations.md) for the fixed matrix,
selection rule, completed run registry, and artifact layout. The public Hub
repository contains the same registry under
[`studies/landuse-v1/README.md`](https://huggingface.co/NoeFlandre/osm-polygon-sentence-classifier/blob/main/studies/landuse-v1/README.md).

The best completed finalist family by mean validation positive-class F1 was
`a06-last2-256` (seeds `42`, `43`, and `44`). This is a validation comparison,
not a claim of held-out test performance; the study has no held-out test set.

## Experiment organization

Release `v0.1.0` freezes the completed landuse-v1 work. The input dataset keeps
its released `v1-afghanistan/` and `v2-worldwide/` lanes; labeling checkpoints
are provenance, not training splits. The model repository keeps completed
artifacts under stable `experiments/` and `studies/` namespaces. Future work
must pin a dataset revision and code commit and use a new experiment lane; the
worldwide V2 classifier and its ablations use separate study lanes. See the
[experiment organization guide](docs/guides/experiment-organization.md).

## Worldwide V2 place relevance

The V2 lane trains a binary place-relevance classifier from the pinned
`v2-worldwide` release. It uses only `sentence_text_normalized`, assigns
polygons deterministically to 80/10/10 train/validation/test partitions, removes
contradictory and duplicate sentence-content hashes, evaluates validation at the
end of each streamed epoch, and evaluates the held-out test set once. The
`jhu-clsp/mmBERT-small` encoder is frozen; only its classification head is
trained. The pinned audit leaves 141,283 train, 17,619 validation, and 16,556
test representatives, so one default epoch is 17,661 steps at batch size 8.

Run or resume the complete guarded workflow across the configured Grid'5000
sites with public model and static Trackio publication enabled:

```bash
uv run grid5000-place-relevance-v2 run \
  --source-commit "$(git rev-parse HEAD)" \
  --model-revision abc32620dd4f6ab06f5fbe905dc25f310618e09f \
  --publish \
  --sync-trackio \
  --execute
```

The V2 operator uses short, policy-checked allocations and allows up to 40
bounded checkpoint continuations by default. It never starts speculative jobs
on multiple sites and resumes only from a verified checkpoint. The public
baseline registry is generated under `studies/place-relevance-v2/` with the
protocol, data audit, and final metrics. The separate V2 ablation lane uses
seven screening runs plus six finalist replications and publishes under
`studies/place-relevance-v2-ablations/` with Trackio project
`place-relevance-v2-ablations`:

V2 uses its own public [Trackio Space](https://huggingface.co/spaces/NoeFlandre/osm-polygon-sentence-classifier-v2-trackio)
and [metric Bucket](https://huggingface.co/buckets/NoeFlandre/osm-polygon-sentence-classifier-v2-trackio-data);
the V1 dashboard is not used for V2. Each ablation has exactly one logical
Trackio run. Short allocations reuse that ablation's stable run name and
Trackio resumes the existing series; the OAR job ID and checkpoint path remain
separate operational evidence. This lets a continuation finish the same
experiment without creating another public run.

```bash
uv run grid5000-place-relevance-v2 ablations \
  --source-commit "$(git rev-parse HEAD)" \
  --model-revision abc32620dd4f6ab06f5fbe905dc25f310618e09f \
  --execute
```

See the [V2 ablation study guide](docs/guides/place-relevance-v2-ablations.md)
and the
[Grid'5000 operator reference](docs/reference/grid5000-operator.md).

## How to read a run

Public run names follow
`landuse-v1|<ablation-id>|seed-<seed>`. The ablation ID identifies the one
controlled change, and the seed identifies the screening run or replication.
The separate `run-<run-id>` directory is the immutable controller identity;
it is not an OAR job ID. A run's `checkpoints/step-N/` directories are complete
resumable Trainer checkpoints, while its `final/` directory is the terminal
model. The Hub study `results.json` contains the complete scalar metrics and
`study.json` contains the pinned protocol and provenance.
V2 uses the logical name `place-relevance-v2|baseline|seed-42` for all
continuations. Its `studies/place-relevance-v2/` registry also contains
`data-audit.json`.
V2 ablation runs use
`place-relevance-v2-ablations|<ablation-id>|seed-<seed>` for every
continuation and are catalogued in
`studies/place-relevance-v2-ablations/`.

## Data and model repositories

- Training source: [NoeFlandre/osm-polygon-wikidata-sentence-relevance](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-sentence-relevance)
- Dedicated model repository: [NoeFlandre/osm-polygon-sentence-classifier](https://huggingface.co/NoeFlandre/osm-polygon-sentence-classifier)
- Maintainer-only local project-data root (not a repository directory): `/Volumes/Seagate M3/projects/osm-polygon-sentence-classifier`

All local datasets, checkpoints, models, and experiment logs must be kept
beneath the external data root. Quality commands do not download data or
create training outputs. An authorized training call stores model caches,
checkpoints, models, and Trackio state beneath that root. Remote publication is
opt-in: the repository root contains documentation only; final model files are
published beneath `experiments/<experiment>/run-<run-id>/final/`, and every
complete checkpoint is published beneath that run's permanent
`checkpoints/step-N/` directory.
The model repository and free static Trackio Space/Bucket are provisioned as
public destinations. V1 and V2 use separate Trackio destinations. The
checkpoint upload is queued in order and drained before final publication;
older checkpoint snapshots remain available for inspection. On network-backed
Grid'5000 storage, Trackio JSONL fragments are imported before each static
snapshot so the public dashboard contains the recorded metrics. Short
Grid'5000 continuations keep the same stable Trackio run name and append to
that logical series; their immutable run ID, checkpoint, and OAR job ID are
kept in the run registry and model artifact paths.

Run the audit only when the pinned dataset review is explicitly authorized:

```bash
uv run audit-landuse-dataset
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
The reproducible Docker runtime, explicit mount contract, and optional
Grid'5000 worker mode are documented in
[`docs/guides/docker.md`](docs/guides/docker.md).

## Grid'5000

The autonomous commands probe all configured Grid'5000 sites concurrently,
selects a factually compatible x86_64 GPU (including CUDA capability `>= 7.5`),
derives the correct OAR queue/resource type from `oarnodes`, stages the exact
checkout, submits one short job, and monitors it. The worker rechecks the
assigned GPU before training and retains five complete, identity-bound
checkpoints. If a job ends before producing a verified model, the controller
continues only from
the newest complete checkpoint, on the same site, with a bounded successor
job; it never restarts from scratch. Landuse uses three continuation jobs by
default; worldwide V2 uses 40. Both can be changed with `--max-continuations`.
If OAR forecasts the
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

Run the complete lifecycle with a pinned source/model pair. This example uses
the current committed source revision and the model revision used by the
completed study:

```bash
uv run grid5000-landuse run \
  --source-commit "$(git rev-parse HEAD)" \
  --model-revision abc32620dd4f6ab06f5fbe905dc25f310618e09f \
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
  --source-commit "$(git rev-parse HEAD)" \
  --model-revision abc32620dd4f6ab06f5fbe905dc25f310618e09f
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

## Citation

If you use this software, please cite:

> Flandre, Noe. (2026). *OSM Polygon Sentence Classifier* (version 0.1.0).
> https://github.com/NoeFlandre/osm-polygon-sentence-classifier

The repository also includes [`CITATION.cff`](CITATION.cff), which GitHub uses
to generate the “Cite this repository” entry.
