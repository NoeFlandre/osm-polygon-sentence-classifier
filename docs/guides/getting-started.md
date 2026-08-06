# Getting started

## Prerequisites

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- Git
- just

The Seagate volume is required when running the explicit audit command or the
autonomous operator because durable local run state belongs there. The quality
commands below do not create project datasets, checkpoints, models, Trackio run
state, or other project data.

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

## Training boundary

The package exposes `train_landuse_classifier` and `TrainingConfig` in
`osm_polygon_sentence_classifier.training`. The entry point consumes only the
clean, two-pass iterator, tokenizes lazily through a Hugging Face
`IterableDataset`, reports locally through Trackio, and writes its model
output beneath the approved external data root. Its default model is the
140M-parameter multilingual
[`jhu-clsp/mmBERT-small`](https://huggingface.co/jhu-clsp/mmBERT-small); the
encoder is frozen and only its binary classification head is trained, using
the same efficient pattern as
[FineWeb-Edu](https://github.com/huggingface/cosmopedia/tree/main/classification).
By default, model publication and remote metric synchronization are disabled.
When explicitly enabled, the completed top-level model files go to the
[dedicated model repository](https://huggingface.co/NoeFlandre/osm-polygon-sentence-classifier)
each complete checkpoint updates its permanent
`checkpoints/step-N/` directory, and Trackio publishes a static metric
snapshot to the free public [Trackio Space](https://huggingface.co/spaces/NoeFlandre/osm-polygon-sentence-classifier-trackio)
through its dedicated Bucket after each complete checkpoint and final
publication. On network-backed Grid'5000 storage, append-only Trackio fragments
are imported into the project database before each snapshot, and the final
snapshot closes the active run first. Evaluation logs include accuracy,
precision, recall, and F1. The model README is generated from safe run metadata,
evaluation metrics, and
checkpoint progress at each checkpoint and final publication. Checkpoint Hub
uploads are queued in order and completed before the final model publication;
older checkpoint snapshots remain available remotely.

The autonomous `grid5000-landuse run` command requires a pinned model revision
and uses the current clean source commit by default. Without `--execute` it
prints a deterministic, side-effect-free plan:

```bash
uv run grid5000-landuse run \
  --source-commit aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  --model-revision bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb \
  --publish \
  --sync-trackio
```

Add `--execute` to perform the complete lifecycle. It probes all configured
sites, derives the live resource class, performs both Grid'5000 usage-policy
checks, reads the home soft quota, records durable intent before OAR, and
refuses unsafe duplicate or ambiguous submissions. With the two publication
flags, it creates the dedicated model/Trackio destinations idempotently,
publishes each complete checkpoint and then the validated final model, verifies
the Hub results, and
cleans only the marked per-run remote root. The worker receives HF auth over
SSH stdin and never records the credential. It retains five complete,
identity-bound checkpoints; after an incomplete terminal job, the controller
submits a bounded successor on the same site and the worker resumes from the
newest valid checkpoint. It never restarts a continuation without checkpoint
evidence. Review
[`Grid'5000 operator`](../reference/grid5000-operator.md) before any explicit
execution. The default is a 20-minute allocation with Europe/Paris
`auto` policy; `--policy-type day` is capped at one hour. A distant queued
fallback is rechecked across all configured sites in at most three bounded
replacement rounds, with one trial at a time and a ten-minute cooldown. A run
allows at most three checkpoint continuations by default.
If that limit is exhausted after a complete checkpoint was retained, use
`resume --run-id RUN_ID --max-continuations N --execute` to extend it; the
controller checks that the previous job is no longer active before submitting.

To run the local hooks against the repository, use:

```bash
uv run pre-commit run --all-files
```

The explicit `audit-landuse-dataset` command is the local review command that
streams the training dataset; it writes only its approved cache, report, and
split manifest beneath the Seagate root. The authorized Grid'5000 worker also
streams the pinned dataset during training. The audit command itself does not
authenticate, train, or submit a Grid'5000 job. Dependency installation may
use the normal Python package index, but quality commands do not access project
data or remote training systems.
