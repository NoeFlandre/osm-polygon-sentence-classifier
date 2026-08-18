# Getting started

## Prerequisites

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/)
- Git
- just

The maintainer's Seagate volume is required when running the explicit audit
command or the autonomous operator because durable local run state belongs in
the fixed external data root. The quality commands below do not create project
datasets, checkpoints, models, Trackio run state, or other project data.

The initial `landuse-v1` study is already complete. Its 13 run records, pinned
configuration, full scalar metrics, and final/checkpoint paths are catalogued
in the [ablation study guide](ablations.md) and mirrored in the public
[Hugging Face study report](https://huggingface.co/NoeFlandre/osm-polygon-sentence-classifier/blob/main/studies/landuse-v1/README.md).

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

For a no-data Docker build and smoke test, see the
[Docker runtime guide](docker.md). It also documents the explicit external
data-root mount, token handling, and optional containerized Grid'5000 worker.

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
When explicitly enabled, the final model files go to the
[dedicated model repository](https://huggingface.co/NoeFlandre/osm-polygon-sentence-classifier)
under that run's `final/` directory, and each complete checkpoint updates its
permanent `experiments/<experiment>/run-<run-id>/checkpoints/step-N/`
directory. Trackio publishes a static metric
snapshot to the free public [Trackio Space](https://huggingface.co/spaces/NoeFlandre/osm-polygon-sentence-classifier-trackio)
through its dedicated Bucket after each complete checkpoint and final
publication. On network-backed Grid'5000 storage, append-only Trackio fragments
are imported into the project database before each snapshot, and the final
snapshot closes the active run first. Short continuation jobs are named with
the experiment, run ID, starting checkpoint, and OAR job ID. Evaluation logs
include accuracy, precision, recall, and F1. The model README is generated from
safe run metadata, evaluation metrics, and checkpoint progress at each
checkpoint and final publication. Checkpoint Hub uploads are queued in order
and completed before the final model publication; older checkpoint snapshots
remain available remotely.

## Reading the completed study

Use the public run name, not the scheduler job ID, when interpreting a result:

```text
landuse-v1|a06-last2-256|seed-43
```

The ablation ID names the controlled change, the seed identifies the screening
run or replication, and the separate `run-<run-id>` artifact directory is the
immutable controller identity. A `checkpoints/step-N/` directory is resumable
state; `final/` is the terminal model. The static Trackio dashboard shows
metrics snapshots for these stable runs, while `results.json` is the
authoritative machine-readable metric registry.

The autonomous `grid5000-landuse run` command requires a pinned model revision
and uses the current clean source commit by default. Without `--execute` it
prints a deterministic, side-effect-free plan:

```bash
uv run grid5000-landuse run \
  --source-commit "$(git rev-parse HEAD)" \
  --model-revision abc32620dd4f6ab06f5fbe905dc25f310618e09f \
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
replacement rounds, with one trial at a time and a ten-minute cooldown. A
landuse run allows at most three checkpoint continuations by default. The
worldwide V2 command uses the same safeguards with a separate task identity,
the `place-relevance-v2` Trackio project, epoch-level validation, a held-out
test evaluation at the end, and 40 bounded checkpoint continuations by default:

```bash
uv run grid5000-place-relevance-v2 run \
  --source-commit "$(git rev-parse HEAD)" \
  --model-revision abc32620dd4f6ab06f5fbe905dc25f310618e09f \
  --publish \
  --sync-trackio \
  --execute
```

The V2 baseline publishes its protocol and aggregate data audit under
`studies/place-relevance-v2/`. Its separate ablation study publishes a 13-run
registry under `studies/place-relevance-v2-ablations/`; see the
[V2 ablation guide](place-relevance-v2-ablations.md) for the exact command and
selection rule.
If that limit is exhausted after a complete checkpoint was retained, use
`resume --run-id RUN_ID --max-continuations N --execute` to extend it; the
controller checks that the previous job is no longer active before submitting.

To run the local hooks against the repository, use:

```bash
uv run pre-commit run --all-files
```

Run the explicit audit only after the pinned dataset review is authorized:

```bash
uv run audit-landuse-dataset
```

It streams the training dataset and writes only its approved cache, report, and
split manifest beneath the Seagate root. The authorized Grid'5000 worker also
streams the pinned dataset during training. The audit command itself does not
authenticate, train, or submit a Grid'5000 job. Dependency installation may
use the normal Python package index, but quality commands do not access project
data or remote training systems.
