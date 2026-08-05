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
and the finished Trackio project is synchronized to the public static
[Trackio Space](https://huggingface.co/spaces/NoeFlandre/osm-polygon-sentence-classifier-trackio)
through its dedicated Bucket.

The `grid5000-landuse` command is plan-first. It requires pinned source and
model revisions, prints the deterministic run identity, and makes no remote
call unless `submit --execute` is explicitly supplied:

```bash
uv run grid5000-landuse plan \
  --site nancy \
  --source-commit aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  --model-revision bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb \
  --publish \
  --sync-trackio
```

The execution path performs both Grid'5000 usage-policy checks, reads the home
soft quota, records submission intent before OAR, and refuses duplicates or
ambiguous state. With the two explicit flags, it publishes only after local
model output validation and synchronizes completed metrics to the static
Trackio Space. The worker requires a pre-existing Hugging Face login or
`HF_TOKEN` on Grid'5000; it never records the credential. It does not clean
data automatically or retry submissions. Review
[`Grid'5000 operator`](../reference/grid5000-operator.md) before any explicit
execution. For an immediately available daytime GPU, add
`--policy-type day`; it defaults to a 30-minute allocation and is capped at
one hour. Probe reachable sites first and submit exactly one selected site.

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
