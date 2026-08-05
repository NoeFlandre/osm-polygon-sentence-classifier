# Grid'5000 landuse training operator design

## Goal

Add a small, guarded operator boundary for one landuse training allocation on
Grid'5000. Planning is the default. A real submission requires an explicit
`--execute` flag, performs live policy and quota checks immediately before
submission, records durable local intent before calling OAR, and refuses to
resubmit when the local state is submitted or ambiguous.

The operator prepares a reproducible request for the already pinned landuse
dataset and the current source checkout. It does not run a job during this
implementation or publish a model to Hugging Face.

## Non-goals

- no live SSH, OAR, Grid'5000, Hugging Face, or Trackio service call;
- no automatic site racing, queue-time prediction, or speculative duplicate
  allocation;
- no automatic cleanup of remote or local data;
- no automatic resubmission or checkpoint editing;
- no model evaluation, model-card publication, or Hub upload;
- no new external dependency or generic arbitrary-command SSH abstraction.

## Safety and reproducibility contract

The implementation follows the read-only sentence-labeling project's
operator contract:

1. A run identity is derived from lowercase 40-character source and dataset
   revisions, the model identifier and revision, and a canonical training
   configuration. The resulting run ID is deterministic and changes when any
   immutable input changes.
2. The first slice requests exactly one GPU through the `exotic` resource type,
   uses the `default` queue, uses a bounded walltime no longer than twelve
   hours, and requests an explicit `night` policy window. The planned command
   contains no credentials or user-provided shell fragment.
3. Execution performs `usagepolicycheck -l --sites SITE` and
   `usagepolicycheck -t`, then checks the Grid'5000 home soft quota. A
   submission is refused when the required headroom is unavailable. The
   operator does not delete anything automatically.
4. State is stored below the approved local data root in a mode-0700 run
   directory with mode-0600 JSON state. The submission intent is written and
   atomically replaced before the OAR command is invoked. A `submitting` state
   without a recorded job ID is ambiguous and fails closed; it must be
   reconciled manually before another submission is attempted.
5. An existing submitted, queued, running, or terminal state for the same run
   ID prevents a second submission. The planned request and immutable identity
   are retained in state for inspection and later reconciliation.
6. The remote worker is a fixed project-specific command. It validates Linux,
   numeric `OAR_JOB_ID`, the exact source commit, a clean checkout, and one
   visible CUDA device before invoking the existing training function. It does
   not fall back to CPU or MPS. Persistent model/cache/checkpoint paths are
   remote operator paths; allocation scratch is not used for durable outputs.

Stopping the local monitor is outside this first slice, but the durable state
boundary is designed so a later status/resume command can inspect the same run
without creating a duplicate.

## Components

### `grid5000.py`

This module owns the pure identity, allocation, command, preflight, and local
state kernel:

- `Grid5000RunIdentity` validates immutable inputs and exposes canonical JSON,
  a SHA-256 fingerprint, and a stable run ID;
- `Grid5000Allocation` validates site, queue, policy, resource type, GPU count,
  and walltime bounds;
- `Grid5000Plan` combines identity, allocation, checkout, and remote paths and
  renders the fixed `oarsub` argument vector;
- `CommandRunner` is a narrow injected protocol used by tests and by the
  explicit execution path; the real implementation uses fixed `subprocess`
  argument vectors and bounded timeouts;
- `Grid5000StateStore` writes and reads only its own run directories using
  atomic replacement and restrictive permissions;
- `Grid5000Operator.plan()` is side-effect free, while
  `Grid5000Operator.submit(execute=False)` returns a plan and performs no
  external calls. `execute=True` is the only route that can run preflight and
  submit.

The first implementation keeps status/resume and remote checkpoint validation
out of the public surface. It records enough identity and submission evidence
to make an ambiguous state safe to stop on.

### `training.py` and `config.py`

`TrainingConfig` gains an optional pinned `model_revision`; when set, it is
passed unchanged to both Hugging Face `from_pretrained` calls. A narrowly
scoped `ProjectConfig.for_remote_root()` constructor supports a remote
Grid'5000 worker root without weakening the default requirement that local
project data live on the Seagate volume. The worker uses that constructor;
ordinary local code continues to use `ProjectConfig()`.

### `grid5000_worker.py`

The worker validates its compute-node environment and invokes
`train_landuse_classifier` with a remote-root project configuration and the
immutable training settings carried by the run plan. It has no scheduler,
network, upload, or retry responsibility.

### CLI

The `grid5000-landuse` command exposes `plan` and `submit`. Both print the
reproducible run identity and scheduler request. `submit` is plan-only unless
`--execute` is supplied. The implementation and tests never invoke the real
command with `--execute`.

## Execution sequence

```text
local plan
  -> canonical identity and fixed oarsub argv
  -> [--execute gate]
  -> usagepolicycheck site + total
  -> soft quota check
  -> write submitting intent atomically
  -> one bounded oarsub call
  -> record returned job id atomically
```

If policy, quota, state, command construction, or output parsing fails, no OAR
submission is made. If OAR returns an error after intent was persisted, the
state remains `submitting` and future submission is refused until reconciled.

## Verification

Tests cover malformed revisions, deterministic identity, allocation bounds,
exact scheduler arguments, policy/quota ordering, plan-only no-call behavior,
atomic state permissions, existing/ambiguous-state refusal, successful
injected submission, remote worker preflight, and model-revision propagation.
All subprocesses are injected or use fixed arguments; no test contacts
Grid'5000. The full repository gate remains:

```bash
RUFF_CACHE_DIR=/tmp/osm-polygon-sentence-classifier-ruff-cache uv run ruff format --check .
RUFF_CACHE_DIR=/tmp/osm-polygon-sentence-classifier-ruff-cache uv run ruff check .
uv run ty check
uv run pytest -q
uv run mkdocs build --strict --site-dir /tmp/osm-polygon-sentence-classifier-site
uv run pre-commit run --all-files
uv lock --check
```

The final review must confirm that only the approved external data root is
used for local artifacts, no `.DS_Store` files are touched, and no live
Grid'5000 or publication command was invoked.
