# Grid'5000 operator

The autonomous entry point is `grid5000-landuse`. It owns one pinned landuse
training run from remote preparation through verified publication:

```bash
uv run grid5000-landuse run \
  --source-commit aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  --model-revision bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb \
  --publish \
  --sync-trackio \
  --execute
```

`--source-commit` may be omitted when the local checkout is clean; the command
then uses its current Git `HEAD`. The model revision remains explicit so a run
cannot silently move to a different base model. The dataset revision is fixed
by the landuse dataset contract. The default model is
[`jhu-clsp/mmBERT-small`](https://huggingface.co/jhu-clsp/mmBERT-small), with
the encoder frozen and only the classifier head trained.

## Plan and recovery commands

Without `--execute`, `run` prints a deterministic JSON plan and performs no
SSH, scheduler, Hub, dataset, or local-state mutation:

```bash
uv run grid5000-landuse run \
  --source-commit aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  --model-revision bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
```

The autonomous command defaults to all configured frontends, a 20-minute
allocation, Europe/Paris automatic policy selection, and removal of the
successful run's marked remote data after Hub verification. Repeat `--site` to
restrict discovery, or use `--keep-remote` to retain the successful run root.
It retains five complete checkpoints and allows at most three successor jobs;
override that bound deliberately with `--max-continuations`.

```bash
uv run grid5000-landuse status --run-id RUN_ID
uv run grid5000-landuse resume --run-id RUN_ID --execute
```

When a run failed exactly because it exhausted its continuation limit, extend
that persisted bound explicitly to resume from its retained checkpoint:

```bash
uv run grid5000-landuse resume \
  --run-id RUN_ID \
  --max-continuations 6 \
  --execute
```

The new limit must be greater than the persisted limit. The controller accepts
only that specific checkpoint-limit failure, verifies the previous recorded
job is no longer active, and then reuses the normal checkpoint, policy, quota,
submission, and monitoring safeguards. Other failed states remain stopped for
diagnosis.
With `--execute`, this explicit extension requires a clean local checkout and
uses its current pinned commit for the resumed worker code while preserving the
original run identity and checkpoint contract.

`status` is local and read-only. `resume` uses the persisted identity and
active site/job, and never creates a second submission from a submitted,
queued, or running state. A `submitting` state remains intentionally
ambiguous: inspect scheduler state before taking any manual action. Older
state documents are archived only after read-only current-user OAR checks find
no active jobs; any active job blocks reconciliation.
Executing `run` and `resume` flush human-readable lifecycle and scheduler
progress to stderr; the final state document remains machine-readable JSON on
stdout.

The compatibility `submit` command remains available for a manually selected
site. It is plan-only unless `--execute` is present; new workflows should use
`run`.

## Site discovery and policy

An executing `run` probes every configured site concurrently using bounded SSH
calls. It records parsed facts from `oarnodes -J`, including CPU architecture,
home quota/free space, and a queue-depth diagnostic. Queue depth is never
interpreted as an ETA. A site is eligible only when it is reachable, has a
compatible x86_64 GPU and enough persistent headroom, and its observed GPU facts
support the request. ARM/aarch64 resources are excluded because the locked
remote `uv` runtime is currently x86_64.

The OAR queue, `standard`/`exotic` resource type, and a property combining
`gpu_mem`, `production`, `cpuarch='x86_64'`, and the observed compatible
`gpu_compute_capability` values are derived from the selected site's live
inventory. This prevents OAR from assigning an observed incompatible GPU such
as a P100 or an incompatible CPU architecture. It supports both Grenoble-style
production/standard resources and Nancy-style default/exotic resources without
hard-coding either site. Only one
fallback job is live at a time. If its forecast is more than ten minutes away,
the controller probes every configured site and tries replacement sites
sequentially with a 20-minute trial allocation. A queued fallback without a
scheduler forecast is treated the same way immediately; it must not be polled
indefinitely just because OAR has no prediction. The controller repeats that
bounded probe after a ten-minute cooldown, for at most three rounds. It cancels
a trial that has a known late forecast or reaches its ten-minute observation
deadline, but lets an unpredicted trial use the full observation window. It
adopts a trial only after it is visibly `Running`, and cancels the old fallback
only after adoption. If the fallback still has no forecast, or its known
forecast is still outside the immediate-start window, after all three rounds,
it is canceled and the run fails explicitly rather than waiting forever. Each
new checkpoint successor gets its own replacement decision; an earlier job's
trial does not suppress optimization for a later queued successor.

The complete walltime must fit the selected policy window. During weekdays,
automatic policy uses `day` only for a complete allocation inside 09:00–19:00
Europe/Paris; otherwise it uses `night`. The default walltime is 20 minutes and
day allocations remain capped at one hour. No speculative multi-site jobs or
unbounded retries are used.

## Remote lifecycle

The selected frontend receives a bounded SSH sequence that:

1. clones or reuses the managed repository checkout;
2. fetches the requested commit and verifies a clean detached tree;
3. creates the exact per-run data root with a mode-600 ownership marker;
4. installs the local Hugging Face token through SSH stdin, never a command;
5. rechecks policy, quota, and the exact checkout before OAR submission; and
6. submits the locked worker with the `training` extra and allocation-local
   `/tmp` environment/cache paths.

The worker requires Linux, its numeric OAR job identity, the exact source
commit, a clean checkout, an x86_64 compute node, exactly one visible CUDA GPU,
and CUDA compute capability at least `7.5`, matching the locked training wheel.
The controller uses the same architecture and capability floor during site
selection, and the worker checks the actual assigned device and locked `uv`
executable again before training so an incompatible allocation fails before
consuming the training budget. It streams the pinned dataset
through the clean training iterator, saves five retained
identity-bound checkpoints, and writes the final model and Trackio data beneath
the marked run root. A successor job is submitted only after the previous job
has terminated, the controller has found complete checkpoint evidence, and the
same site passes the storage/policy preflight again. The successor worker
requires a valid checkpoint and passes the newest one to the Trainer; missing,
partial, or identity-mismatched checkpoints stop the run. When requested, the
worker queues each complete checkpoint to the dedicated Hugging Face model
repository under a permanent `checkpoints/step-N/` directory. The Trainer
records accuracy, precision, recall, and F1 locally through Trackio and syncs a static snapshot to the free
Trackio Space and Bucket after each complete checkpoint and final publication,
including continuations. Each checkpoint and the final model also receive a
generated credential-free README containing pinned
identity, safe configuration, progress, and scalar metrics. The ordered Hub
queue is drained before the final top-level model publication; older remote
checkpoint snapshots remain available.

After a successful terminal job, the controller validates the manifest
identity, verifies the recorded model commit and Trackio Space through the Hub,
marks the remote run complete, and removes only that exact marked run root
unless `--keep-remote` was supplied. Failed or ambiguous runs are retained for
diagnosis; an interrupted local monitor leaves scheduler work untouched for
`resume`.

## Local state and credentials

Durable local state is stored only beneath:

```text
/Volumes/Seagate M3/projects/osm-polygon-sentence-classifier/grid5000/runs/<run-id>/
```

The run directory is mode `0700`, the state and event documents are mode
`0600`, and state facts reject credential-like keys. Tokens are read from the
local Hugging Face token file or `HF_TOKEN`, sent to the selected frontend via
stdin, and never written to commands, events, completion manifests, or the
local state JSON.
