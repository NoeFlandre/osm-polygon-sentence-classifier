# Grid'5000 operator

The autonomous entry point is `grid5000-landuse`. It owns one pinned landuse
training run from remote preparation through verified publication:

```bash
uv run grid5000-landuse run \
  --source-commit "$(git rev-parse HEAD)" \
  --model-revision abc32620dd4f6ab06f5fbe905dc25f310618e09f \
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

## Worldwide V2 place relevance

The separate `grid5000-place-relevance-v2` entry point owns the worldwide
place-relevance experiment. It uses the same policy, quota, site-probing,
checkpoint, cleanup, and publication safeguards, but has its own task identity,
Trackio project, model namespace, and public study registry:

```bash
uv run grid5000-place-relevance-v2 run \
  --source-commit "$(git rev-parse HEAD)" \
  --model-revision abc32620dd4f6ab06f5fbe905dc25f310618e09f \
  --publish \
  --sync-trackio \
  --execute
```

The V2 contract pins dataset revision
`4d0d2b5d53630c24acfb280e9d8159bf6ed0d3fa` and uses only
`sentence_text_normalized`. Polygon IDs determine deterministic 80/10/10
train/validation/test splits with seed 42. The audited clean counts are
141,283 / 17,619 / 16,556, so the default one streamed epoch is 17,661 steps
at batch size 8. Validation runs at each logical epoch end; the held-out test
set is evaluated once after training. The public study files include the exact
aggregate audit and are written beneath `studies/place-relevance-v2/`.

V2 uses the stable Trackio project `place-relevance-v2` and run name
`place-relevance-v2|baseline|seed-42`, so checkpoint continuations extend one
logical public run instead of creating one run per OAR allocation. Its default
continuation bound is 40; all replacements remain sequential and policy
checked.

### Worldwide V2 ablations

The separate ablation command applies the same safeguards to 13 controlled
runs. It uses the `place-relevance-v2-ablations` Trackio project and publishes
under `studies/place-relevance-v2-ablations/`:

```bash
uv run grid5000-place-relevance-v2 ablations \
  --source-commit "$(git rev-parse HEAD)" \
  --model-revision abc32620dd4f6ab06f5fbe905dc25f310618e09f \
  --publish \
  --sync-trackio \
  --execute
```

Seven seed-42 screening runs are followed by seed-43 and seed-44 replications
of the baseline and the two best screening variants. Validation positive-class
F1 selects finalists; the held-out test split is evaluated after each run and
does not affect selection. The command is idempotent and resumes one study run
from its newest verified checkpoint after interruption.

## Plan and recovery commands

Without `--execute`, `run` prints a deterministic JSON plan and performs no
SSH, scheduler, Hub, dataset, or local-state mutation:

```bash
uv run grid5000-landuse run \
  --source-commit "$(git rev-parse HEAD)" \
  --model-revision abc32620dd4f6ab06f5fbe905dc25f310618e09f
```

The autonomous command defaults to all configured frontends, a 20-minute
allocation, Europe/Paris automatic policy selection, and removal of the
successful run's marked remote data after Hub verification. Repeat `--site` to
restrict discovery, or use `--keep-remote` to retain the successful run root.
It retains five complete checkpoints. Landuse allows at most three successor
jobs by default; worldwide V2 allows 40. Override either bound deliberately
with `--max-continuations`.

Grid'5000 home storage is site-local. A continuation first uses the newest
identity-matching checkpoint on the selected site; when that site has no local
copy and Hub publication is enabled, it restores the newest complete
identity-matching checkpoint from the model repository. This keeps a site
replacement resumable without assuming that another site's home directory is
mounted or shared.

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

## Ablation study

The multi-run workflow uses the same guarded lifecycle one ablation at a time:

```bash
uv run grid5000-landuse ablations --execute
```

It persists its own study state, keeps the existing model repository and
Trackio Space, and scopes model artifacts under `studies/landuse-v1/`. The
ablation run names identify the variant and seed. Each Grid'5000 allocation
restores the previous static Trackio snapshot before logging, so a new isolated
allocation does not erase earlier runs from the public dashboard. The command
is plan-only without `--execute` and is safe to repeat after interruption.
The completed `landuse-v1` registry contains 13 runs; interpret the stable
`landuse-v1|<ablation-id>|seed-<seed>` name and its `run-<run-id>` artifact
directory in the [public study report](https://huggingface.co/NoeFlandre/osm-polygon-sentence-classifier/blob/main/studies/landuse-v1/README.md).
An OAR job ID belongs only to one short allocation segment and is not the
experiment identity.

## Site discovery and policy

An executing `run` probes every configured site concurrently using bounded SSH
calls. It records parsed facts from `oarnodes -J`, including CPU architecture,
home quota/free space, and a queue-depth diagnostic. Queue depth is never
interpreted as an ETA. A site is eligible only when it is reachable, has a
compatible x86_64 GPU and enough persistent headroom, and its observed GPU facts
support the request. ARM/aarch64 resources are excluded because the locked
remote `uv` runtime and the reproducible container image target x86_64.

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

The requested policy is stored separately from the scheduler’s derived `day`
or `night` flag, so an `auto` run recomputes its policy on every continuation.
For a run created before this distinction was recorded, pass
`--policy-type auto` to `resume` if it was launched with automatic policy
selection.

## Remote lifecycle

The selected frontend receives a bounded SSH sequence that:

1. clones or reuses the managed repository checkout;
2. fetches the requested commit and verifies a clean detached tree;
3. creates the exact per-run data root with a mode-600 ownership marker;
4. installs the local Hugging Face token through SSH stdin, never a command;
5. rechecks policy, quota, and the exact checkout before OAR submission; and
6. submits either the default locked `uv` worker with the `training` extra and
   allocation-local `/tmp` environment/cache paths, or the explicitly selected
   preloaded Docker/Podman image with the clean checkout and per-run data root
   mounted into it.

The worker requires Linux, its numeric OAR job identity, the exact source
commit, a clean checkout, an x86_64 compute node, exactly one visible CUDA GPU,
and CUDA compute capability at least `7.5`, matching the locked training wheel.
The controller uses the same architecture and capability floor during site
selection, and the worker checks the actual assigned device and either the
locked `uv` executable or the selected container runtime/image again before
training so an incompatible allocation fails before consuming the training
budget. It streams the pinned dataset
through the clean training iterator, saves five retained
identity-bound checkpoints, and writes the final model and Trackio data beneath
the marked run root. A successor job is submitted only after the previous job
has terminated, the controller has found complete checkpoint evidence, and the
same site passes the storage/policy preflight again. The successor worker
requires a valid local or published checkpoint and passes the newest one to the
Trainer; missing, partial, or identity-mismatched checkpoints stop the run.
When requested, the
worker queues each complete checkpoint to the dedicated Hugging Face model
repository under the run-scoped permanent
`experiments/<experiment>/run-<run-id>/checkpoints/step-N/` directory. The Trainer
records accuracy, precision, recall, positive-class F1, macro-F1, balanced
accuracy, and class support locally through Trackio and syncs a static snapshot to the free
Trackio Space and Bucket after each complete checkpoint and final publication,
including continuations. Each checkpoint and the final model also receive a
generated credential-free README containing pinned
identity, safe configuration, progress, and scalar metrics. The ordered Hub
queue is drained before the final model publication; older remote checkpoint
snapshots remain available. The repository root is documentation only; final
model files live in the same run's `final/` directory. Trackio names each short
allocation segment with the experiment, run ID, starting checkpoint, and OAR
job ID so continuations are distinguishable in the public dashboard.

After a successful terminal job, the controller validates the manifest
identity, verifies the recorded model commit and Trackio Space through the Hub,
marks the remote run complete, and removes only that exact marked run root
unless `--keep-remote` was supplied. Failed or ambiguous runs are retained for
diagnosis; an interrupted local monitor leaves scheduler work untouched for
`resume`.

## Local state and credentials

On the maintainer's local volume, durable state is stored only beneath:

```text
/Volumes/Seagate M3/projects/osm-polygon-sentence-classifier/grid5000/runs/<run-id>/
```

The run directory is mode `0700`, the state and event documents are mode
`0600`, and state facts reject credential-like keys. Tokens are read from the
local Hugging Face token file or `HF_TOKEN`, sent to the selected frontend via
stdin, and never written to commands, events, completion manifests, or the
local state JSON.
