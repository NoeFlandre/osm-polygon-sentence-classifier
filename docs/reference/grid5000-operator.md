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

The autonomous command defaults to all configured frontends, a 30-minute
allocation, Europe/Paris automatic policy selection, and removal of the
successful run's marked remote data after Hub verification. Repeat `--site` to
restrict discovery, or use `--keep-remote` to retain the successful run root.
It retains two complete checkpoints and allows at most three successor jobs;
override that bound deliberately with `--max-continuations`.

```bash
uv run grid5000-landuse status --run-id RUN_ID
uv run grid5000-landuse resume --run-id RUN_ID --execute
```

`status` is local and read-only. `resume` uses the persisted identity and
active site/job, and never creates a second submission from a submitted,
queued, or running state. A `submitting` state remains intentionally
ambiguous: inspect scheduler state before taking any manual action. Older
state documents are archived only after read-only current-user OAR checks find
no active jobs; any active job blocks reconciliation.

The compatibility `submit` command remains available for a manually selected
site. It is plan-only unless `--execute` is present; new workflows should use
`run`.

## Site discovery and policy

An executing `run` probes every configured site concurrently using bounded SSH
calls. It records parsed facts from `oarnodes -J`, home quota/free space, and a
queue-depth diagnostic. Queue depth is never interpreted as an ETA. A site is
eligible only when it is reachable, has a compatible GPU and enough persistent
headroom, and its observed GPU facts support the request.

The OAR queue, `standard`/`exotic` resource type, and
`gpu_mem>=... AND production='YES|NO'` property are derived from the selected
site's live inventory. This supports both Grenoble-style production/standard
resources and Nancy-style default/exotic resources without hard-coding either
site. Only one fallback job is submitted. If its forecast is more than ten
minutes away, the controller tries replacement sites sequentially with a
20-minute trial allocation. It cancels a trial that misses its immediate-start
window or reaches its deadline, adopts a trial only after it is visibly
`Running`, and cancels the old fallback only after adoption.

The complete walltime must fit the selected policy window. During weekdays,
automatic policy uses `day` only for a complete allocation inside 09:00–19:00
Europe/Paris; otherwise it uses `night`. The default walltime is 30 minutes and
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
commit, a clean checkout, and exactly one visible CUDA GPU. It streams the
pinned dataset through the clean training iterator, saves two retained
identity-bound checkpoints, and writes the final model and Trackio data beneath
the marked run root. A successor job is submitted only after the previous job
has terminated, the controller has found complete checkpoint evidence, and the
same site passes the storage/policy preflight again. The successor worker
requires a valid checkpoint and passes the newest one to the Trainer; missing,
partial, or identity-mismatched checkpoints stop the run. When requested, the
worker publishes the final model to the dedicated Hugging Face model repository
and synchronizes Trackio to its static Space and Bucket.

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
