# Grid'5000 operator

## Plan first

The CLI requires pinned source and model revisions. The dataset revision is
fixed by the landuse contract. Replace the example revisions with the exact
lowercase 40-character commits selected for the run:

```bash
uv run grid5000-landuse plan \
  --site nancy \
  --source-commit aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  --model-revision bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
```

Unless `--model-name` is supplied, the run uses
[`jhu-clsp/mmBERT-small`](https://huggingface.co/jhu-clsp/mmBERT-small), a
140M-parameter multilingual encoder whose encoder is frozen during training.
Add `--publish --sync-trackio` to pin final model publication and completed
Trackio synchronization into the run identity:

```bash
uv run grid5000-landuse plan \
  --site nancy \
  --source-commit aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  --model-revision bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb \
  --publish \
  --sync-trackio
```

The command prints JSON containing the deterministic run ID and the exact OAR
request. It does not contact Grid'5000, create local state, load the dataset,
or create model artifacts.

`submit` is still plan-only unless the explicit gate is present:

```bash
uv run grid5000-landuse submit \
  --site nancy \
  --source-commit aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa \
  --model-revision bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb
```

Only a separately reviewed command ending in `--execute` can run SSH, policy,
quota, and OAR operations. The publication flags remain opt-in and are part of
the immutable run identity.

## Execution gates

For an explicit execution, the operator performs these actions in order:

1. refuse an existing or ambiguous local run state;
2. verify the pre-staged remote checkout has the exact source commit and a
   clean working tree;
3. run `usagepolicycheck -l --sites SITE`;
4. run `usagepolicycheck -t`;
5. parse the remote home quota and require 22 GiB of soft headroom;
6. atomically write a `submitting` intent below the approved external data
   root;
7. make one bounded OAR request for one night GPU allocation; and
8. record the single returned job ID.

Any failed check stops before OAR. An OAR error leaves the intent ambiguous;
the operator does not retry automatically. No cleanup, site racing, queue-time
prediction, or checkpoint editing is performed. When the explicit publication
flags are present, the worker validates and uploads only the completed model's
top-level files after training, then synchronizes finished Trackio metrics to
the configured static Space and Bucket. It requires an existing Grid'5000
Hugging Face login or `HF_TOKEN`; credentials are not placed in commands or
state.

## Compute-node contract

The scheduled worker runs from the staged clean checkout and validates:

- Linux and a numeric `OAR_JOB_ID`;
- the exact source commit recorded in the identity;
- a clean Git checkout; and
- CUDA availability with exactly one visible GPU.

It then invokes `train_landuse_classifier` with the pinned model revision and a
remote `ProjectConfig` rooted beneath the Grid'5000 user's home. There is no
CPU or MPS fallback. The worker does not publish checkpoints or retry a failed
publication. The scheduler command uses the pre-staged
`$HOME/.local/bin/uv` executable with `--locked`, while `UV_PROJECT_ENVIRONMENT`
and `UV_CACHE_DIR` point to job-local `/tmp` paths. This keeps the large Linux
CUDA environment off the persistent home quota; model caches, outputs,
checkpoints, and Trackio state remain under the managed home data root.

## Local state and recovery boundary

Local state is kept only under:

```text
/Volumes/Seagate M3/projects/osm-polygon-sentence-classifier/grid5000/runs/<run-id>/state.json
```

The run directory is mode `0700` and the state document is mode `0600`. The
state contains the immutable identity, exact SSH/OAR submission command, phase,
and job ID; it contains no credentials. A later status/resume implementation
must inspect this evidence and live OAR state before reattaching. It must never
infer that an absent job ID is safe to resubmit.
