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
quota, and OAR operations. This repository implementation has not invoked
that path.

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
prediction, checkpoint editing, or Hub publication is performed.

## Compute-node contract

The scheduled worker runs from the staged clean checkout and validates:

- Linux and a numeric `OAR_JOB_ID`;
- the exact source commit recorded in the identity;
- a clean Git checkout; and
- CUDA availability with exactly one visible GPU.

It then invokes `train_landuse_classifier` with the pinned model revision and a
remote `ProjectConfig` rooted beneath the Grid'5000 user's home. There is no
CPU or MPS fallback. Hugging Face authentication, uploads, and model
publication are outside this operator.

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
