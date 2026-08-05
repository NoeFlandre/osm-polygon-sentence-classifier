# Grid'5000 boundary

The primary Grid'5000 boundary is the explicit `run --execute` lifecycle. It
keeps planning side-effect free while making one guarded command responsible
for site discovery, remote preparation, scheduler submission, monitoring,
publication, and successful-run cleanup. The older `plan` and `submit`
commands remain compatibility interfaces.

The autonomous identity pins:

- the exact source Git commit;
- the pinned landuse dataset revision;
- model name and exact model revision; and
- the complete training configuration, including publication choices.

The execution contract is:

- probe all configured frontends with bounded, fixed-argument SSH;
- select a reachable site with a compatible x86_64 GPU and at least 8 GiB of
  persistent soft-quota headroom;
- derive OAR queue, resource type, production property, and Europe/Paris policy
  from live facts and the requested short walltime;
- prepare one clean detached checkout and one marker-owned per-run root;
- install Hugging Face authentication through SSH stdin only;
- run policy and quota checks immediately before one OAR submission;
- record durable intent before OAR and refuse ambiguous resubmission;
- monitor one fallback job, with at most three sequential replacement rounds;
  each round probes all configured sites and runs one 20-minute trial at a time;
- retain two identity-bound checkpoints and submit at most three bounded
  successor jobs when a terminal job has no verified completion;
- verify the completion manifest and remote Hub facts before success cleanup; and
- remove only the exact marked successful run root unless retention is requested.

The controller never uses queue depth as an ETA, submits speculative jobs to
multiple sites, or retries an ambiguous scheduler response. A trial is adopted
only after OAR reports it `Running`; until then the original queued job remains
the fallback. A timed-out or late trial is canceled. A monitor interruption
does not cancel the active scheduler job; `resume --run-id RUN_ID --execute`
reattaches to the durable state.

The compute worker validates Linux, OAR identity, exact checkout revision and
cleanliness, an x86_64 compute-node architecture, exactly one visible CUDA GPU,
and CUDA capability `>= 7.5` before entering the existing training module. The
site probe applies the same architecture and capability floor, while the worker
rechecks the actual assigned device and the executability of `uv`. The locked
`training` dependency extra runs with the uv
environment and package cache in allocation-local `/tmp`. Durable model,
checkpoint, dataset-cache, and Trackio paths remain below the managed remote
run root. Each complete checkpoint carries the immutable run identity. A
successor worker must find the newest complete identity-matching checkpoint and
passes it to `Trainer.train(resume_from_checkpoint=...)`; it never silently
starts a continuation from scratch. The completion manifest contains no token
or raw scheduler output. ARM/aarch64 nodes are deliberately ineligible until
an architecture-specific locked runtime is provided; this prevents an x86_64
uv binary from reaching an incompatible compute node.

The local state root is the approved external data volume:

```text
/Volumes/Seagate M3/projects/osm-polygon-sentence-classifier/grid5000/runs
```

State directories are mode `0700`; state and event files are mode `0600`.
Legacy ambiguous state is never overwritten. It is archived only after
read-only current-user OAR checks find no active jobs, otherwise execution
stops for inspection.
