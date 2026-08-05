# Grid'5000 boundary

The repository contains a plan-first Grid'5000 operator. Planning is
side-effect free. A real submission is possible only through `submit
--execute`, which is an explicit operator decision and is not inferred from
CI, environment variables, or terminal interactivity.

The execution contract is deliberately narrow:

- the run identity pins the source checkout, dataset revision, model revision,
  model name, and complete training configuration;
- the allocation requests exactly one `gpu` through OAR's `exotic` resource
  type, the `default` queue, and an explicit `day` or `night` policy. Day
  allocations are limited to one hour; night allocations are limited to
  twelve hours;
- immediately before OAR, the operator runs
  `usagepolicycheck -l --sites SITE` and `usagepolicycheck -t`, then reads the
  remote home soft quota;
- execution fails closed unless at least 4 GiB of soft-quota headroom remains;
  this margin covers the final model, one retained checkpoint, metadata, and
  publication artifacts. It never deletes data automatically;
- a local submission intent is atomically recorded beneath
  `/Volumes/Seagate M3/projects/osm-polygon-sentence-classifier/grid5000/runs`;
- an existing submission or ambiguous intent refuses another submission, so
  the operator does not race sites or create speculative duplicates; and
- the compute worker requires a Linux node, the recorded OAR job ID, the exact
  source commit, a clean checkout, and exactly one visible CUDA GPU before
  calling the existing training boundary.

The remote checkout must already be staged beneath the Grid'5000 user's home.
Before OAR, a read-only SSH guard checks that it is at the pinned source commit
and has no working-tree changes; a missing or dirty checkout therefore fails
before an allocation is requested.

The site is an execution choice, not a project limitation. Before a daytime
attempt, probe the reachable configured frontends for a factually idle
compatible GPU and select one site only. Stage the same pinned checkout and
required Hugging Face authentication there first; never submit speculative jobs
to several sites.

The operator has no automatic retry, checkpoint editing, publication, or Hub
upload path. Remote durable outputs belong under the Grid'5000 user's home;
the locked uv environment and its package cache are created in allocation-local
`/tmp` scratch so they do not consume persistent home quota. Scratch is not a
model, checkpoint, or Trackio store. Stopping local monitoring is therefore not
a reason to resubmit. Reconciliation and validated resume remain a later
milestone.
