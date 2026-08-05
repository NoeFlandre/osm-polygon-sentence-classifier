# Grid'5000 boundary

The repository now contains a plan-first Grid'5000 operator, but no live job
has been submitted by this implementation. Planning is side-effect free. A
real submission is possible only through `submit --execute`, which is an
explicit operator decision and is not inferred from CI, environment variables,
or terminal interactivity.

The execution contract is deliberately narrow:

- the run identity pins the source checkout, dataset revision, model revision,
  model name, and complete training configuration;
- the allocation requests exactly one `gpu` through OAR's `exotic` resource
  type, the `default` queue, an explicit `night` policy, and no more than twelve
  hours;
- immediately before OAR, the operator runs
  `usagepolicycheck -l --sites SITE` and `usagepolicycheck -t`, then reads the
  remote home soft quota;
- execution fails closed unless at least 22 GiB of soft-quota headroom remains;
  it never deletes data automatically;
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

The operator has no automatic retry, checkpoint editing, publication, or Hub
upload path. Remote durable outputs belong under the Grid'5000 user's home;
allocation scratch is not a model or checkpoint store. Stopping local
monitoring is therefore not a reason to resubmit. Reconciliation and validated
resume remain a later milestone.
