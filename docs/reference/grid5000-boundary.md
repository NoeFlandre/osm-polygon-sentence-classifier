# Grid5000 boundary

Later training runs are expected to run on Grid5000. This boundary is derived
from the read-only reference project and is a safety contract for a future
operator, not an implementation or submission command.

A later operator must:

- assign an immutable run identity and durable checkpoint state;
- validate the staged runtime before allocating compute;
- perform live usage and quota checks;
- request bounded allocations with scheduler margin;
- preserve and resume the exact run without creating duplicates;
- leave remote work and checkpoints intact if local monitoring stops; and
- publish only after the resulting artifacts have been validated.

The current foundation makes no SSH, OAR, Grid5000, dataset, model, remote
Trackio, or Hugging Face calls. It does not authenticate, allocate resources,
submit jobs, or publish artifacts.
