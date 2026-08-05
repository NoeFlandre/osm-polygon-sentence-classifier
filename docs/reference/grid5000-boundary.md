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

The current code makes no SSH, OAR, Grid5000, or publication calls. The
separate review-only audit is the only command that streams the source dataset.
The explicit training function can load a configured model and train locally,
but it does not authenticate, upload artifacts, allocate resources, submit
jobs, or publish results.
