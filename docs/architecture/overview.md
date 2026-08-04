# Architecture overview

The foundation keeps the first interfaces narrow and local:

- `config.py` defines immutable project metadata, including the landuse task,
  the read-only source dataset identifier, the eventual model repository
  identifier, and the approved data root.
- `paths.py` resolves application paths beneath a containment-checked root and
  rejects absolute, traversal, or escaping-symlink paths.
- `tracking.py` builds Trackio project and directory settings beneath the
  managed root without importing, initializing, or starting a Trackio run.

The following surfaces are intentionally deferred until their contracts and
safety gates have been reviewed: dataset schemas, labels, models, metrics,
checkpoints, Grid5000 operators, and OAR integration. The foundation therefore
describes where those future components may connect without implementing their
data or remote execution behavior.
