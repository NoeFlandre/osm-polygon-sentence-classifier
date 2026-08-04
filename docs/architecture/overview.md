# Architecture overview

The foundation keeps the first interfaces narrow and local:

- `config.py` defines immutable project metadata, including the landuse task,
  the read-only source dataset identifier, the eventual model repository
  identifier, and the approved data root.
- `paths.py` resolves application paths beneath a containment-checked root and
  rejects absolute, traversal, or escaping-symlink paths.
- `dataset_contract.py` defines the pinned landuse schema, supported labels,
  and source provenance.
- `dataset_loader.py` exposes the lazy pinned-source loader and deterministic
  polygon split function.
- `dataset_audit.py` reduces the stream to immutable counts, sentence-level
  content-hash readiness reasons, and a sorted polygon split manifest;
  `audit_cli.py` writes those derived artifacts beneath `audit/landuse`.
- `tracking.py` builds Trackio project and directory settings beneath the
  managed root without importing, initializing, or starting a Trackio run.

The audit command is the only current data-consuming surface. It consumes the
pinned dataset, writes only its managed cache, report, and manifest, and never
trains, uploads, publishes, or submits Grid5000 work. Model architecture,
metrics, checkpoints, Grid5000 operators, and OAR integration remain deferred
until their contracts and safety gates have been reviewed.

Audit readiness is based on sentence-only model inputs. Mixed labels within a
polygon and duplicate hashes across polygons remain diagnostic metrics, while
content-hash label conflicts, content hashes crossing the deterministic split,
and split-level missing-label reasons block readiness. The report and manifest
are written before the CLI returns status 2 for any blocker.
