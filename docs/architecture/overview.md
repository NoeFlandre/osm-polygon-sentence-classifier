# Architecture overview

The foundation keeps the first interfaces narrow and local:

- `config.py` defines immutable project metadata, including the landuse task,
  the read-only source dataset identifier, the eventual model repository
  identifier, and the approved data root.
- `paths.py` resolves application paths beneath a containment-checked root and
  rejects absolute, traversal, or escaping-symlink paths.
- `dataset_contract.py` defines the pinned landuse schema, supported labels,
  and source provenance.
- `dataset_loader.py` exposes the lazy pinned-source loader, deterministic
  polygon split function, and the two-pass `iter_clean_training_examples`
  boundary. Training code must consume only that clean iterator: it
  excludes contradictory sentence-content-hash groups and duplicate
  representatives without writing a cleaned dataset. Internally the clean
  iterator separates a conflict-discovery phase from an emission phase: the
  public iterator stays lazy until consumed; the first fresh stream is then
  fully consumed to discover contradictory hashes; the second fresh stream is
  consumed to emit clean representatives; rows are processed incrementally
  rather than materialized; and no cleaned dataset is written.
- `dataset_audit.py` reduces the stream to immutable counts, sentence-level
  content-hash readiness reasons, and a sorted polygon split manifest;
  `audit_cli.py` writes those derived artifacts beneath `audit/landuse`.
- `tracking.py` builds Trackio project, managed-directory, static-Space, and
  Bucket settings. It also owns the explicit final synchronization boundary;
  normal configuration does not import or start Trackio.
- `training.py` adapts the clean iterator to lazy split-specific Trainer
  records and wires a Hugging Face sequence-classification Trainer. Its
  default `jhu-clsp/mmBERT-small` encoder is multilingual and frozen; only the
  binary classification head is trained, following the FineWeb-Edu pattern.
  Model caches, checkpoints, outputs, and Trackio state are directed beneath
  the managed root. Explicit flags may publish validated final top-level model
  files and synchronize completed metrics; checkpoint directories are not
  published. The module does not submit Grid5000 work.
- `publication.py` validates a completed model directory and performs one
  add-only commit to the configured Hugging Face model repository. It rejects
  incomplete output before any Hub call.
- `grid5000.py` owns the plan-first Grid'5000 boundary: immutable run identity,
  bounded one-GPU allocation, policy and soft-quota preflight, fixed SSH/OAR
  argument construction, a read-only pre-staged-checkout guard, and restrictive
  durable submission state. Plan mode is side-effect free; only an explicit
  `--execute` path can submit, and it refuses existing or ambiguous state.
- `grid5000_worker.py` validates the Linux compute node, OAR job identity,
  exact clean checkout, and one visible CUDA GPU before invoking the existing
  training boundary. It uses a home-scoped remote project root, requires
  existing Hugging Face authentication when publication or Trackio sync is
  enabled, and has no retry, scheduler, or CPU-fallback responsibility.

The audit command is the only current data-consuming CLI command. It consumes
the pinned dataset and writes only its managed cache, report, and manifest. The
training entry point is an explicit call. The Grid'5000 planner consumes no
dataset and makes no remote call; its explicit execution path is documented in
the Grid'5000 operator reference.

Audit readiness is based on sentence-only model inputs. Mixed labels within a
polygon and duplicate hashes across polygons remain diagnostic metrics, while
content-hash label conflicts, content hashes crossing the deterministic split,
and split-level missing-label reasons block readiness. The report and manifest
are written before the CLI returns status 2 for any blocker.
