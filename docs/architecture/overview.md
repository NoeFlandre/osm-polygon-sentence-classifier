# Architecture overview

The implementation keeps the interfaces narrow and separates data validity,
training, publication, tracking, and Grid'5000 orchestration:

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
  Bucket settings. It provisions the free static Space and does not deploy a
  Gradio service; normal configuration does not import or start Trackio.
- `training.py` adapts the clean iterator to lazy split-specific Trainer
  records and wires a Hugging Face sequence-classification Trainer. Its
  default `jhu-clsp/mmBERT-small` encoder is multilingual and frozen; only the
  binary classification head is trained, following the FineWeb-Edu pattern.
  Model caches, five retained checkpoints, outputs, and Trackio state are
  directed beneath the managed root. Checkpoints carry the immutable run
  identity so a later Grid5000 worker can resume safely. Explicit flags publish
  each complete checkpoint to its permanent
  `experiments/<experiment>/run-<run-id>/checkpoints/step-N/` directory and
  records accuracy, precision, recall, and F1 locally through Trackio. After
  each complete checkpoint and final publication it imports any network-safe
  Trackio JSONL fragments, then explicitly syncs a static snapshot to the public
  Space and Bucket; the final sync closes the active Trackio run first. It
  generates a credential-free model README
  with checkpoint progress and scalar evaluation metrics, then publishes
  validated final model files beneath the same run directory and a
  documentation-only root README. Checkpoint
  uploads are drained before final publication; older remote snapshots remain
  available. The module does not submit
  Grid5000 work.
- `publication.py` validates complete checkpoints and final model directories,
  rejects symlinked model output, renders the safe model README, and performs
  add-only commits to the configured Hugging Face model repository. It rejects
  incomplete or unsafe output before any Hub call.
- `grid5000.py` keeps the immutable identity, policy-bounded allocation, fixed
  SSH/OAR argument construction, and compatibility submission boundary. Its
  worker command rejects non-x86_64 compute nodes before invoking the locked
  runtime. `grid5000_sites.py` probes all configured frontends from
  `oarnodes -J`, records CPU architecture, and includes `cpuarch='x86_64'` in
  generated GPU requests. `grid5000_remote.py` stages exact clean checkouts and
  marker-owned data, and recognizes both single-file and sharded checkpoint
  weights when verifying continuation evidence,
  `grid5000_oar.py` normalizes scheduler lifecycle facts,
  `grid5000_state.py` persists secure phases/events and recoverable legacy
  reconciliation, and `grid5000_autonomous.py` coordinates the one-command
  prepare/submit/monitor/verify/cleanup lifecycle.
- `grid5000_worker.py` validates the Linux compute node, OAR job identity,
  exact clean checkout, and one visible CUDA GPU before invoking the existing
  training boundary. It uses a home-scoped remote project root, requires
  existing Hugging Face authentication when publication or Trackio sync is
  enabled, and bootstraps the locked uv environment and package cache in
  allocation-local scratch. Durable model/data/Trackio paths remain
  home-scoped, and it has no retry, scheduler, or CPU-fallback responsibility.
- `ablation_study.py` owns the immutable `landuse-v1` matrix, screening and
  replication order, durable study state, public run registry, and generated
  `study.json`/`results.json` documents. It executes one ablation at a time
  through the autonomous controller and never races multiple jobs. Malformed
  persisted run records are rejected rather than silently ignored.

The audit command is the only local data-consuming command. The autonomous
Grid'5000 worker consumes the pinned dataset only after the explicit
`run --execute` gate. OAR submission, Hub provisioning, model publication,
Trackio synchronization, and cleanup are each behind the same controller;
plan mode remains side-effect free.

## Public artifact identity

The Hub repository is a catalogue of immutable run outputs rather than one
unnamed “latest” model. A study run name has the form
`landuse-v1|<ablation-id>|seed-<seed>`. The ablation ID maps to one controlled
configuration, and the seed distinguishes the screening run from a
replication. The corresponding `run-<run-id>` directory is the autonomous
controller identity; an OAR job ID identifies only one short allocation
segment. `checkpoints/step-N/` is resumable state and `final/` is the terminal
model for that run. The public study report is the human-readable index, while
`study.json` and `results.json` are the machine-readable source of truth.

Audit readiness is based on sentence-only model inputs. Mixed labels within a
polygon and duplicate hashes across polygons remain diagnostic metrics, while
content-hash label conflicts, content hashes crossing the deterministic split,
and split-level missing-label reasons block readiness. The report and manifest
are written before the CLI returns status 2 for any blocker.
