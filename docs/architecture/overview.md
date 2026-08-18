# Architecture overview

The implementation keeps the interfaces narrow and separates data validity,
training, publication, tracking, and Grid'5000 orchestration:

- `config.py` defines immutable project metadata, including the original
  landuse task,
  the read-only source dataset identifier, the eventual model repository
  identifier, and the approved data root.
- `paths.py` resolves application paths beneath a containment-checked root and
  rejects absolute, traversal, or escaping-symlink paths.
- `dataset_contract.py` defines the pinned landuse and worldwide V2 schemas,
  supported labels, model-input exclusions, and source provenance.
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
  The V2 lane publishes its pinned aggregate audit beside its study protocol;
  it uses an 80/10/10 polygon split and a one-epoch budget derived from the
  clean train count.
- `tracking.py` builds Trackio project, managed-directory, static-Space, and
  Bucket settings. It provisions the free static Space and does not deploy a
  Gradio service; normal configuration does not import or start Trackio.
- `training.py` adapts the clean iterator to lazy split-specific Trainer
  records and wires a Hugging Face sequence-classification Trainer. The pure
  evaluation and model-card metric helpers live in `training_metrics.py`, so
  the orchestration module does not also own metric calculation. The original
  landuse wrapper preserves its step-based validation; the V2 wrapper uses
  epoch validation and evaluates its held-out test split once after training.
  `training_freezing.py` owns the frozen-head and last-two-layer parameter
  policies. Its default `jhu-clsp/mmBERT-small` encoder is multilingual and
  frozen; only the binary classification head is trained, following the
  FineWeb-Edu pattern. `training_publication.py` owns checkpoint manifests,
  checkpoint model cards, Hub publication, and static Trackio updates.
  `training_runtime.py` owns lazy optional-dependency loading, Trainer
  construction, balanced-loss wiring, and resume invocation.
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
  default worker command rejects non-x86_64 compute nodes before invoking the
  locked `uv` runtime. When an image is explicitly selected, the same
  host-side controller instead mounts the clean checkout and per-run data root
  into a non-root Docker/Podman worker after a runtime, image, and GPU
  preflight; it never moves SSH, OAR, policy, quota, Hub, or cleanup into the
  container. `grid5000_sites.py` probes all configured frontends from
  `oarnodes -J`, records CPU architecture, and includes `cpuarch='x86_64'` in
  generated GPU requests. `grid5000_remote.py` stages exact clean checkouts and
  marker-owned data, and recognizes both single-file and sharded checkpoint
  weights when verifying continuation evidence. `grid5000_checkpointing.py`
  owns bounded checkpoint-evidence probes and retry timing, while
  `grid5000_policy.py` owns the pure policy-window and bounded replacement
  decisions.
  `grid5000_oar.py` normalizes scheduler lifecycle facts,
  `grid5000_state.py` persists secure phases/events and recoverable legacy
  reconciliation. `grid5000_replacement.py` owns candidate filtering and
  short-trial coordination, while `grid5000_autonomous.py` coordinates the
  one-command prepare/submit/monitor/verify/cleanup lifecycle and durable
  state transitions. `training_tasks.py` keeps task contracts, training
  defaults, and continuation budgets outside the CLI parser.
- `grid5000_worker.py` validates the Linux compute node, OAR job identity,
  exact clean checkout, and one visible CUDA GPU before invoking the existing
  training boundary. It uses a home-scoped remote project root, requires
  existing Hugging Face authentication when publication or Trackio sync is
  enabled, and has no retry, scheduler, or CPU-fallback responsibility. The
  optional container path supplies the same training dependencies from the
  image and keeps the durable model/data/Trackio paths home-scoped.
- `ablation_study.py` owns immutable study protocols, screening and replication
  order, durable study state, and public run-row preparation. It supports the
  completed `landuse-v1` matrix and the separate worldwide V2 ablation lane.
  `ablation_reporting.py` renders the public README and generated
  `study.json`/`results.json` documents from those prepared rows. Each study
  executes one ablation at a time through the autonomous controller and never
  races multiple jobs. Malformed persisted run records are rejected rather
  than silently ignored.
- `place_relevance_reporting.py` renders the credential-free V2 study protocol,
  exact aggregate data audit, and final metric registry under
  `studies/place-relevance-v2/`. `grid5000_place_relevance_cli.py` exposes the
  separate autonomous V2 command, including its baseline and ablation
  subcommands, without changing the landuse run identity.

The audit command is the only local data-consuming command. The autonomous
Grid'5000 worker consumes the pinned dataset only after the explicit
`run --execute` gate. OAR submission, Hub provisioning, model publication,
Trackio synchronization, and cleanup are each behind the same controller;
plan mode remains side-effect free.

## Public artifact identity

The Hub repository is a catalogue of immutable run outputs rather than one
unnamed “latest” model. A study run name has the form
`landuse-v1|<ablation-id>|seed-<seed>` or
`place-relevance-v2-ablations|<ablation-id>|seed-<seed>`. The ablation ID maps
to one controlled configuration, and the seed distinguishes the screening run
from a replication. The corresponding `run-<run-id>` directory is the
autonomous controller identity; an OAR job ID identifies only one short
allocation segment. `checkpoints/step-N/` is resumable state and `final/` is
the terminal model for that run. The public study report is the human-readable
index, while `study.json` and `results.json` are the machine-readable source
of truth.

Audit readiness is based on sentence-only model inputs. Mixed labels within a
polygon and duplicate hashes across polygons remain diagnostic metrics, while
content-hash label conflicts, content hashes crossing the deterministic split,
and split-level missing-label reasons block readiness. The report and manifest
are written before the CLI returns status 2 for any blocker.
