# Data policy

## Approved storage root

The only approved local project-data root is:

`/Volumes/Seagate M3/projects/osm-polygon-sentence-classifier`

Future datasets, checkpoints, models, Trackio logs, and run state must live
beneath that root. The repository contains no project data. Runtime path
resolution rejects absolute paths, traversal outside the root, and symlink
escapes. The explicit `audit-landuse-dataset` command is the only current
data-consuming command: it reads the pinned source in streaming mode and
writes its Hugging Face cache plus derived `audit/landuse/audit_report.json`
and `audit/landuse/split_manifest.json` beneath this root.

The source for the training stage is the
[NoeFlandre/osm-polygon-wikidata-sentence-relevance dataset](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-sentence-relevance).
The dedicated model destination is the
[NoeFlandre/osm-polygon-sentence-classifier model repository](https://huggingface.co/NoeFlandre/osm-polygon-sentence-classifier).
Completed metrics use the public static
[Trackio Space](https://huggingface.co/spaces/NoeFlandre/osm-polygon-sentence-classifier-trackio)
and its dedicated Hugging Face Bucket.

## Audit and foundation boundary

The audit is a review-only transformation. Its loader may populate the
approved Hugging Face cache while streaming; the reducer validates and
summarizes rows, and the explicit writer assigns the report and manifest
described above. It does not train, upload or publish artifacts, or submit
Grid'5000 work. The training module is a separate explicit call: it consumes
only the clean iterator and directs model caches, checkpoints, outputs, and
Trackio state beneath the approved root. Publication and static Trackio sync
are separate opt-in flags, occur only after training output is saved, and use
the preconfigured remote destinations.

Credentials are used only by an explicitly authorized external CLI or runtime
authentication flow. They must never be committed to the repository or placed
in its documentation, commands, or durable Grid'5000 run state. A publishing
worker must find an existing Grid'5000 login or `HF_TOKEN` before training.

These two derived artifacts are the only ones the writer produces:
`audit_report.json` contains aggregate and readiness information only and no
raw rows or sentence text, and `split_manifest.json` contains deterministic
polygon-to-split assignments.

## Sentence-level readiness policy

The classifier consumes sentence text, so readiness is evaluated at the
content-hash level rather than by requiring one label per polygon. The audit
reports `mixed_label_polygons` as a diagnostic; different sentences in one
polygon may legitimately have different trainable labels. It also retains
`cross_polygon_duplicate_groups` as a diagnostic.

The readiness blockers are exactly:

- `content_hash_label_conflicts`: one content-hash group contains both
  trainable labels, `no` and `yes`.
- `cross_split_duplicate_groups`: one content-hash group has rows assigned to
  both deterministic polygon splits.
- `<split>_split_missing_label`: a train or validation split lacks `no` or
  `yes` trainable rows.

The report stores these reasons in sorted order. The CLI writes the report and
manifest before exiting with status 2 whenever any blocker remains.

## Clean training input

`iter_clean_training_examples` in `dataset_loader.py` is the only training
input boundary. It receives a factory for fresh streams and the public iterator
stays lazy until consumed. The first fresh stream is fully consumed to discover
sentence-content-hash groups carrying both `no` and `yes`; the second fresh
stream is then consumed to emit clean representatives, keeping only the first
trainable occurrence of every remaining usable hash. Rows are processed
incrementally as they arrive from each stream rather than materialized into a
list, and no cleaned dataset is written. `uncertain` rows are never emitted and
do not create conflicts. Rows without a usable hash keep the existing per-row
behavior. Training code must consume the iterator's output directly.
