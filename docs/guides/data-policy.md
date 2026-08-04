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

The source for a future read-only training stage is the
[NoeFlandre/osm-polygon-wikidata-sentence-relevance dataset](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-sentence-relevance).
The eventual model destination is the
[NoeFlandre/osm-polygon-sentence-classifier model repository](https://huggingface.co/NoeFlandre/osm-polygon-sentence-classifier).

## Audit and foundation boundary

The audit is a review-only transformation. It validates and summarizes rows,
assigns a deterministic polygon split, and writes only the report and manifest
described above. It does not train, upload or publish artifacts, or submit
Grid'5000 work. The remaining foundation defines local configuration,
containment-checked paths, future Trackio directory settings, tests, and
quality/documentation gates.

Credentials are used only by an explicitly authorized external CLI or runtime
authentication flow. They must never be committed to the repository or placed
in its documentation.
