# Data policy

## Approved storage root

The only approved local project-data root is:

`/Volumes/Seagate M3/projects/osm-polygon-sentence-classifier`

Future datasets, checkpoints, models, Trackio logs, and run state must live
beneath that root. The repository contains no project data. Runtime path
resolution rejects absolute paths, traversal outside the root, and symlink
escapes.

The source for a future read-only training stage is the
[NoeFlandre/osm-polygon-wikidata-sentence-relevance dataset](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-sentence-relevance).
The eventual model destination is the
[NoeFlandre/osm-polygon-sentence-classifier model repository](https://huggingface.co/NoeFlandre/osm-polygon-sentence-classifier).

## Foundation boundary

This foundation does not download or transform the dataset, upload or publish
artifacts, or start remote training. It only defines local configuration,
containment-checked paths, future Trackio directory settings, tests, and
quality/documentation gates.

Credentials are used only by an explicitly authorized external CLI or runtime
authentication flow. They must never be committed to the repository or placed
in its documentation.
