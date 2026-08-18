# Experiment organization

This repository is the code and documentation boundary. Data, checkpoints,
models, and run logs stay outside the checkout or in their dedicated public
Hugging Face repositories.

## Frozen landuse baseline

The `v0.1.0` GitHub release records the completed landuse-v1 work. Its public
artifacts remain available without being moved:

| Concern | Location |
| --- | --- |
| Code and operator | This GitHub repository, pinned by commit |
| Input data | `NoeFlandre/osm-polygon-wikidata-sentence-relevance`, pinned by revision |
| Completed study | Model repository `studies/landuse-v1/` |
| Single-run baseline | Model repository `experiments/landuse-mmbert-small-frozen-head/` |
| Metrics snapshots | The dedicated static Trackio Space and Bucket |

The existing model paths are stable public artifact paths. They are not renamed
or duplicated for a cleanup.

## Input dataset

The dataset repository separates released data from labeling provenance:

```text
v1-afghanistan/                         Preserved V1 release
v2-worldwide/                           Preserved worldwide V2 release
provenance/v2-worldwide/labeling-run-*/  Resumability evidence, not a split
```

The `v2-worldwide/` directory is the planned input for the next classifier
experiment. It is not a training result, and this repository does not yet
implement or run that experiment.

## Future experiment lanes

Every new training lane must have:

1. an immutable input-dataset revision;
2. an immutable GitHub source commit;
3. a stable experiment or study name;
4. a run-scoped artifact directory; and
5. a concise registry entry describing configuration, metrics, and limitations.

Use `experiments/<experiment>/run-<run-id>/` for an individual run and
`studies/<study-id>/<ablation-id>/run-<run-id>/` for a controlled comparison.
The future worldwide classifier lane should use a new identifier rather than
adding files to `landuse-v1`.

## Public organization rules

- Do not commit datasets, checkpoints, model caches, logs, credentials, or
  internal planning notes to the GitHub repository.
- Do not overwrite completed model artifacts or reuse a run ID.
- Pin the exact dataset revision and source commit in every published study.
- Keep validation results distinct from held-out test results; do not imply a
  test result when a study has no held-out test set.
