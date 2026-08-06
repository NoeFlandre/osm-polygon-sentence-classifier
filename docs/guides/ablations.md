# Landuse ablation study

The ablation workflow is separate from the completed single-run baseline. It
publishes to the same public model repository and Trackio dashboard, but uses
the `studies/landuse-v1/` model namespace and run names beginning with
`landuse-v1|`.

## Study matrix

The screening stage changes one factor at a time from the frozen-head control:

| ID | Change |
|---|---|
| `a00-baseline-head-256-lr3e-4` | Frozen encoder, 256 tokens, learning rate `3e-4` |
| `a01-head-128` | Maximum length `128` |
| `a02-head-512` | Maximum length `512` |
| `a03-head-lr1e-4` | Learning rate `1e-4` |
| `a04-head-lr1e-3` | Learning rate `1e-3` |
| `a05-balanced-head` | Class-balanced cross-entropy |
| `a06-last2-256` | Unfreeze the last two encoder layers |

All seven screening runs use seed `42`. The control and the two highest
positive-class F1 variants are then replicated with seeds `43` and `44`. This
produces 13 distinct runs: the seven screening runs plus six replications.
The screening stage is exploratory; the replications are the confirmatory
comparison.

The dataset revision, polygon split, clean iterator, model revision, maximum
steps, checkpoint cadence, and batch sizes are fixed. The primary selection
metric is positive-class F1 (`eval_f1`), with macro-F1 (`eval_macro_f1`) as the
tie-break. Accuracy, precision, recall, macro-F1, balanced accuracy, and class
support are recorded. There is no held-out test set, so the report describes
validation results only.

## Run the study

The command is plan-only unless `--execute` is present. With a clean checkout,
the source revision and pinned model revision use their defaults:

```bash
uv run grid5000-landuse ablations --execute
```

To inspect the exact plan without submitting jobs:

```bash
uv run grid5000-landuse ablations
```

The controller persists study state beneath the approved external data root.
Running the same command again resumes the current ablation or continues with
the next one. It never submits more than one Grid'5000 job at a time. Each
allocation uses the existing bounded site search, policy and quota preflight,
checkpoint continuation, architecture guard, and exact marked-root cleanup.

The study specification is immutable by default. If an incomplete study must
adopt a source revision containing a worker bug fix, pass
`--allow-source-commit-update` together with the new pinned `--source-commit`.
This is accepted only when the dataset, model, ablation matrix, and study
settings are unchanged and no recorded run is active. Completed run records
retain their original source revision in the study state history.

The default allocation is a 20-minute GPU job with at most six checkpoint
continuations. Complete checkpoints are uploaded under the ablation's model
namespace, and Trackio is synchronized after every checkpoint and at final
publication.

## Public artifacts

The model repository is
[NoeFlandre/osm-polygon-sentence-classifier](https://huggingface.co/NoeFlandre/osm-polygon-sentence-classifier).
The study report is written to:

```text
studies/landuse-v1/README.md
studies/landuse-v1/study.json
studies/landuse-v1/results.json
studies/landuse-v1/<ablation-id>/run-<run-id>/final/
studies/landuse-v1/<ablation-id>/run-<run-id>/checkpoints/step-N/
```

Metrics remain in the existing public
[Trackio dashboard](https://huggingface.co/spaces/NoeFlandre/osm-polygon-sentence-classifier-trackio).
The dashboard is a static snapshot on the free account. Refreshing it shows
the latest completed checkpoint synchronization; it is not a per-log live
stream.

Grid'5000 resources are used for the computation. Publications based on the
study should include the official Grid'5000 acknowledgment.
