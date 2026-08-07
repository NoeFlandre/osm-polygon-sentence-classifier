# Landuse ablation study

`landuse-v1` is the completed reproducible comparison of the first landuse
sentence-classifier configurations. It contains 13 completed runs: seven
screening runs with seed `42`, followed by two replications of the control and
the two strongest non-control screening variants with seeds `43` and `44`.

The public model repository contains the machine-readable protocol and results
as well as the final model and checkpoint for every run:

- [Study report on Hugging Face](https://huggingface.co/NoeFlandre/osm-polygon-sentence-classifier/blob/main/studies/landuse-v1/README.md)
- [Study specification](https://huggingface.co/NoeFlandre/osm-polygon-sentence-classifier/blob/main/studies/landuse-v1/study.json)
- [Complete results](https://huggingface.co/NoeFlandre/osm-polygon-sentence-classifier/blob/main/studies/landuse-v1/results.json)
- [Static Trackio dashboard](https://huggingface.co/spaces/NoeFlandre/osm-polygon-sentence-classifier-trackio)

## Protocol

The study changes one factor at a time from the frozen-head control:

| ID | Controlled change |
|---|---|
| `a00-baseline-head-256-lr3e-4` | Frozen encoder, 256-token maximum, learning rate `3e-4` |
| `a01-head-128` | Maximum length `128` |
| `a02-head-512` | Maximum length `512` |
| `a03-head-lr1e-4` | Learning rate `1e-4` |
| `a04-head-lr1e-3` | Learning rate `1e-3` |
| `a05-balanced-head` | Class-balanced cross-entropy |
| `a06-last2-256` | Unfreeze the last two encoder layers |

All runs use the same pinned dataset revision, model revision, clean training
iterator, polygon split, maximum training steps, batch sizes, and checkpoint
cadence. The primary screening and selection metric is positive-class F1
(`eval_f1`); macro-F1 (`eval_macro_f1`) is the tie-break. Accuracy, precision,
recall, balanced accuracy, and class support are also recorded.

The screening ranking selected the control plus `a06-last2-256` and
`a03-head-lr1e-4` for replication. This selection was made from seed `42`
only, before looking at replication results.

## How to read names and paths

Every public study run has the stable name
`landuse-v1|<ablation-id>|seed-<seed>`.

- `<ablation-id>` identifies the controlled configuration in the matrix above.
- `<seed>` distinguishes the screening run (`42`) from its replications (`43`,
  `44`).
- `run-<run-id>` is the immutable autonomous-controller identity. It is not an
  OAR job ID. If a short Grid'5000 continuation is needed, it keeps the same
  study run ID while receiving a new scheduler job ID.
- `checkpoints/step-N/` is a complete, identity-checked Trainer checkpoint at
  global step `N`.
- `final/` is the terminal model directory for that study run.

The standard path is:

```text
studies/landuse-v1/<ablation-id>/run-<run-id>/
├── checkpoints/step-N/
└── final/
```

The root `README.md` is a catalogue, `study.json` is the immutable protocol
and provenance record, and `results.json` is the complete scalar-metrics
registry. The generated README inside each final/checkpoint directory carries
the corresponding identity and metrics.

## Completed run registry

The following table is the human-readable registry. `F1` is positive-class
F1. Validation support is shown as `no / yes`. For the full metric set,
including losses and runtime, use `results.json`.

| Public run name | Seed | Accuracy | Precision | Recall | F1 | Macro F1 | Balanced accuracy | Final artifact (relative to `studies/landuse-v1/`) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `landuse-v1\|a00-baseline-head-256-lr3e-4\|seed-42` | 42 | 0.7651 | 0.4231 | 0.7086 | 0.5298 | 0.6866 | 0.7433 | `a00-baseline-head-256-lr3e-4/run-512ae0ed99cada394e92/final/` |
| `landuse-v1\|a01-head-128\|seed-42` | 42 | 0.7287 | 0.3835 | 0.7446 | 0.5062 | 0.6596 | 0.7348 | `a01-head-128/run-a832ace9a2828d3336ce/final/` |
| `landuse-v1\|a02-head-512\|seed-42` | 42 | 0.7712 | 0.4283 | 0.6725 | 0.5233 | 0.6864 | 0.7332 | `a02-head-512/run-2cb999756d0aaa9fb5cd/final/` |
| `landuse-v1\|a03-head-lr1e-4\|seed-42` | 42 | 0.7844 | 0.4468 | 0.6475 | 0.5287 | 0.6945 | 0.7317 | `a03-head-lr1e-4/run-420de3e6eee0cf426835/final/` |
| `landuse-v1\|a04-head-lr1e-3\|seed-42` | 42 | 0.7614 | 0.4149 | 0.6766 | 0.5144 | 0.6781 | 0.7287 | `a04-head-lr1e-3/run-9dc101ddf09ac6b050cd/final/` |
| `landuse-v1\|a05-balanced-head\|seed-42` | 42 | 0.6734 | 0.3467 | 0.8464 | 0.4919 | 0.6256 | 0.7400 | `a05-balanced-head/run-f3f05c5da666cbfee65e/final/` |
| `landuse-v1\|a06-last2-256\|seed-42` | 42 | 0.7868 | 0.4563 | 0.7382 | 0.5640 | 0.7115 | 0.7681 | `a06-last2-256/run-00dd441f018671492658/final/` |
| `landuse-v1\|a00-baseline-head-256-lr3e-4\|seed-43` | 43 | 0.8364 | 0.6044 | 0.5010 | 0.5479 | 0.7240 | 0.7101 | `a00-baseline-head-256-lr3e-4/run-c67387ed1d5139bed5b4/final/` |
| `landuse-v1\|a00-baseline-head-256-lr3e-4\|seed-44` | 44 | 0.8371 | 0.6387 | 0.4687 | 0.5406 | 0.7208 | 0.7002 | `a00-baseline-head-256-lr3e-4/run-4da9f4b0bb72bd8e1970/final/` |
| `landuse-v1\|a06-last2-256\|seed-43` | 43 | 0.8712 | 0.6840 | 0.6488 | 0.6660 | 0.7931 | 0.7875 | `a06-last2-256/run-36d4a9122f30917f56ac/final/` |
| `landuse-v1\|a06-last2-256\|seed-44` | 44 | 0.8574 | 0.6495 | 0.6580 | 0.6537 | 0.7820 | 0.7833 | `a06-last2-256/run-04df35034eb04acf79c7/final/` |
| `landuse-v1\|a03-head-lr1e-4\|seed-43` | 43 | 0.8428 | 0.6430 | 0.4617 | 0.5374 | 0.7214 | 0.6992 | `a03-head-lr1e-4/run-6eb48e0182ace81c2796/final/` |
| `landuse-v1\|a03-head-lr1e-4\|seed-44` | 44 | 0.8291 | 0.5975 | 0.5040 | 0.5468 | 0.7207 | 0.7083 | `a03-head-lr1e-4/run-6ccb920c18a024fc409a/final/` |

## Interpretation

The replicated families have the following mean positive-class F1 (sample
standard deviation) across seeds `42`, `43`, and `44`:

| Family | Mean F1 | Sample SD | Mean macro-F1 |
|---|---:|---:|---:|
| `a00-baseline-head-256-lr3e-4` | 0.5394 | 0.0091 | 0.7105 |
| `a03-head-lr1e-4` | 0.5377 | 0.0090 | 0.7122 |
| `a06-last2-256` | **0.6279** | 0.0557 | **0.7622** |

`a06-last2-256` is therefore the strongest completed finalist family by the
study's primary metric. The result is a candidate for the next evaluation
stage, not a production-quality claim: the study has no held-out test set,
confidence intervals, or cross-dataset evaluation.

## Reproduce or inspect

The command is plan-only unless `--execute` is present. A completed study is
idempotent; rerunning it reads the existing state and submits no new runs:

```bash
uv run grid5000-landuse ablations
uv run grid5000-landuse ablations --execute
```

The controller stores durable state only beneath the approved external root:

```text
/Volumes/Seagate M3/projects/osm-polygon-sentence-classifier/grid5000/ablation-studies/landuse-v1/
```

Each allocation uses the existing policy checks, all-site bounded search, one
short job at a time, architecture guard, identity-bound checkpoints, static
Trackio synchronization, and exact marked-root cleanup. Grid'5000 resources
were used for the computation; publications based on this study should include
the official Grid'5000 acknowledgment.

## Provenance

- Dataset: [NoeFlandre/osm-polygon-wikidata-sentence-relevance](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-sentence-relevance)
- Dataset revision: `07e421a3020127ced2c19304645a6f63e6735966`
- Base model: [`jhu-clsp/mmBERT-small`](https://huggingface.co/jhu-clsp/mmBERT-small)
- Model revision: `abc32620dd4f6ab06f5fbe905dc25f310618e09f`
- Source commit: `496de7e5fec1b08d92e9bf295b78340224e134e0`
- Previous source commit retained for historical provenance: `5476bac6864486a715442454419bc97274e7c88d`
- Study specification SHA-256: `2418c39a44c09c474ffa978a9f120a83ad7756b6f1d444d0dafb41dd52908e92`
