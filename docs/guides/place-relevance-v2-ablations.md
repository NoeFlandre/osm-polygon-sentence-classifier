# Worldwide V2 ablation study

The worldwide V2 ablations are a separate study from the completed V2 baseline.
They use the same pinned dataset, model revision, polygon-level split, and one
logical-epoch budget. Their public artifacts live under
`studies/place-relevance-v2-ablations/`, and their Trackio runs use the separate
`place-relevance-v2-ablations` project.

## Fixed protocol

- Seven screening runs use seed `42`.
- The baseline and the two highest validation positive-class F1 variants are
  replicated with seeds `43` and `44`.
- This produces 13 runs in total.
- Selection uses validation `eval_f1`, with validation `eval_macro_f1` as the
  tie-break. The held-out test set is evaluated once after each run and is not
  used for selection.
- Every run uses the same deterministic polygon-level 80/10/10 split and the
  same 17,661-step one-epoch budget.

The screening matrix changes one factor at a time:

| ID | Controlled change |
|---|---|
| `a00-baseline-head-256-lr3e-4` | Frozen encoder, 256 tokens, learning rate `3e-4` |
| `a01-head-128` | Maximum length `128` |
| `a02-head-512` | Maximum length `512` |
| `a03-head-lr1e-4` | Learning rate `1e-4` |
| `a04-head-lr1e-3` | Learning rate `1e-3` |
| `a05-balanced-head` | Class-balanced loss |
| `a06-last2-256` | Unfreeze the last two encoder layers |

## Run the study

The command is idempotent. It runs one ablation at a time, uses short
policy-checked Grid'5000 allocations, publishes complete checkpoints to the
model repository, synchronizes static Trackio snapshots, and resumes only from
verified checkpoints:

```bash
uv run grid5000-place-relevance-v2 ablations \
  --source-commit "$(git rev-parse HEAD)" \
  --model-revision abc32620dd4f6ab06f5fbe905dc25f310618e09f \
  --execute
```

The controller may move a run to another compatible site after a stale queue,
but it never starts a replacement from scratch after a checkpoint exists. The
default continuation bound is 40 per run. Remote per-run data is removed only
after publication and completion evidence have been verified.

## Public outputs

- [Study report](https://huggingface.co/NoeFlandre/osm-polygon-sentence-classifier/blob/main/studies/place-relevance-v2-ablations/README.md)
- [Protocol](https://huggingface.co/NoeFlandre/osm-polygon-sentence-classifier/blob/main/studies/place-relevance-v2-ablations/study.json)
- [Results](https://huggingface.co/NoeFlandre/osm-polygon-sentence-classifier/blob/main/studies/place-relevance-v2-ablations/results.json)
- [Trackio dashboard](https://huggingface.co/spaces/NoeFlandre/osm-polygon-sentence-classifier-trackio)

Run names have the stable form
`place-relevance-v2-ablations|<ablation-id>|seed-<seed>`. Scheduler job IDs
are operational details and do not identify an experiment.
