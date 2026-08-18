# Worldwide V2 place relevance

This lane is separate from the completed Afghanistan landuse study. It trains a
binary classifier that predicts whether a normalized sentence is relevant to a
place.

## Frozen protocol

| Item | Value |
| --- | --- |
| Dataset | `NoeFlandre/osm-polygon-wikidata-sentence-relevance` |
| Dataset config | `v2-worldwide` |
| Dataset revision | `4d0d2b5d53630c24acfb280e9d8159bf6ed0d3fa` |
| Label | `place_relevance` (`no` / `yes`) |
| Model | `jhu-clsp/mmBERT-small` |
| Trainable parameters | Classification head only |
| Input feature | `sentence_text_normalized` only |
| Split unit | `polygon_id` |
| Split | 80% train / 10% validation / 10% test |
| Split seed | `42` |

The clean boundary validates two fresh streams, removes contradictory
sentence-content-hash groups, and keeps one representative per remaining hash.
The audit found 200,000 raw rows and 175,458 clean representatives:

| Split | Rows | Polygons |
| --- | ---: | ---: |
| Train | 141,283 | 7,658 |
| Validation | 17,619 | 967 |
| Test | 16,556 | 913 |

At the default batch size of 8, one streamed training epoch is 17,661 steps.
Validation runs at the end of each logical epoch. The held-out test split is
evaluated once after training and is not used for model selection.

The exact aggregate audit is published as
[`data-audit.json`](https://huggingface.co/NoeFlandre/osm-polygon-sentence-classifier/blob/main/studies/place-relevance-v2/data-audit.json).
It contains counts only, never sentence text.

## Run it

Use the pinned source commit and model revision:

```bash
uv run grid5000-place-relevance-v2 run \
  --source-commit "$(git rev-parse HEAD)" \
  --model-revision abc32620dd4f6ab06f5fbe905dc25f310618e09f \
  --publish \
  --sync-trackio \
  --execute
```

The operator probes every configured Grid'5000 site, submits only one
policy-checked short allocation at a time, and resumes from the newest
verified checkpoint after an interrupted allocation. Worldwide V2 allows up
to 40 bounded continuation jobs by default. `--max-continuations` can raise
that bound deliberately if the scheduler or walltime requires it.

The public Trackio project is `place-relevance-v2` with the stable run name
`place-relevance-v2|baseline|seed-42`. This keeps continuation jobs in one
logical run while the model repository stores each checkpoint under:

```text
studies/place-relevance-v2/baseline/run-<run-id>/checkpoints/step-N/
studies/place-relevance-v2/baseline/run-<run-id>/final/
```

The generated study registry contains the protocol, data audit, and final
metrics. The separate [V2 ablation study](place-relevance-v2-ablations.md)
uses a different Trackio project and public namespace; its results must not be
mixed with this baseline.
