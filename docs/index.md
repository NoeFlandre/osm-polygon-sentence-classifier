# OSM Polygon Sentence Classifier

This project trains sentence classifiers for OSM polygon descriptions. It
keeps data, compute, evaluation, and publication gates explicit and
reproducible.

## Current status

The initial `landuse-v1` ablation study is complete: all seven screening runs
and six planned replications finished successfully. The completed artifacts
are catalogued in the public model repository's
[`studies/landuse-v1/README.md`](https://huggingface.co/NoeFlandre/osm-polygon-sentence-classifier/blob/main/studies/landuse-v1/README.md).
The `a06-last2-256` family has the highest mean validation positive-class F1
among the replicated finalists. This is validation evidence only; the study
does not contain a held-out test set.

The initial training model is
[`jhu-clsp/mmBERT-small`](https://huggingface.co/jhu-clsp/mmBERT-small): a
140M-parameter encoder trained across 1,800+ languages. The training boundary
freezes that encoder and learns only a binary landuse classification head,
following the efficient frozen-encoder approach used by
[FineWeb-Edu](https://github.com/huggingface/cosmopedia/tree/main/classification).

Start with the [getting-started guide](guides/getting-started.md), use the
[completed ablation registry](guides/ablations.md), then read the
[data policy](guides/data-policy.md), the
[experiment organization guide](guides/experiment-organization.md), the
[Grid'5000 boundary](reference/grid5000-boundary.md), and the
[operator reference](reference/grid5000-operator.md) before proposing a new
training workflow.

The worldwide V2 baseline is complete and its held-out test result is recorded
under `studies/place-relevance-v2/`. A separate 13-run V2 ablation protocol is
implemented but must be run through its explicit Grid'5000 execution gate
before results are claimed. Read the [V2 ablation guide](guides/place-relevance-v2-ablations.md)
and [operator reference](reference/grid5000-operator.md) first.

## Citation

The repository's [CITATION.cff](https://github.com/NoeFlandre/osm-polygon-sentence-classifier/blob/main/CITATION.cff)
contains the machine-readable citation metadata. The same citation is shown
in the repository [README](https://github.com/NoeFlandre/osm-polygon-sentence-classifier#citation).
