# OSM Polygon Sentence Classifier

This project will train a sentence classifier for OSM polygon descriptions,
starting with the landuse task. The classifier is intended to support a
landuse-first progression while keeping data, compute, and publication gates
explicit.

The current milestone is a safe local foundation, an explicit review-only
landuse dataset audit, a typed training boundary, and an autonomous Grid'5000
operator. It establishes small package boundaries, managed data paths, Trackio
settings, tests, quality tooling, and public documentation. The audit may
stream the pinned dataset and write only its approved cache, report, and split
manifest. The training entry point consumes the clean stream and writes local
model outputs beneath the approved root. An explicitly enabled autonomous
Grid'5000 run discovers a site, submits and monitors one short job, publishes
the completed model, synchronizes finished metrics, and cleans only its marked
successful remote root. Audit readiness is sentence-level: mixed-label
polygons are diagnostic, while contradictory or cross-split duplicate sentence
hashes remain blockers.

The initial training model is
[`jhu-clsp/mmBERT-small`](https://huggingface.co/jhu-clsp/mmBERT-small): a
140M-parameter encoder trained across 1,800+ languages. The training boundary
freezes that encoder and learns only a binary landuse classification head,
following the efficient frozen-encoder approach used by
[FineWeb-Edu](https://github.com/huggingface/cosmopedia/tree/main/classification).

Start with the [getting-started guide](guides/getting-started.md), review the
[data policy](guides/data-policy.md), and read the
[Grid'5000 boundary](reference/grid5000-boundary.md) and the
[operator reference](reference/grid5000-operator.md) before proposing a
training workflow. The reproducible comparison workflow is documented in the
[landuse ablation study guide](guides/ablations.md).
