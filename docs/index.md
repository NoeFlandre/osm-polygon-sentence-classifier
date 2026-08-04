# OSM Polygon Sentence Classifier

This project will train a sentence classifier for OSM polygon descriptions,
starting with the landuse task. The classifier is intended to support a
landuse-first progression while keeping data, compute, and publication gates
explicit.

The current milestone is a safe local foundation. It establishes small package
boundaries, managed data paths, Trackio settings, tests, quality tooling, and
public documentation. It does not process the training dataset or submit
remote jobs.

Start with the [getting-started guide](guides/getting-started.md), review the
[data policy](guides/data-policy.md), and read the
[Grid5000 boundary](reference/grid5000-boundary.md) before proposing a later
training workflow.
