# OSM Polygon Sentence Classifier

This project will train a sentence classifier for OSM polygon descriptions,
starting with the landuse task. The classifier is intended to support a
landuse-first progression while keeping data, compute, and publication gates
explicit.

The current milestone is a safe local foundation, an explicit review-only
landuse dataset audit, and a typed training boundary. It establishes small
package boundaries, managed data paths, Trackio settings, tests, quality
tooling, and public documentation. The audit may stream the pinned dataset and
write only its approved cache, report, and split manifest. The training entry
point consumes the clean stream and writes local model outputs beneath the
approved root, but does not upload or submit remote jobs. Audit readiness is
sentence-level: mixed-label polygons are diagnostic, while contradictory or
cross-split duplicate sentence hashes remain blockers.

Start with the [getting-started guide](guides/getting-started.md), review the
[data policy](guides/data-policy.md), and read the
[Grid5000 boundary](reference/grid5000-boundary.md) before proposing a later
training workflow.
