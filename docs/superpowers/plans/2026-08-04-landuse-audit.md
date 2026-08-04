# Landuse Dataset Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Audit the pinned Afghanistan landuse dataset before training and write only a JSON audit report and deterministic polygon split manifest beneath the approved Seagate project-data root.

**Architecture:** Keep the audit reducer pure and streaming: it validates rows through `LANDUSE_DATASET_CONTRACT`, counts label/polygon/language/source/text/duplicate risks, and returns immutable report values. A small CLI composes the existing lazy Hugging Face loader with the reducer; the loader may populate its approved Seagate cache, while the explicit writer writes two derived JSON artifacts under `audit/landuse` and never writes raw rows or starts training.

**Tech Stack:** Python 3.12+, existing `datasets` optional extra, standard-library counters/JSON, pytest, Ruff, ty, and uv.

---

### Task 1: Specify audit and artifact behavior with failing tests

**Files:**

- Create: `tests/unit/test_dataset_audit.py`
- Create: `tests/unit/test_audit_cli.py`

- [x] **Step 1: Write failing reducer tests**

Build exact-contract in-memory rows containing `no`, `yes`, and `uncertain` labels. Require the audit to count all labels, count only `no`/`yes` as trainable, count polygons, preserve deterministic split assignments, count languages and sources, calculate training text-length statistics, and report polygons containing both training labels as conflicts.

- [x] **Step 2: Write failing duplicate and split-readiness tests**

Use repeated `sentence_content_hash` values in one polygon and across two polygons. Require the report to distinguish duplicate hash groups from cross-polygon duplicate groups and to add a review reason when a split lacks one of the training labels.

- [x] **Step 3: Write failing validation and streaming tests**

Require the reducer to validate the first row’s exact ordered schema and every row through the existing contract, reject invalid identifiers with a row-numbered `DatasetAuditError`, and consume rows lazily without materializing them.

- [x] **Step 4: Write failing artifact and CLI tests**

Require JSON serialization to contain dataset provenance, audit counts, readiness reasons, and sorted polygon assignments. Patch the filesystem boundary in the writer test and require output paths exactly beneath `/Volumes/Seagate M3/projects/osm-polygon-sentence-classifier/audit/landuse`. Patch the CLI’s loader/reducer/writer in a test and require it to print both artifact paths and exit with status 2 when review reasons remain.

- [x] **Step 5: Run the focused tests and confirm RED**

Run:

```bash
PYTHONPATH=src .venv/bin/pytest tests/unit/test_dataset_audit.py tests/unit/test_audit_cli.py -q
```

Expected: collection fails because the audit modules do not exist yet.

### Task 2: Implement the streaming audit and CLI

**Files:**

- Create: `src/osm_polygon_sentence_classifier/dataset_audit.py`
- Create: `src/osm_polygon_sentence_classifier/audit_cli.py`
- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `docs/guides/data-policy.md`
- Modify: `docs/architecture/overview.md`
- Test: `tests/unit/test_dataset_audit.py`
- Test: `tests/unit/test_audit_cli.py`

- [x] **Step 1: Implement immutable audit result types**

Use frozen/slotted dataclasses with tuple-based sorted counters. Include the pinned dataset identity/provenance, split parameters, total and trainable row counts, label/polygon/split counts, language/source counts, duplicate metrics, conflicting-polygon count, text-length statistics, review reasons, and a sorted `(polygon_id, split)` manifest.

- [x] **Step 2: Implement the streaming reducer**

Consume rows through `itertools.chain` after validating the first row’s exact columns. Validate every row, require non-empty string `sentence_id` and `polygon_id`, skip `uncertain` only from training counters, assign deterministic splits with `split_for_polygon`, and retain only counters/sets needed for the report. Track duplicate sentence hashes only for trainable rows and identify hashes spanning multiple polygons.

- [x] **Step 3: Implement JSON artifact writing under the fixed root**

Resolve `audit/landuse` with `ManagedPaths(config).child(...)`, create that directory only when the explicit writer is called, and write `audit_report.json` plus `split_manifest.json` with stable sorted JSON and a trailing newline. The loader's separate Hugging Face cache is allowed only under the approved Seagate root. Never serialize raw sentences or full input rows.

- [x] **Step 4: Implement the CLI entry point**

Add `audit-landuse-dataset = "osm_polygon_sentence_classifier.audit_cli:main"` to `[project.scripts]`. The CLI must call `load_streaming_rows()`, reduce the stream, write both artifacts, print their paths and readiness, and exit with status 2 after writing when review reasons are present.

- [x] **Step 5: Update public boundary documentation**

Document that the audit command is the only current data-consuming command, that its cache/report/manifest are confined to the Seagate root, and that it does not train, upload, or submit Grid5000 work. Remove stale statements claiming the repository has no dataset transformation at all.

- [x] **Step 6: Run the focused and complete verification gates**

Run:

```bash
PYTHONPATH=src .venv/bin/pytest tests/unit/test_dataset_audit.py tests/unit/test_audit_cli.py -q
PYTHONPATH=src .venv/bin/pytest -q
.venv/bin/ruff format --check . --cache-dir /private/tmp/osm-polygon-sentence-classifier-ruff-cache
.venv/bin/ruff check . --cache-dir /private/tmp/osm-polygon-sentence-classifier-ruff-cache
.venv/bin/ty check src tests
UV_CACHE_DIR=/private/tmp/osm-polygon-sentence-classifier-uv-cache uv lock --check
git diff --check
```

Expected: all tests and checks pass without accessing the real dataset during unit tests.

- [ ] **Step 7: Commit the completed audit**

```bash
git add pyproject.toml README.md docs/guides/data-policy.md docs/architecture/overview.md src/osm_polygon_sentence_classifier/dataset_audit.py src/osm_polygon_sentence_classifier/audit_cli.py tests/unit/test_dataset_audit.py tests/unit/test_audit_cli.py docs/superpowers/plans/2026-08-04-landuse-audit.md
git commit -m "feat: add landuse dataset audit"
```
