# Streaming Landuse Dataset Loader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a small, testable streaming loader for the pinned Afghanistan landuse dataset that filters training labels and assigns all sentences from one polygon to the same deterministic split.

**Architecture:** Keep row transformation independent from the Hugging Face client. A pure iterator validates the existing dataset contract, drops `uncertain` rows, and yields immutable `TrainingExample` values. A lazy Hugging Face boundary supplies streaming rows with the pinned repository revision and an application-owned cache path beneath the approved Seagate root.

**Tech Stack:** Python 3.12+, `datasets` in the existing `training` extra, pytest, uv, Ruff, and ty.

---

### Task 1: Define loader behavior with failing tests

**Files:**

- Create: `tests/unit/test_dataset_loader.py`

- [ ] **Step 1: Write failing tests for deterministic polygon grouping**

Add tests that require `split_for_polygon` to return only `"train"` or `"validation"`, return the same result for repeated calls and the same polygon, reject validation fractions outside `[0, 1]`, and reject blank polygon IDs.

- [ ] **Step 2: Write failing tests for filtered training examples**

Use small in-memory rows with the exact contract columns. Require `iter_training_examples` to skip `uncertain`, retain `no` and `yes`, expose normalized text and identifiers, and assign every row sharing a polygon to the same split.

- [ ] **Step 3: Write failing tests for schema and identifier validation**

Require the iterator to reject an extra/missing first-row column through the existing `DatasetContract`, and reject empty or non-string `sentence_id`/`polygon_id` values with `DatasetLoaderError`.

- [ ] **Step 4: Write a failing test for the lazy Hugging Face boundary**

Inject a fake `load_dataset` callable and assert the loader passes the dataset ID, config, split, pinned repository revision, `streaming=True`, and a cache directory equal to `ProjectConfig().data_root / "cache/huggingface/datasets"`. The fake must be returned without materializing it into a list.

- [ ] **Step 5: Run the focused tests and confirm the expected RED failure**

Run:

```bash
./.venv/bin/pytest tests/unit/test_dataset_loader.py -q
```

Expected: collection fails because `osm_polygon_sentence_classifier.dataset_loader` does not exist yet.

### Task 2: Implement the minimal streaming loader

**Files:**

- Create: `src/osm_polygon_sentence_classifier/dataset_loader.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Test: `tests/unit/test_dataset_loader.py`

- [ ] **Step 1: Add the optional runtime dependency**

Add `datasets>=4,<5` to the existing `training` optional dependency list, then run `uv lock` so the lock file records the exact resolution.

- [ ] **Step 2: Implement stable split assignment**

Hash `f"{seed}:{polygon_id}"` with SHA-256, convert the first eight digest bytes to an unsigned integer, divide by `2**64`, and classify values below `validation_fraction` as `"validation"`; classify the rest as `"train"`.

- [ ] **Step 3: Implement the pure row iterator**

Validate the first row’s exact ordered columns and every row with `LANDUSE_DATASET_CONTRACT`. Skip labels outside `training_label_values`, require non-empty string identifiers, and yield frozen `TrainingExample` instances containing `sentence_id`, `polygon_id`, normalized text, the `no`/`yes` label, and the deterministic split.

- [ ] **Step 4: Implement the lazy Hugging Face boundary**

Import `datasets.load_dataset` only when called. Pass the contract’s dataset ID, config, split, repository revision, `streaming=True`, and `ManagedPaths(config).child("cache/huggingface/datasets")` as `cache_dir`. Wrap only a missing dependency in `DatasetLoaderError`; leave remote/network errors visible to the caller.

- [ ] **Step 5: Run focused tests and then the complete local gate**

Run:

```bash
./.venv/bin/pytest tests/unit/test_dataset_loader.py -q
./.venv/bin/pytest -q
./.venv/bin/ruff format --check . --cache-dir /private/tmp/osm-polygon-sentence-classifier-ruff-cache
./.venv/bin/ruff check . --cache-dir /private/tmp/osm-polygon-sentence-classifier-ruff-cache
./.venv/bin/ty check src tests
git diff --check
```

Expected: all tests and static checks pass; no dataset rows or generated artifacts appear in the repository.

- [ ] **Step 6: Commit the completed loader**

```bash
git add pyproject.toml uv.lock src/osm_polygon_sentence_classifier/dataset_loader.py tests/unit/test_dataset_loader.py docs/superpowers/plans/2026-08-04-dataset-loader.md
git commit -m "feat: add streaming landuse dataset loader"
```
