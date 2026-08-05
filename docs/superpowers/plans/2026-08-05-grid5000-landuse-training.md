# Grid'5000 landuse training operator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a plan-by-default, explicitly guarded Grid'5000 operator for one reproducible landuse training allocation, with live-policy/quota gates, durable no-duplicate submission state, and strict compute-node validation.

**Architecture:** Keep pure run identity, bounded allocation, command rendering, injected command execution, and local state in `grid5000.py`. Keep remote environment validation and training invocation in `grid5000_worker.py`. Extend the existing training/config contracts only where the operator needs a pinned model revision and a remote managed root. Use stdlib-only operator code and preserve the existing clean streaming dataset boundary.

**Tech Stack:** Python 3.12, dataclasses, `pathlib`, `hashlib`, `json`, `subprocess`, `argparse`, pytest, Ruff, ty, MkDocs Material, existing Hugging Face/Trackio training extras.

---

## Task 1: Add failing contract tests before production changes

**Files:**

- Create `tests/unit/test_grid5000.py`.
- Extend `tests/unit/test_config.py` for the narrowly scoped remote-root constructor.
- Extend `tests/unit/test_training.py` for `model_revision` propagation.

**Tests to write first:**

1. `Grid5000RunIdentity` rejects non-string and non-lowercase-40-character
   source, dataset, or model revisions, produces stable canonical JSON, and
   changes its run ID when a training setting changes.
2. `Grid5000Allocation` rejects zero/multiple GPUs, non-`exotic` resource type,
   non-`default` queue, non-`night` policy, non-positive walltime, and walltime
   over twelve hours. It renders one exact `oarsub` request with `gpu=1`.
3. A plan-only submit does not call an injected runner and does not create a
   local state directory.
4. An execute submit calls the two policy commands and quota check before the
   single OAR command, writes `submitting` intent before OAR, records the
   returned numeric job ID, and preserves the exact run identity.
5. Existing `submitted` state and ambiguous `submitting` state both refuse a
   second submission without calling any runner.
6. State writes use a private run directory and restrictive file/directory
   modes; malformed or symlinked state is rejected.
7. The worker rejects non-Linux, missing/non-numeric `OAR_JOB_ID`, wrong commit,
   dirty checkout, missing CUDA, or a visible-GPU count other than one before
   calling the injected training function.
8. `TrainingConfig(model_revision=...)` passes the revision to tokenizer and
   model loading while the default keeps the existing calls unchanged.

**RED command:**

```bash
uv run pytest -q tests/unit/test_grid5000.py tests/unit/test_config.py tests/unit/test_training.py
```

The new tests must fail because the new module, constructor, worker checks, and
model revision do not exist yet. Do not weaken assertions to obtain a green
run.

## Task 2: Implement immutable identity, allocation, and fixed command planning

**Files:**

- Create `src/osm_polygon_sentence_classifier/grid5000.py`.
- Update `src/osm_polygon_sentence_classifier/config.py` only for
  `ProjectConfig.for_remote_root()`.

Implement the exact dataclasses and errors in the design spec. Canonicalize
identity through sorted JSON with stable separators; hash it with SHA-256 and
use a deterministic 20-character hexadecimal run ID. Require all revisions to
match `[0-9a-f]{40}`. Include the complete training configuration in identity,
not arbitrary object representations.

Use fixed scheduler values (`default`, `exotic`, `night`, one GPU) and a
12-hour maximum. Render an argument tuple, not a shell string. The worker
command must be a fixed project command with only validated scalar arguments;
use `shlex.quote` only for human-readable display if a shell string is printed.

Run the RED command, then:

```bash
uv run pytest -q tests/unit/test_grid5000.py tests/unit/test_config.py
```

## Task 3: Implement fail-closed policy, quota, and durable state gates

**Files:**

- Continue `src/osm_polygon_sentence_classifier/grid5000.py`.
- Extend `tests/unit/test_grid5000.py` as needed for each failure path.

Implement injected command execution with fixed argv and a bounded timeout.
The execution sequence is strictly:

1. reject any existing or ambiguous state;
2. run `usagepolicycheck -l --sites <validated-site>`;
3. run `usagepolicycheck -t`;
4. read and parse the remote home soft quota and require the documented
   staging headroom;
5. atomically persist `submitting` intent;
6. invoke one `oarsub` request;
7. require one numeric job ID and atomically persist `submitted` state.

Fail closed on nonzero policy/quota commands, malformed quota output, missing
soft headroom, state permission/symlink problems, malformed scheduler output,
or OAR failure. Do not retry OAR and do not delete anything. Never include
tokens, passwords, or arbitrary user shell text in commands or persisted state.

Run targeted tests and the full unit suite after the implementation is green.

## Task 4: Add strict compute-node worker boundary and pinned model revision

**Files:**

- Create `src/osm_polygon_sentence_classifier/grid5000_worker.py`.
- Update `src/osm_polygon_sentence_classifier/training.py`.
- Update `tests/unit/test_training.py` and `tests/unit/test_grid5000.py`.

Add `model_revision: str | None` to `TrainingConfig`; when non-`None`, validate
the same lowercase-40-character revision format and pass it to both
`from_pretrained` calls. Omit the keyword when it is `None` to preserve the
existing default behavior and test call shape.

Add an injected worker preflight that validates Linux, numeric OAR job ID,
exact source commit, strict clean checkout, CUDA availability, and exactly one
visible GPU. The worker calls the existing `train_landuse_classifier` only
after all checks pass, using `ProjectConfig.for_remote_root()` and the plan's
managed remote output paths. It must not implement CPU fallback, network
upload, scheduler submission, or automatic retries.

Run:

```bash
uv run pytest -q tests/unit/test_training.py tests/unit/test_grid5000.py
```

## Task 5: Add the guarded CLI and documentation

**Files:**

- Create `src/osm_polygon_sentence_classifier/grid5000_cli.py`.
- Update `pyproject.toml` with `grid5000-landuse` only; add no dependency.
- Update `README.md`.
- Update `docs/guides/getting-started.md`.
- Update `docs/architecture/overview.md`.
- Replace/update `docs/reference/grid5000-boundary.md`.
- Add `docs/reference/grid5000-operator.md` and link it in `mkdocs.yml`.

The CLI must default to a side-effect-free plan. Without `--execute`, `submit`
prints the plan and exits successfully; the flag is the only way to cross the
execution gate. Document that exact distinction.
The CLI must never infer `--execute` from environment variables, CI, or a
non-interactive terminal. Document policy checks, soft-quota fail-closed
behavior, one-allocation/no-racing behavior, durable state, and the fact that
this implementation has not submitted a job or published a model.

Add CLI tests proving plan mode is the default and `--execute` is the only
execution gate. Avoid a Typer dependency for this small operator.

## Task 6: Full verification and review

Run all of the following with caches/site output outside the repository when
the checkout's ignored cache/output directories are unavailable:

```bash
RUFF_CACHE_DIR=/tmp/osm-polygon-sentence-classifier-ruff-cache uv run ruff format --check .
RUFF_CACHE_DIR=/tmp/osm-polygon-sentence-classifier-ruff-cache uv run ruff check .
uv run ty check
uv run pytest -q
uv run mkdocs build --strict --site-dir /tmp/osm-polygon-sentence-classifier-site
uv run pre-commit run --all-files
uv lock --check
git diff --check
```

Review the final diff for accidental network calls, shell injection, data
outside `/Volumes/Seagate M3/projects/osm-polygon-sentence-classifier`, live
Grid'5000 invocation, changed public training semantics, unnecessary
abstractions, or touched `.DS_Store` files. Confirm no real `--execute` command
was run. Commit with a Conventional Commit message and push only after all
checks pass; verify `HEAD`, `origin/main`, and the remote `main` ref match.
