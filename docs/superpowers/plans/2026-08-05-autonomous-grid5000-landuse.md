# Autonomous Grid'5000 Landuse Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the one-site, submit-only Grid'5000 boundary with one guarded command that probes all configured sites, prepares the selected site, submits a policy-compliant short training job, monitors and replaces queued allocations safely, verifies Hugging Face/Trackio completion, and cleans only pipeline-owned remote data.

**Architecture:** Keep the existing immutable run identity and training worker, but split Grid'5000 concerns into factual site inventory, remote preparation, OAR lifecycle, durable run state, and orchestration modules. Site selection will use `oarnodes -J` facts and hard quota/compatibility constraints; it will derive `default`/`production` and `standard`/`exotic` from the selected resource instead of assuming Nancy's resource class. A single autonomous run may have one fallback and one short replacement trial, but it never submits speculative jobs to multiple sites and never retries an ambiguous submission without reconciliation.

**Tech Stack:** Python 3.12, stdlib `argparse`/`subprocess`/`json`/`concurrent.futures`, `uv`, Ruff, Ty, pytest, MkDocs, Hugging Face Hub, Trackio, OAR, and the existing fixed Grid'5000 SSH boundary.

---

## Evidence and non-negotiable contracts

- The read-only sentence-relevance operator probes all configured sites, treats `oarnodes` idle state as factual availability, never uses queue depth as an ETA, stages the immutable run before submitting, and uses one bounded replacement trial against a queued fallback.
- The target operator currently hard-codes `default + exotic`. The Grenoble probe showed 44 idle compatible GPUs, all `production=YES, exotic=NO`; the real request returned OAR exit 8 with “There are not enough resources”. Queue and resource type therefore must be derived from the inventory.
- The target worker currently invokes `uv run --locked` without `--extra training`, although all model/data/Trackio dependencies are optional. The autonomous worker command must install/use the locked `training` extra in allocation-local scratch.
- Day jobs are policy-bounded to at most one hour; the default short allocation is 30 minutes. Night fallback jobs also use a 30-minute walltime because this classifier has a bounded 1,000-step frozen-head workload.
- No remote credential, token, or raw scheduler output containing credentials may enter local durable state, logs, command JSON, or progress messages.
- A submission intent is written before OAR. If OAR returns no job ID, the operator must reconcile the exact run marker against the user’s current OAR jobs and stop if the result is ambiguous; it must never blindly submit a duplicate.
- Cleanup is confined to paths created/marked by this pipeline. It runs only after a verified completed publication unless the user explicitly invokes a separate cleanup command.

## File map before implementation

- Modify `src/osm_polygon_sentence_classifier/grid5000.py` to generalize allocation queue/resource validation, render resource-property constraints, and expose the reusable command/state primitives without changing the existing plan/submit compatibility surface.
- Create `src/osm_polygon_sentence_classifier/grid5000_sites.py` for all-site constants, `oarnodes -J` parsing, quota/compatibility facts, and deterministic candidate selection.
- Create `src/osm_polygon_sentence_classifier/grid5000_remote.py` for bounded SSH/scp command construction, exact-checkout preparation, HF-token staging, marker-based remote paths, and safe pipeline-owned cleanup.
- Create `src/osm_polygon_sentence_classifier/grid5000_oar.py` for normalized OAR job status parsing, submission/reconciliation/cancellation, and bounded remote log reads.
- Create `src/osm_polygon_sentence_classifier/grid5000_state.py` for atomic mode-700/mode-600 autonomous run state and append-only attempt facts. It must accept and classify the current legacy `submitting` state rather than silently overwrite it.
- Create `src/osm_polygon_sentence_classifier/grid5000_autonomous.py` for the one-command controller, all-site preparation, fallback/trial lifecycle, completion-manifest verification, and cleanup decision.
- Modify `src/osm_polygon_sentence_classifier/grid5000_cli.py` to add `run`, `status`, and `resume` while retaining side-effect-free `plan` and explicit `submit` compatibility.
- Modify `src/osm_polygon_sentence_classifier/grid5000_worker.py` to use `uv run --locked --extra training` through the scheduler command and write a credential-free completion manifest after publication/Trackio synchronization.
- Modify `src/osm_polygon_sentence_classifier/publication.py` only if a small read-only Hub verification helper is needed; do not broaden the existing add-only publication contract.
- Modify `tests/unit/test_grid5000.py`, `tests/unit/test_grid5000_cli.py`, and `tests/unit/test_training.py` only for changed compatibility contracts.
- Create `tests/unit/test_grid5000_sites.py`, `tests/unit/test_grid5000_remote.py`, `tests/unit/test_grid5000_oar.py`, `tests/unit/test_grid5000_state.py`, and `tests/unit/test_grid5000_autonomous.py` with injected runners and no live scheduler calls.
- Modify `docs/reference/grid5000-boundary.md`, `docs/reference/grid5000-operator.md`, `docs/guides/getting-started.md`, and `docs/architecture/overview.md` to document the autonomous command, all-site policy, short-job fallback/trial behavior, state/recovery boundary, remote cleanup ownership, and the no-credential rule.

### Task 1: Generalize the allocation contract and fix the worker dependency boundary

**Files:**
- Modify: `src/osm_polygon_sentence_classifier/grid5000.py`
- Modify: `src/osm_polygon_sentence_classifier/grid5000_cli.py`
- Modify: `src/osm_polygon_sentence_classifier/grid5000_worker.py`
- Test: `tests/unit/test_grid5000.py`
- Test: `tests/unit/test_grid5000_cli.py`

- [ ] **Step 1: Write failing tests for dynamic queue/resource types.** Add tests that construct `Grid5000Allocation(site="grenoble", queue="production", resource_type="standard", policy_type="day", walltime_seconds=1800)` and require the exact request to contain `-q production`, no `-t exotic`, `-t day`, and `gpu=1,walltime=00:30:00`. Add rejection tests for queues other than `default|production` and resource types other than `standard|exotic`.

- [ ] **Step 2: Run the focused tests and verify the intended red failure.**

  ```bash
  env UV_CACHE_DIR=/tmp/osm-polygon-sentence-classifier-t1 uv run pytest -q tests/unit/test_grid5000.py tests/unit/test_grid5000_cli.py
  ```

  Expected: the new dynamic-resource tests fail because the current allocation rejects `production` and `standard` and always renders `-t exotic`.

- [ ] **Step 3: Implement the minimal allocation change.** Replace the hard-coded checks with `queue in {"default", "production"}` and `resource_type in {"standard", "exotic"}`. Render `-t exotic` only for `resource_type == "exotic"`; always render the selected policy type. Add a generated OAR property argument only when a selected site supplies one, and validate it against the internally generated `gpu_mem`/`production` form rather than accepting arbitrary shell text.

- [ ] **Step 4: Add the worker-extra regression test before changing the command.** Extend `test_worker_command_uses_allocation_local_locked_uv_environment` to require `"run --locked --extra training python -m"` in the worker command. Run that test and observe the expected failure against the current command.

- [ ] **Step 5: Implement the worker command fix.** Change the fixed command to `uv run --locked --extra training python -m ...`. Keep `UV_PROJECT_ENVIRONMENT` and `UV_CACHE_DIR` under the OAR allocation’s `/tmp`, keep `umask 077`, and leave all durable model/data/Trackio paths under the managed remote data root.

- [ ] **Step 6: Add the short-job CLI contract.** Make `run`’s default walltime 1,800 seconds and preserve `plan`/`submit`’s explicit day/night behavior. Keep `night` as the compatibility default for the existing commands. The autonomous command must calculate `day` only when the full 30-minute walltime fits the current Europe/Paris policy window; otherwise it must choose `night`.

- [ ] **Step 7: Run focused green validation.**

  ```bash
  env UV_CACHE_DIR=/tmp/osm-polygon-sentence-classifier-t1-green uv run pytest -q tests/unit/test_grid5000.py tests/unit/test_grid5000_cli.py
  env UV_CACHE_DIR=/tmp/osm-polygon-sentence-classifier-t1-green uv run ruff check src/osm_polygon_sentence_classifier/grid5000.py src/osm_polygon_sentence_classifier/grid5000_cli.py src/osm_polygon_sentence_classifier/grid5000_worker.py tests/unit/test_grid5000.py tests/unit/test_grid5000_cli.py
  ```

  Expected: all focused tests pass and the generated command is compatible with both Nancy-style exotic/default resources and Grenoble-style standard/production resources.

### Task 2: Add factual all-site inventory and deterministic resource selection

**Files:**
- Create: `src/osm_polygon_sentence_classifier/grid5000_sites.py`
- Create: `tests/unit/test_grid5000_sites.py`
- Modify: `src/osm_polygon_sentence_classifier/grid5000.py`

- [ ] **Step 1: Write parser tests first.** Use a fixture containing alive/dead GPU nodes with `jobs`, `gpu_mem`, `gpu_compute_capability_major`, `production`, and `exotic` fields. Require:
  - dead nodes and nodes with assigned jobs are not idle;
  - a node with at least 8,000 MiB and capability `(7, 0)` is compatible;
  - a Grenoble-style compatible group selects `queue="production"`, `resource_type="standard"`;
  - a Nancy-style compatible non-production exotic group selects `queue="default"`, `resource_type="exotic"`;
  - malformed JSON, non-mapping records, negative values, and missing fields fail closed with a stable `SiteProbeError`.

- [ ] **Step 2: Run the parser tests to verify red.**

  ```bash
  env UV_CACHE_DIR=/tmp/osm-polygon-sentence-classifier-t2 uv run pytest -q tests/unit/test_grid5000_sites.py
  ```

  Expected: collection fails because `grid5000_sites.py` does not exist.

- [ ] **Step 3: Implement the small pure data model.** Define:

  ```python
  DEFAULT_SITES = (
      "bordeaux", "grenoble", "lille", "louvain", "luxembourg", "lyon",
      "nancy", "nantes", "rennes", "sophia", "strasbourg", "toulouse",
  )
  MINIMUM_GPU_MEMORY_MB = 8_000
  MINIMUM_CUDA_CAPABILITY = (7, 0)

  @dataclass(frozen=True, slots=True)
  class GpuResource:
      state: str
      jobs_assigned: int
      gpu_memory_mb: int
      cuda_capability: tuple[int, int]
      production: bool
      exotic: bool
  ```

  Add `parse_oarnodes_payload`, `parse_oarnodes_stdout`, `SiteProbe`, and `select_site_probe`. `select_site_probe` must sort by hard compatibility, factual idle state, reuse of an already marked managed checkout, and site name only. It must never use queue depth as a forecast. The selected probe must carry the derived queue/resource pair and a generated `resource_property` such as `gpu_mem>=8000 AND production='YES'`.

- [ ] **Step 4: Add injected command-boundary tests.** Test that a site probe’s SSH command includes only `oarnodes -J`, `quota`, and the policy checks; it uses bounded `BatchMode`/connect timeouts; and a DNS/SSH failure yields an unreachable probe rather than aborting the entire all-site scan. Test all configured sites are attempted and a duplicate site is de-duplicated.

- [ ] **Step 5: Implement bounded all-site probing.** Use a bounded `ThreadPoolExecutor` with one probe per configured target, maximum workers equal to the number of targets, and deterministic result sorting. Do not submit during probing. Store only parsed facts, never raw tokens or full remote output.

- [ ] **Step 6: Run green focused validation.**

  ```bash
  env UV_CACHE_DIR=/tmp/osm-polygon-sentence-classifier-t2-green uv run pytest -q tests/unit/test_grid5000_sites.py
  env UV_CACHE_DIR=/tmp/osm-polygon-sentence-classifier-t2-green uv run ty check
  ```

### Task 3: Add secure autonomous remote preparation and cleanup

**Files:**
- Create: `src/osm_polygon_sentence_classifier/grid5000_remote.py`
- Create: `tests/unit/test_grid5000_remote.py`

- [ ] **Step 1: Write failing preparation tests.** With an injected `CommandRunner`, require that preparation:
  - clones the public repository only when the exact managed checkout is absent;
  - fetches and detaches the requested 40-character source commit;
  - rejects a dirty checkout instead of deleting or overwriting it;
  - writes a mode-600 marker under the exact project path;
  - stages the HF token through stdin/scp without putting token bytes in an argv tuple, command string, result, or state payload;
  - verifies `uv` exists but does not build a persistent training environment;
  - cleanup removes only the managed checkout, managed data root, run marker, token marker, and job-local OAR logs for the completed run.

- [ ] **Step 2: Run the tests and verify red.**

  ```bash
  env UV_CACHE_DIR=/tmp/osm-polygon-sentence-classifier-t3 uv run pytest -q tests/unit/test_grid5000_remote.py
  ```

  Expected: collection fails because the remote module does not exist.

- [ ] **Step 3: Implement safe shell generation.** Use `shlex.quote` for every generated path/value; reject symlinked managed roots; use `git clone --no-tags` for an absent checkout and `git fetch --no-tags origin` plus detached checkout for a clean existing checkout. Never run `rm -rf` on a caller-provided arbitrary path. The only cleanup path constants are:

  ```python
  REMOTE_CHECKOUT = PurePosixPath("$HOME/osm-polygon-sentence-classifier")
  REMOTE_DATA = PurePosixPath("$HOME/osm-polygon-sentence-classifier-data")
  REMOTE_MARKER = REMOTE_DATA / ".grid5000-managed.json"
  ```

  The token is written with `umask 077`, a temporary file, `chmod 600`, and atomic rename. The local implementation may use `scp` or an SSH stdin channel, but tests must assert the token is never interpolated into a remote command.

- [ ] **Step 4: Add completion-preserving preparation.** Return a `RemotePreparation` record containing site, checkout path, data root, whether checkout was reused, and whether this run created the remote token. Do not return raw command output.

- [ ] **Step 5: Run focused green validation.**

  ```bash
  env UV_CACHE_DIR=/tmp/osm-polygon-sentence-classifier-t3-green uv run pytest -q tests/unit/test_grid5000_remote.py
  ```

### Task 4: Add normalized OAR lifecycle and durable reconciliation

**Files:**
- Create: `src/osm_polygon_sentence_classifier/grid5000_oar.py`
- Create: `src/osm_polygon_sentence_classifier/grid5000_state.py`
- Create: `tests/unit/test_grid5000_oar.py`
- Create: `tests/unit/test_grid5000_state.py`

- [ ] **Step 1: Write OAR parser tests.** Cover JSON states `Waiting`, `Hold`, `Launching`, `Running`, `Finishing`, `Terminated`, and `Error`; missing job exit code 6; integer and string `scheduled_start`; and parse failures. Require `JobStatus` to expose `job_id`, normalized state, message, exit code, node, forecast, and walltime.

- [ ] **Step 2: Write state tests before implementation.** Require atomic mode-700 directory/mode-600 state files, schema validation, no unsafe fact keys (`token`, `secret`, `authorization`, `raw_response`), monotonically increasing sequence, and recovery of a `submitting` intent. Add a test that a failed submission with no matching OAR job can be reconciled to `failed`, while a matching job is adopted and an ambiguous match remains blocked.

- [ ] **Step 3: Run both test files to verify red.**

  ```bash
  env UV_CACHE_DIR=/tmp/osm-polygon-sentence-classifier-t4 uv run pytest -q tests/unit/test_grid5000_oar.py tests/unit/test_grid5000_state.py
  ```

  Expected: collection fails because the new modules do not exist.

- [ ] **Step 4: Implement normalized OAR lifecycle.** Define `JobState`, `JobStatus`, `OarClient.submit/status/cancel`, and `parse_job_id`. `OarClient.status` must treat OAR return code 6 as `missing`, never as permission to resubmit automatically. `OarClient.submit` must accept one generated `Grid5000Allocation` command only.

- [ ] **Step 5: Implement durable attempts.** Store `RunState` with `schema_version`, immutable run identity, phase, sequence, active attempt, and an append-only list of attempts `{site, job_id, role, scheduler_command, state}`. Write intent before submission; after a successful job ID, atomically record it. Reconcile a submitting intent by searching the current user job list for the exact run marker; return one adopted job, no job, or ambiguous. Preserve the existing legacy `state.json` as an importable `legacy_submission` record rather than overwriting it.

- [ ] **Step 6: Run green lifecycle validation.**

  ```bash
  env UV_CACHE_DIR=/tmp/osm-polygon-sentence-classifier-t4-green uv run pytest -q tests/unit/test_grid5000_oar.py tests/unit/test_grid5000_state.py
  env UV_CACHE_DIR=/tmp/osm-polygon-sentence-classifier-t4-green uv run ty check
  ```

### Task 5: Implement the autonomous controller and short-job replacement

**Files:**
- Create: `src/osm_polygon_sentence_classifier/grid5000_autonomous.py`
- Create: `tests/unit/test_grid5000_autonomous.py`
- Modify: `src/osm_polygon_sentence_classifier/grid5000.py`

- [ ] **Step 1: Write pure policy/lifecycle tests.** Require `policy_type_for(now, 1800)` to return `day` only when the entire 30-minute allocation fits Monday-Friday 09:00–19:00 Europe/Paris and `night` otherwise. Require candidate selection to prefer a currently idle compatible prepared site, then a compatible unprepared site, then deterministic name order; queue count must not affect the result.

- [ ] **Step 2: Write controller tests with fake site/OAR/clock adapters.** Cover these exact sequences:
  1. all-site probe finds a Grenoble-style standard/production idle GPU, stages it, rechecks it, submits exactly one day job, waits for `Running`, then waits for terminal success;
  2. no idle GPU exists, one 30-minute night fallback is submitted and monitored without a second job;
  3. fallback is queued with a forecast over ten minutes, one day trial is submitted on an idle site, trial becomes `Running`, fallback is cancelled, and only the trial remains active;
  4. fallback starts first, trial is cancelled;
  5. trial misses the ten-minute immediate-start window, is cancelled, and fallback is retained;
  6. a submission error with no matching OAR job is reconciled once and reported without retrying;
  7. a submission error with an ambiguous matching job stops with durable `ambiguous` state;
  8. Ctrl-C/KeyboardInterrupt leaves active OAR work and durable state intact for `resume`.

- [ ] **Step 3: Implement the controller as dependency-injected orchestration.** The public `run` flow must be:

  ```text
  validate local clean pinned checkout and local HF auth
  -> create immutable run identity and durable run state
  -> probe every configured site concurrently
  -> preflight policy/quota/resource facts
  -> prepare one selected site and recheck facts
  -> submit one 30-minute day job when immediately idle
     or one 30-minute night fallback when no day candidate exists
  -> monitor status and bounded log offsets
  -> if fallback forecast is distant, try one 10-minute day replacement trial at a time
  -> classify terminal result
  -> read/validate completion manifest
  -> verify model publication and Trackio Space ID
  -> clean only pipeline-owned remote paths after verification
  -> persist complete state and stop
  ```

  The controller must never use queue depth as ETA, never submit to more than one trial site at a time, and never cancel a fallback before the replacement is confirmed `Running`. Poll intervals, immediate-start deadline, and site list must be injectable and bounded for tests.

- [ ] **Step 4: Add remote completion manifest support.** The worker must atomically write a JSON manifest containing only run ID, source/dataset/model revisions, model output relative path, model publication commit/url/files, and Trackio Space ID. The controller must reject missing, malformed, mismatched, or unpublished manifests and preserve remote data for diagnosis on failure.

- [ ] **Step 5: Implement local Hub verification.** Add a lazy `HfApi.model_info` check that requires the target model repository to contain `config.json`, tokenizer files, and model weights at or after the recorded publication commit. Do not print or persist the token. Trackio verification is the non-empty static Space ID returned by the worker plus the configured Space/Bucket IDs in code, not a credential-bearing response.

- [ ] **Step 6: Run autonomous controller tests green.**

  ```bash
  env UV_CACHE_DIR=/tmp/osm-polygon-sentence-classifier-t5-green uv run pytest -q tests/unit/test_grid5000_autonomous.py
  env UV_CACHE_DIR=/tmp/osm-polygon-sentence-classifier-t5-green uv run ruff check src/osm_polygon_sentence_classifier/grid5000_autonomous.py
  env UV_CACHE_DIR=/tmp/osm-polygon-sentence-classifier-t5-green uv run ty check
  ```

### Task 6: Expose the one-command workflow and recovery commands

**Files:**
- Modify: `src/osm_polygon_sentence_classifier/grid5000_cli.py`
- Modify: `pyproject.toml` only if a new entry point is necessary; prefer the existing `grid5000-landuse` entry point.
- Modify: `tests/unit/test_grid5000_cli.py`
- Modify: `docs/reference/grid5000-operator.md`
- Modify: `docs/reference/grid5000-boundary.md`
- Modify: `docs/guides/getting-started.md`
- Modify: `docs/architecture/overview.md`

- [ ] **Step 1: Write CLI red tests.** Require the following command to build an autonomous request without requiring site, source commit, or model revision when defaults can be resolved locally:

  ```bash
  uv run grid5000-landuse run --publish --sync-trackio --execute
  ```

  The parser must default to all `DEFAULT_SITES`, 1,800 seconds, automatic Europe/Paris policy, the pinned dataset revision, the pinned model revision, and the current clean local Git HEAD. Require `--site` to remain repeatable for a bounded test or operator override, and require `--keep-remote` to disable success cleanup explicitly.

- [ ] **Step 2: Implement `run`, `status`, and `resume`.** Keep `plan` side-effect free and `submit` as a compatibility primitive. `run --execute` is the one explicit external-state gate. `status RUN_ID` is local/read-only. `resume RUN_ID` loads durable identity, reconciles any submitting/live attempt, and continues monitoring without creating a duplicate run identity.

- [ ] **Step 3: Add CLI error handling tests.** Verify missing local HF auth, dirty checkout, invalid source/model revision, no compatible site, ambiguous state, and failed publication return non-zero without deleting remote paths or submitting a second job.

- [ ] **Step 4: Document the exact workflow.** Document that the operator automatically scans all configured frontends, selects queue/resource type from `oarnodes`, prepares the exact checkout and token, uses 30-minute day/night jobs, replaces only after a running trial, uploads the model and Trackio metrics, verifies completion, and removes only marked pipeline-owned remote data. Document that Ctrl-C leaves the job running and the printed `resume RUN_ID` is the recovery command.

- [ ] **Step 5: Run CLI/docs validation.**

  ```bash
  env UV_CACHE_DIR=/tmp/osm-polygon-sentence-classifier-t6 uv run pytest -q tests/unit/test_grid5000_cli.py
  env UV_CACHE_DIR=/tmp/osm-polygon-sentence-classifier-t6 uv run mkdocs build --strict --site-dir /tmp/osm-polygon-sentence-classifier-autonomous-site
  ```

### Task 7: Full regression gate and controlled external verification

**Files:**
- Modify only files justified by failing validation from Tasks 1–6.

- [ ] **Step 1: Run the complete local gate.**

  ```bash
  env UV_CACHE_DIR=/tmp/osm-polygon-sentence-classifier-final-uv-cache RUFF_CACHE_DIR=/tmp/osm-polygon-sentence-classifier-final-ruff-cache just check
  env UV_CACHE_DIR=/tmp/osm-polygon-sentence-classifier-final-uv-cache uv run pre-commit run --all-files
  env UV_CACHE_DIR=/tmp/osm-polygon-sentence-classifier-final-uv-cache uv run mkdocs build --strict --site-dir /tmp/osm-polygon-sentence-classifier-final-site
  git diff --check
  ```

  Expected: zero test failures, Ruff/Ty/pre-commit success, strict documentation build success, and no diff whitespace errors.

- [ ] **Step 2: Run only read-only live gates.** From the Mac, invoke the new `run --plan`/dry-run path and independently probe every configured frontend. Confirm no OAR submission occurs in plan mode, all outputs omit credentials, and selection for the current inventory chooses a resource class matching the live `production`/`exotic` facts. Do not run the real autonomous command in this validation task.

- [ ] **Step 3: Review the final diff.** Check that no code contains hard-coded Nancy-only behavior, raw token interpolation, broad remote deletion, queue-depth ETA logic, speculative multi-site submission, unbounded retries, or hidden CPU fallback. Confirm the legacy plan/submit tests still pass.

- [ ] **Step 4: Commit and push only after all gates pass.** Use a Conventional Commit such as:

  ```bash
  git add src tests docs pyproject.toml
  git commit -m "feat: automate Grid5000 landuse training"
  git push origin main
  ```

  Verify local `HEAD`, `origin/main`, and GitHub `refs/heads/main` are identical. Do not submit a training job as part of this implementation pass; the user will review the one-command workflow before the first autonomous production run.

## Acceptance checklist

- [ ] One documented command owns preparation, all-site discovery, dynamic OAR resource selection, policy checks, short-job submission, monitoring, safe replacement, completion verification, HF model publication, Trackio synchronization, and marked-path cleanup.
- [ ] The command works with both Grenoble-style `production + standard` resources and Nancy-style `default + exotic` resources.
- [ ] The target worker installs the locked `training` extra and retains no large environment/cache under persistent home storage.
- [ ] Every external mutation is behind `--execute`, durable state is atomic and credential-free, and interrupted/ambiguous runs cannot create duplicates.
- [ ] No live training job is submitted manually during implementation; live verification remains read-only until the code is reviewed.
- [ ] Full tests, lint, type checks, pre-commit, docs, and diff checks pass; main is committed and pushed.
