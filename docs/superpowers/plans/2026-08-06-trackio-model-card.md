# Static Trackio and generated model-card plan

## Goal

Make explicitly published landuse runs expose free, static Trackio snapshots
and publish a deterministic, credential-free model README at every complete
checkpoint and at final publication.

## Design

- Keep the Trainer's local Trackio callback for metric logging.
- Provision the dedicated static Space and Trackio Bucket before a run.
- Synchronize the local project explicitly with `trackio.sync(..., sdk="static")`
  after each complete checkpoint and final publication.
- Do not deploy Gradio or pass remote Trackio identifiers to TrainingArguments;
  this keeps the workflow available on a free HF account.
- Render model cards from allowlisted scalar metadata and metrics only.
- Write each checkpoint card inside its checkpoint directory before queuing its
  asynchronous Hub upload, preventing shared-root races.

## Implementation

1. Provision the static Space and Bucket idempotently and cover the repository
   and bucket calls in tests.
2. Keep local metric logging in training and add checkpoint/final static
   synchronization with explicit error handling.
3. Add safe model-card rendering and include README files in final and
   checkpoint publication operations.
4. Update README and operator/data-policy/architecture/getting-started docs.
5. Run unit tests, full tests, Ruff, Ty, MkDocs, pre-commit, and diff checks.
6. Review the complete diff, commit with:

   fix: publish free static Trackio snapshots

7. Push only main and verify local HEAD equals origin/main.

## Acceptance criteria

- The Trackio Space is a static Space and does not require paid HF compute.
- Trackio-enabled training records metrics locally and syncs a static snapshot
  after each complete checkpoint and final publication.
- TrainingResult and generated model cards identify the static Trackio Space.
- Checkpoint and final model Hub commits include generated README files.
- README content contains model, dataset, source, configuration, progress, and
  scalar metric facts without credentials, sentence text, or raw rows.
- Existing publication, checkpoint, resume, filtering, and error contracts stay
  unchanged.
- No new Grid'5000 job is started by this change.
- The worktree is clean and main is synchronized after the commit and push.
