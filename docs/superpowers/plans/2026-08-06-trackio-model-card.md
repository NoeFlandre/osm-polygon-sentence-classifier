# Live Trackio and generated model-card plan

## Goal

Make explicitly published landuse runs expose live Trackio metrics and publish a
deterministic, credential-free model README at every complete checkpoint and
at final publication. Keep the active Grid'5000 run and the legacy static
Space untouched.

## Design

- Keep Transformers' built-in Trackio callback as the only metric logger.
- Provision the dedicated Gradio Space and Trackio Bucket before a run.
- Pass the live Space and Bucket identifiers explicitly to TrainingArguments.
- Keep the legacy static synchronization helper available only for compatibility;
  new training runs do not call it.
- Render model cards from allowlisted scalar metadata and metrics only.
- Write each checkpoint card inside its checkpoint directory before queuing its
  asynchronous Hub upload, preventing shared-root races.

## Implementation

1. Add live Space deployment to tracking resource provisioning and cover the
   repo, bucket, and deploy calls in tests.
2. Wire live Trackio identifiers through training and remove per-checkpoint
   static synchronization.
3. Add safe model-card rendering and include README files in final and
   checkpoint publication operations.
4. Update README and operator/data-policy/architecture/getting-started docs.
5. Run unit tests, full tests, Ruff, Ty, MkDocs, pre-commit, and diff checks.
6. Review the complete diff, commit with:

   fix: publish live Trackio metrics and model cards

7. Push only main and verify local HEAD equals origin/main.

## Acceptance criteria

- The live Space is a real Gradio Trackio deployment, not a static placeholder.
- Trackio-enabled training passes the live Space and Bucket IDs and returns the
  live Space ID in TrainingResult.
- Checkpoint and final model Hub commits include generated README files.
- README content contains model, dataset, source, configuration, progress, and
  scalar metric facts without credentials, sentence text, or raw rows.
- Existing publication, checkpoint, resume, filtering, and error contracts stay
  unchanged.
- No new Grid'5000 job is started by this change.
- The worktree is clean and main is synchronized after the commit and push.
