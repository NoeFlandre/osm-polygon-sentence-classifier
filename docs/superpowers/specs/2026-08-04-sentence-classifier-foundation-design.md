# Sentence Classifier Foundation Design

## Goal

Create the local, testable foundation for a sentence classifier whose first
task is landuse classification. The foundation will establish package
boundaries, reproducible development tooling, safe local-data handling,
experiment-tracking boundaries, and documentation for the later Grid'5000
training workflow.

The training dataset is the existing Hugging Face dataset
`NoeFlandre/osm-polygon-wikidata-sentence-relevance`. The eventual model and
derived data will be published to the existing Hugging Face repository
`NoeFlandre/osm-polygon-sentence-classifier`.

## Scope

This milestone includes:

- a Python package managed by `uv`;
- a small configuration and path-policy API;
- an explicit landuse task identity without inventing a label schema;
- a thin Trackio boundary whose local storage is under the approved external
  data root;
- unit and repository-contract tests written RED -> GREEN;
- Ruff, ty, pytest, pre-commit, Just, and GitHub Actions integration;
- a strict MkDocs Material site and a GitHub Pages workflow;
- public documentation for setup, development, data ownership, and the
  planned Grid'5000 boundary;
- a local Git repository with the GitHub origin recorded, without pushing or
  publishing anything in this milestone.

This milestone does not include:

- downloading or transforming the Hugging Face dataset;
- selecting a model architecture, tokenizer, label set, split policy, or
  evaluation metric;
- training, checkpointing, or model inference;
- authenticating to Hugging Face or uploading data, models, or Trackio runs;
- submitting, cancelling, or monitoring Grid'5000/OAR allocations;
- creating remote Hugging Face Spaces, Buckets, or repositories;
- storing local datasets, checkpoints, models, caches, or experiment logs in
  the Git repository or an unapproved local directory.

## Design alternatives

### Package-first foundation (selected)

Build the smallest package that can own the task identity, data-root policy,
and tracking boundary, then add the training and Grid'5000 subsystems after
their contracts are known. This keeps the initial change reviewable and lets
tests lock down the local-data rule before any expensive or irreversible
workflow exists.

### Training-first foundation

Add Hugging Face dataset loading and a baseline model immediately. This would
produce an experiment faster, but it would make unreviewed assumptions about
the current dataset schema, labels, splits, and model runtime part of the
public package before the classifier contract is designed.

### Operator-first foundation

Implement Grid'5000 reservation, SSH, checkpoint, and resume machinery before
training exists. This would front-load the most operationally complex part and
could reproduce useful safeguards from the reference project, but it would
also create a large subsystem with no classifier behavior to exercise.

The package-first alternative is selected because it satisfies the requested
foundation while preserving YAGNI and leaving real-data and remote-operation
gates explicit.

## Architecture

The package will have three initial responsibilities:

1. **Project configuration** identifies the project, first task, source
   dataset, and eventual model repository. It is immutable after construction
   and contains no credentials.
2. **Managed paths** resolve paths relative to the exact approved external
   root, `/Volumes/Seagate M3/projects/osm-polygon-sentence-classifier`.
   Paths outside that root are rejected before any future read or write.
3. **Experiment tracking** exposes a narrow application-owned boundary around
   Trackio. The boundary will configure Trackio's local directory beneath the
   managed root and defer the actual `trackio.init`, `trackio.log`, and
   `trackio.finish` calls to training code that does not exist yet.

The package will not expose dataset rows, labels, model classes, or scheduler
objects until a later design establishes those contracts.

## Proposed repository layout

```text
osm-polygon-sentence-classifier/
├── .github/workflows/
│   ├── ci.yml
│   └── docs.yml
├── docs/
│   ├── architecture/overview.md
│   ├── guides/data-policy.md
│   ├── guides/development.md
│   ├── guides/getting-started.md
│   ├── reference/grid5000-boundary.md
│   └── index.md
├── src/osm_polygon_sentence_classifier/
│   ├── __init__.py
│   ├── config.py
│   ├── paths.py
│   └── tracking.py
├── tests/
│   ├── unit/
│   │   ├── test_config.py
│   │   ├── test_paths.py
│   │   └── test_tracking.py
│   └── contract/
│       ├── test_repository_hygiene.py
│       └── test_tooling_contract.py
├── .gitignore
├── .pre-commit-config.yaml
├── CONTRIBUTING.md
├── README.md
├── justfile
├── mkdocs.yml
├── pyproject.toml
└── uv.lock
```

The layout is deliberately smaller than the read-only sentence-relevance
reference. Its Grid'5000 operator remains evidence for later operational
contracts, not a source tree to copy into this repository.

## Configuration and data policy

The initial configuration will expose these non-secret values:

- project name: `osm-polygon-sentence-classifier`;
- first task: `landuse`;
- source dataset: `NoeFlandre/osm-polygon-wikidata-sentence-relevance`;
- target model repository: `NoeFlandre/osm-polygon-sentence-classifier`;
- local data root: `/Volumes/Seagate M3/projects/osm-polygon-sentence-classifier`.

The path API will accept only relative child paths and will reject absolute
paths, `..` escapes, and paths that resolve outside the canonical root. It
will not create the external volume or any data directories during import,
testing, or package initialization. Future commands will explicitly create
only named, pipeline-owned subdirectories beneath the root.

The repository will ignore common generated directories such as `.venv`,
build output, caches, checkpoints, models, runs, and local data. Ignoring a
path is only a hygiene measure; the runtime path policy is the enforcement
boundary.

## Trackio boundary

Trackio will be a training dependency and will be imported lazily by the
tracking boundary. The initial configuration will derive a directory such as
`<data-root>/tracking` and set `TRACKIO_DIR` for future local tracking runs.
The foundation will not start a Trackio run or create local tracking files.

Remote Trackio Spaces, Buckets, write tokens, and synchronization are outside
this milestone. Any future remote tracking step must be separately authorized,
documented, and verified against the live remote object.

## Grid'5000 boundary

The first documentation will record the operational rules learned from the
read-only reference project:

- use a durable immutable run identity;
- stage and validate a complete runtime before a production allocation;
- perform live usage-policy and home-quota checks before submission;
- use bounded allocations with scheduler margin;
- persist checkpoints and job identifiers outside the repository;
- resume the exact recorded run rather than submitting a duplicate;
- treat Ctrl-C as a monitoring stop that leaves remote work recoverable;
- publish only after complete output validation.

No implementation will invoke SSH, OAR, Grid5000, or a remote shell during
this foundation milestone. The later operator design must define its own
tests, state format, security policy, and external-write gates before it is
implemented.

## Tooling and quality gates

`pyproject.toml` will be the source of truth for package metadata and tool
configuration. `uv.lock` will lock the environment. The initial development
group will include pytest, Ruff, ty, pre-commit, MkDocs, and MkDocs Material;
Trackio will be available through the training dependency group.

The Justfile will provide these focused recipes:

- `format` and `format-check`;
- `lint`;
- `typecheck`;
- `test`;
- `docs`;
- `check` for the complete local gate.

CI will run the locked environment, tests, Ruff formatting/checking, ty, and
the strict MkDocs build. The Pages workflow will build only the generated
documentation artifact and will not include the external data root.

## Testing strategy

Each Python behavior will follow RED -> GREEN -> REFACTOR:

1. write one focused failing test;
2. run that test and record the expected missing-behavior failure;
3. implement the smallest behavior that passes;
4. run the focused test and then the complete suite;
5. refactor only while the suite remains green.

The first tests will prove:

- project configuration exposes the agreed non-secret task and repository
  identities;
- a managed child path resolves beneath the approved root;
- absolute paths, traversal paths, and escaped resolved paths are rejected;
- Trackio configuration points beneath the managed root without importing or
  starting a remote service;
- required tooling and documentation files exist with the documented command
  names;
- repository hygiene excludes local data, credentials, caches, and generated
  artifacts.

No test will download the dataset, write to Hugging Face, allocate Grid'5000
resources, or require a mounted Seagate volume.

## Acceptance criteria

The foundation is ready for the next classifier-design milestone when:

- the target repository has the documented layout and a clean local Git
  history containing the design and foundation commits;
- `uv sync --locked` succeeds in a clean checkout;
- `uv run pytest -q`, `uv run ruff format --check .`, `uv run ruff check .`,
  and `uv run ty check` succeed;
- `uv run mkdocs build --strict` succeeds without reading the external data
  root;
- the path-policy tests prove that no approved application path can escape
  the Seagate root;
- no dataset, model, credential, remote job, Trackio run, or Hugging Face
  object was created as part of this milestone;
- the GitHub origin is recorded locally, but remote synchronization and
  publication remain explicit follow-up actions.
