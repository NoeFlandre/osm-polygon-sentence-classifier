# Sentence Classifier Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the local, testable foundation for a landuse sentence classifier without downloading data, training a model, allocating Grid'5000 resources, or mutating Hugging Face.

**Architecture:** Keep the package limited to immutable project metadata, a managed external-data path policy, and a Trackio configuration boundary. Put all future dataset, model, and Grid'5000 behavior behind later reviewed contracts. Treat MkDocs, CI, Just, pre-commit, and repository hygiene as first-class tested project infrastructure.

**Tech Stack:** Python 3.12+, `uv`, setuptools, pytest, Ruff, ty, pre-commit, Just, MkDocs Material, Trackio as a training dependency, and GitHub Actions with pinned action SHAs.

---

## Repository and safety assumptions

- Work only in `/Users/noeflandre/osm-polygon-sentence-classifier`.
- The local project-data root is exactly `/Volumes/Seagate M3/projects/osm-polygon-sentence-classifier`.
- Do not create, download, inspect, or modify project data in the Seagate root during this plan. The path policy is tested without requiring the volume to be mounted.
- Do not modify `/Users/noeflandre/osm-polygon-wikidata-sentence-relevance`; it remains read-only reference material.
- Keep the existing local `origin` URL and do not push during implementation.
- Use Conventional Commit messages and commit each completed task after its verification gate.

## File map

| File | Responsibility |
| --- | --- |
| `pyproject.toml` | Package metadata, dependency groups, pytest/Ruff/ty configuration, and build configuration. |
| `uv.lock` | Reproducible dependency resolution. |
| `src/osm_polygon_sentence_classifier/__init__.py` | Package identity and version. |
| `src/osm_polygon_sentence_classifier/config.py` | Immutable non-secret project/task configuration and the fixed approved data root. |
| `src/osm_polygon_sentence_classifier/paths.py` | Containment-checked resolution of relative children beneath a configured root. |
| `src/osm_polygon_sentence_classifier/tracking.py` | Trackio project/directory settings without starting a run or importing Trackio. |
| `tests/unit/test_config.py` | Configuration behavior. |
| `tests/unit/test_paths.py` | Relative-path, traversal, and symlink-escape behavior. |
| `tests/unit/test_tracking.py` | Trackio directory/environment behavior. |
| `tests/contract/test_repository_hygiene.py` | Data, credential, cache, and generated-output repository rules. |
| `tests/contract/test_tooling_contract.py` | Required Just, pre-commit, CI, MkDocs, and Pages commands. |
| `README.md` | Public project status, setup, scope, and safe command entry points. |
| `CONTRIBUTING.md` | Development gates and external-write boundaries. |
| `docs/index.md` | Concise documentation landing page. |
| `docs/guides/getting-started.md` | Local setup and verification commands. |
| `docs/guides/development.md` | TDD, modularity, and contribution workflow. |
| `docs/guides/data-policy.md` | Exact local-data-root and remote-data policy. |
| `docs/architecture/overview.md` | Initial package boundaries and deferred subsystems. |
| `docs/reference/grid5000-boundary.md` | Read-only-derived operational boundary for a later implementation. |
| `mkdocs.yml` | Strict MkDocs Material site configuration and navigation. |
| `justfile` | Local command aliases for formatting, linting, typing, tests, docs, and the complete gate. |
| `.pre-commit-config.yaml` | Local hooks for formatting, linting, typing, and focused tests. |
| `.gitignore` | Repository hygiene for environments, caches, generated outputs, and project data. |
| `.github/workflows/ci.yml` | Locked quality/test checks on pushes and pull requests. |
| `.github/workflows/docs.yml` | Strict Pages build/deploy workflow; it does not publish data or models. |

## Task 1: Bootstrap the package metadata and repository hygiene

**Files:**

- Create: `pyproject.toml`
- Create: `README.md`
- Create: `.gitignore`
- Create: `CONTRIBUTING.md`
- Create: `src/osm_polygon_sentence_classifier/__init__.py`

The package marker is intentionally behavior-free. The version and public
configuration behavior are added only after their tests exist in later tasks.

- [ ] **Step 1: Add the minimal project configuration and package marker**

Create `pyproject.toml` with this content:

```toml
[project]
name = "osm-polygon-sentence-classifier"
version = "0.1.0"
description = "A sentence classifier for OSM polygon descriptions, starting with landuse."
readme = "README.md"
requires-python = ">=3.12"
authors = [
    { name = "NoeFlandre" },
]
keywords = [
    "openstreetmap",
    "wikidata",
    "sentence-classification",
    "landuse",
    "machine-learning",
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Typing :: Typed",
]
dependencies = []

[project.urls]
Homepage = "https://github.com/NoeFlandre/osm-polygon-sentence-classifier"
Repository = "https://github.com/NoeFlandre/osm-polygon-sentence-classifier"
Issues = "https://github.com/NoeFlandre/osm-polygon-sentence-classifier/issues"
Documentation = "https://noeflandre.github.io/osm-polygon-sentence-classifier/"

[project.optional-dependencies]
training = [
    "trackio>=0.26,<1",
]
docs = [
    "mkdocs-material>=9.6,<10",
]

[dependency-groups]
dev = [
    "pre-commit>=4,<5",
    "pytest>=8,<9",
    "ruff>=0.14,<0.15",
    "ty>=0.0.65,<0.1",
]

[build-system]
requires = ["setuptools>=77"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = ["--strict-config", "--strict-markers"]

[tool.ruff]
target-version = "py312"
line-length = 88
src = ["src", "tests"]

[tool.ruff.format]
quote-style = "double"

[tool.ruff.lint]
select = [
    "E",
    "W",
    "F",
    "I",
    "UP",
    "B",
    "C4",
    "SIM",
    "PT",
    "RUF100",
]
ignore = ["E501"]

[tool.ruff.lint.isort]
known-first-party = ["osm_polygon_sentence_classifier"]

[tool.ty.src]
include = ["src", "tests"]
```

Create `src/osm_polygon_sentence_classifier/__init__.py` with only:

```python
"""Local foundation for the OSM polygon sentence classifier."""
```

- [ ] **Step 2: Add the repository hygiene rules**

Create `.gitignore` with:

```gitignore
.venv/
__pycache__/
.pytest_cache/
.ruff_cache/
.ty_cache/
.mypy_cache/
.coverage
htmlcov/
build/
dist/
*.egg-info/
site/

# Project data and experiment outputs never belong in this repository.
data/
models/
checkpoints/
runs/
outputs/
tracking/
*.parquet
*.sqlite
*.sqlite3
*.jsonl

# Credentials and local configuration never belong in this repository.
.env
.env.*
!.env.example
```

- [ ] **Step 3: Add the initial README and contribution rules**

Create `README.md` with:

````markdown
# OSM Polygon Sentence Classifier

Train a sentence classifier for OSM polygon descriptions, starting with the
`landuse` task.

## Status

The repository currently contains the safe, testable project foundation. It
does not download data, train a model, submit Grid'5000 jobs, authenticate to
Hugging Face, or publish remote artifacts.

## Data and model repositories

- Training source: [NoeFlandre/osm-polygon-wikidata-sentence-relevance](https://huggingface.co/datasets/NoeFlandre/osm-polygon-wikidata-sentence-relevance)
- Eventual model repository: [NoeFlandre/osm-polygon-sentence-classifier](https://huggingface.co/NoeFlandre/osm-polygon-sentence-classifier)
- Local project-data root: `/Volumes/Seagate M3/projects/osm-polygon-sentence-classifier`

All local datasets, checkpoints, models, and experiment logs must be kept
beneath the external data root. Nothing in this repository downloads or
creates them yet.

## Development

```bash
uv sync --locked --all-extras --dev
uv run pytest -q
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run mkdocs build --strict --site-dir site
```

The equivalent Just recipes are documented in
[`docs/guides/getting-started.md`](docs/guides/getting-started.md).

## Grid'5000

Computation will run on Grid'5000 in a later milestone. The current safety
boundary is documented in
[`docs/reference/grid5000-boundary.md`](docs/reference/grid5000-boundary.md).
````

Create `CONTRIBUTING.md` with:

````markdown
# Contributing

## Required local gate

Use the locked environment and run:

```bash
uv sync --locked --all-extras --dev
just check
uv run mkdocs build --strict --site-dir site
```

New Python behavior follows RED -> GREEN -> REFACTOR: add a focused failing
test, observe the expected failure, implement the smallest passing behavior,
then run the complete gate.

## Scope boundaries

Do not place datasets, checkpoints, models, experiment logs, credentials, or
temporary job state in the repository. Local project data belongs beneath
`/Volumes/Seagate M3/projects/osm-polygon-sentence-classifier`.

Do not download the Hugging Face dataset, authenticate, upload, submit a
Grid'5000 job, or publish a model without an explicit task-level authorization
and a documented verification plan.
````

- [ ] **Step 4: Resolve and install the locked environment**

Run:

```bash
uv lock
uv sync --locked --all-extras --dev
```

Expected: `uv lock` writes `uv.lock`; the sync completes with exit code 0 and
does not create any project-data directory. If dependency resolution is
blocked by the sandbox network policy, rerun the same command with approved
network escalation; do not substitute an unlocked installer.

- [ ] **Step 5: Commit the bootstrap task**

Run:

```bash
git add pyproject.toml uv.lock README.md CONTRIBUTING.md .gitignore src/osm_polygon_sentence_classifier/__init__.py
git commit -m "chore: bootstrap classifier project"
```

## Task 2: Add immutable project configuration using RED -> GREEN

**Files:**

- Create: `tests/unit/test_config.py`
- Modify: `src/osm_polygon_sentence_classifier/__init__.py`
- Create: `src/osm_polygon_sentence_classifier/config.py`

- [ ] **Step 1: Write the failing configuration tests**

Create `tests/unit/test_config.py`:

```python
from pathlib import Path

import pytest

from osm_polygon_sentence_classifier import __version__
from osm_polygon_sentence_classifier.config import (
    APPROVED_DATA_ROOT,
    PROJECT_NAME,
    SOURCE_DATASET_ID,
    TARGET_MODEL_REPOSITORY_ID,
    ConfigurationError,
    ProjectConfig,
)


def test_package_exposes_the_foundation_version() -> None:
    assert __version__ == "0.1.0"


def test_default_config_identifies_the_landuse_task() -> None:
    config = ProjectConfig()

    assert config.project_name == PROJECT_NAME == "osm-polygon-sentence-classifier"
    assert config.task_name == "landuse"
    assert config.source_dataset_id == SOURCE_DATASET_ID
    assert config.target_model_repository_id == TARGET_MODEL_REPOSITORY_ID
    assert config.data_root == APPROVED_DATA_ROOT


def test_data_root_cannot_be_replaced_by_another_local_directory(
    tmp_path: Path,
) -> None:
    with pytest.raises(ConfigurationError, match="approved external data root"):
        ProjectConfig(data_root=tmp_path)


def test_config_is_immutable() -> None:
    config = ProjectConfig()

    with pytest.raises(AttributeError):
        config.task_name = "polygon-relevance"  # type: ignore[misc]
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
uv run pytest -q tests/unit/test_config.py
```

Expected: FAIL because `__version__` and the configuration module behavior do
not exist yet. If collection fails rather than producing assertions, add only
the missing module/package marker needed for collection, rerun, and continue
until the behavior assertions fail for the expected missing attributes.

- [ ] **Step 3: Implement the smallest configuration API**

Replace `src/osm_polygon_sentence_classifier/__init__.py` with:

```python
"""Local foundation for the OSM polygon sentence classifier."""

__version__ = "0.1.0"

__all__ = ["__version__"]
```

Create `src/osm_polygon_sentence_classifier/config.py`:

```python
from dataclasses import dataclass
from pathlib import Path

PROJECT_NAME = "osm-polygon-sentence-classifier"
TASK_NAME = "landuse"
SOURCE_DATASET_ID = "NoeFlandre/osm-polygon-wikidata-sentence-relevance"
TARGET_MODEL_REPOSITORY_ID = "NoeFlandre/osm-polygon-sentence-classifier"
APPROVED_DATA_ROOT = Path(
    "/Volumes/Seagate M3/projects/osm-polygon-sentence-classifier"
)


class ConfigurationError(ValueError):
    """Raised when immutable project configuration violates the contract."""


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """Non-secret configuration shared by future local and remote workflows."""

    data_root: Path = APPROVED_DATA_ROOT

    def __post_init__(self) -> None:
        root = Path(self.data_root).expanduser()
        if root != APPROVED_DATA_ROOT:
            raise ConfigurationError(
                "data_root must be the approved external data root: "
                f"{APPROVED_DATA_ROOT}"
            )
        object.__setattr__(self, "data_root", APPROVED_DATA_ROOT)

    @property
    def project_name(self) -> str:
        return PROJECT_NAME

    @property
    def task_name(self) -> str:
        return TASK_NAME

    @property
    def source_dataset_id(self) -> str:
        return SOURCE_DATASET_ID

    @property
    def target_model_repository_id(self) -> str:
        return TARGET_MODEL_REPOSITORY_ID
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Run:

```bash
uv run pytest -q tests/unit/test_config.py
```

Expected: all four tests pass with exit code 0.

- [ ] **Step 5: Run formatting and typing for the new Python files**

Run:

```bash
uv run ruff format src/osm_polygon_sentence_classifier tests/unit/test_config.py
uv run ruff check src/osm_polygon_sentence_classifier tests/unit/test_config.py
uv run ty check
```

Expected: all commands exit 0. Keep the formatter's output; do not manually
reformat it differently.

- [ ] **Step 6: Commit the configuration task**

Run:

```bash
git add src/osm_polygon_sentence_classifier/__init__.py src/osm_polygon_sentence_classifier/config.py tests/unit/test_config.py
git commit -m "feat: add immutable project configuration"
```

## Task 3: Add containment-checked managed paths using RED -> GREEN

**Files:**

- Create: `tests/unit/test_paths.py`
- Create: `src/osm_polygon_sentence_classifier/paths.py`

- [ ] **Step 1: Write the failing path-policy tests**

Create `tests/unit/test_paths.py`:

```python
from pathlib import Path

import pytest

from osm_polygon_sentence_classifier.config import ProjectConfig
from osm_polygon_sentence_classifier.paths import (
    ManagedPathError,
    ManagedPaths,
    resolve_managed_path,
)


def test_relative_child_resolves_under_the_supplied_root(tmp_path: Path) -> None:
    root = tmp_path / "approved"

    result = resolve_managed_path(root, "runs/first")

    assert result == (root / "runs/first").resolve()
    assert result.is_relative_to(root.resolve())


@pytest.mark.parametrize("candidate", ["/tmp/outside", "../outside", "runs/../../outside"])
def test_absolute_or_traversal_paths_are_rejected(
    tmp_path: Path,
    candidate: str,
) -> None:
    with pytest.raises(ManagedPathError, match="beneath the managed root"):
        resolve_managed_path(tmp_path / "approved", candidate)


def test_symlink_that_escapes_the_root_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "approved"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ManagedPathError, match="beneath the managed root"):
        resolve_managed_path(root, "escape/file.json")


def test_application_paths_use_the_fixed_project_root() -> None:
    paths = ManagedPaths(ProjectConfig())

    assert paths.child("tracking") == ProjectConfig().data_root / "tracking"
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
uv run pytest -q tests/unit/test_paths.py
```

Expected: FAIL during collection because `paths.py` does not expose the
requested API. Add no behavior before recording this expected failure.

- [ ] **Step 3: Implement the smallest containment policy**

Create `src/osm_polygon_sentence_classifier/paths.py`:

```python
from dataclasses import dataclass
from pathlib import Path

from .config import ProjectConfig


class ManagedPathError(ValueError):
    """Raised when a path would escape the managed data root."""


def resolve_managed_path(root: Path, relative_path: str | Path) -> Path:
    """Resolve a child path and require it to remain beneath ``root``."""

    child = Path(relative_path)
    canonical_root = root.resolve()
    if child.is_absolute():
        raise ManagedPathError("path must remain beneath the managed root")

    candidate = (canonical_root / child).resolve()
    if not candidate.is_relative_to(canonical_root):
        raise ManagedPathError("path must remain beneath the managed root")
    return candidate


@dataclass(frozen=True, slots=True)
class ManagedPaths:
    """Application-owned paths derived from the fixed project configuration."""

    config: ProjectConfig

    def child(self, relative_path: str | Path) -> Path:
        return resolve_managed_path(self.config.data_root, relative_path)
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Run:

```bash
uv run pytest -q tests/unit/test_paths.py
```

Expected: all four test functions pass, including all parameterized cases.

- [ ] **Step 5: Run the complete Python suite and static checks**

Run:

```bash
uv run pytest -q
uv run ruff format --check .
uv run ruff check .
uv run ty check
```

Expected: exit code 0 for every command.

- [ ] **Step 6: Commit the path-policy task**

Run:

```bash
git add src/osm_polygon_sentence_classifier/paths.py tests/unit/test_paths.py
git commit -m "feat: enforce managed data paths"
```

## Task 4: Add the Trackio configuration boundary using RED -> GREEN

**Files:**

- Create: `tests/unit/test_tracking.py`
- Create: `src/osm_polygon_sentence_classifier/tracking.py`

- [ ] **Step 1: Write the failing Trackio-boundary tests**

Create `tests/unit/test_tracking.py`:

```python
from osm_polygon_sentence_classifier.config import ProjectConfig
from osm_polygon_sentence_classifier.tracking import (
    TRACKING_SUBDIRECTORY,
    TrackioSettings,
    settings_for,
)


def test_tracking_settings_use_the_project_name_and_managed_directory() -> None:
    config = ProjectConfig()

    settings = settings_for(config)

    assert isinstance(settings, TrackioSettings)
    assert settings.project == "osm-polygon-sentence-classifier"
    assert settings.directory == config.data_root / TRACKING_SUBDIRECTORY
    assert settings.directory.is_relative_to(config.data_root)


def test_tracking_environment_only_points_trackio_at_managed_storage() -> None:
    settings = settings_for(ProjectConfig())

    assert settings.environment() == {"TRACKIO_DIR": str(settings.directory)}


```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
uv run pytest -q tests/unit/test_tracking.py
```

Expected: FAIL during collection because `tracking.py` does not expose the
requested settings API.

- [ ] **Step 3: Implement Trackio settings without starting Trackio**

Create `src/osm_polygon_sentence_classifier/tracking.py`:

```python
from dataclasses import dataclass
from pathlib import Path

from .config import ProjectConfig
from .paths import resolve_managed_path

TRACKING_SUBDIRECTORY = Path("tracking")


@dataclass(frozen=True, slots=True)
class TrackioSettings:
    """Non-secret Trackio settings for a future training process."""

    project: str
    directory: Path

    def environment(self) -> dict[str, str]:
        """Return environment values needed to keep local Trackio data managed."""

        return {"TRACKIO_DIR": str(self.directory)}


def settings_for(config: ProjectConfig) -> TrackioSettings:
    """Build Trackio settings without importing or initializing Trackio."""

    directory = resolve_managed_path(config.data_root, TRACKING_SUBDIRECTORY)
    return TrackioSettings(project=config.project_name, directory=directory)
```

- [ ] **Step 4: Run the focused test and verify GREEN**

Run:

```bash
uv run pytest -q tests/unit/test_tracking.py
```

Expected: both tests pass. The settings function returns paths and environment
values only; it does not create the Seagate tracking directory.

- [ ] **Step 5: Run the complete Python gate**

Run:

```bash
uv run pytest -q
uv run ruff format --check .
uv run ruff check .
uv run ty check
```

Expected: exit code 0 for every command.

- [ ] **Step 6: Commit the Trackio boundary**

Run:

```bash
git add src/osm_polygon_sentence_classifier/tracking.py tests/unit/test_tracking.py
git commit -m "feat: define managed Trackio settings"
```

## Task 5: Add tooling, documentation, and Pages configuration behind contract tests

**Files:**

- Create: `tests/contract/test_tooling_contract.py`
- Create: `tests/contract/test_repository_hygiene.py`
- Create: `justfile`
- Create: `.pre-commit-config.yaml`
- Create: `mkdocs.yml`
- Create: `.github/workflows/ci.yml`
- Create: `.github/workflows/docs.yml`
- Create: `docs/index.md`
- Create: `docs/guides/getting-started.md`
- Create: `docs/guides/development.md`
- Create: `docs/guides/data-policy.md`
- Create: `docs/architecture/overview.md`
- Create: `docs/reference/grid5000-boundary.md`

- [ ] **Step 1: Write failing repository-contract tests**

Create `tests/contract/test_tooling_contract.py`:

```python
from pathlib import Path

ROOT = Path(__file__).parents[2]


def _text(relative_path: str) -> str:
    path = ROOT / relative_path
    return path.read_text(encoding="utf-8") if path.exists() else ""


def test_justfile_exposes_the_documented_quality_recipes() -> None:
    content = _text("justfile")

    for recipe in ("format", "format-check", "lint", "typecheck", "test", "docs", "check"):
        assert f"{recipe}:" in content


def test_pre_commit_runs_the_locked_project_tools() -> None:
    content = _text(".pre-commit-config.yaml")

    assert "uv run ruff format --check ." in content
    assert "uv run ruff check ." in content
    assert "uv run ty check" in content
    assert "uv run pytest -q" in content


def test_ci_runs_locked_tests_and_static_checks() -> None:
    content = _text(".github/workflows/ci.yml")

    assert "uv sync --locked --all-extras --dev" in content
    assert "uv run pytest -q" in content
    assert "uv run ruff format --check ." in content
    assert "uv run ruff check ." in content
    assert "uv run ty check" in content


def test_pages_workflow_builds_strict_mkdocs_and_deploys_pages_artifact() -> None:
    content = _text(".github/workflows/docs.yml")

    assert "uv run mkdocs build --strict" in content
    assert "actions/upload-pages-artifact@" in content
    assert "actions/deploy-pages@" in content
    assert "pages: write" in content
    assert "id-token: write" in content


def test_training_dependency_declares_trackio() -> None:
    content = _text("pyproject.toml")

    assert "training = [" in content
    assert "trackio" in content


def test_mkdocs_uses_material_and_excludes_internal_superpowers_docs() -> None:
    content = _text("mkdocs.yml")

    assert "name: material" in content
    assert "superpowers/*" in content
    assert "reference/grid5000-boundary.md" in content
```

Create `tests/contract/test_repository_hygiene.py`:

```python
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_gitignore_keeps_project_data_and_credentials_out_of_git() -> None:
    content = (ROOT / ".gitignore").read_text(encoding="utf-8")

    for entry in ("data/", "models/", "checkpoints/", "runs/", ".env", ".venv/"):
        assert entry in content


def test_public_docs_name_the_exact_external_data_root() -> None:
    expected = "/Volumes/Seagate M3/projects/osm-polygon-sentence-classifier"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    policy = ROOT / "docs/guides/data-policy.md"
    policy_text = policy.read_text(encoding="utf-8") if policy.exists() else ""

    assert expected in readme
    assert expected in policy_text


def test_public_docs_do_not_contain_literal_hugging_face_tokens() -> None:
    for relative_path in ("README.md", "CONTRIBUTING.md"):
        content = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "hf_" not in content
```

- [ ] **Step 2: Run the contract tests and verify RED**

Run:

```bash
uv run pytest -q tests/contract
```

Expected: the tests fail because the required tooling and documentation files
do not exist yet.

- [ ] **Step 3: Add the Justfile and pre-commit hooks**

Create `justfile`:

```just
set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

format:
    uv run ruff format .

format-check:
    uv run ruff format --check .

lint:
    uv run ruff check .

typecheck:
    uv run ty check

test:
    uv run pytest -q

docs:
    uv run mkdocs build --strict --site-dir site

check: format-check lint typecheck test
```

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: local
    hooks:
      - id: ruff-format-check
        name: Ruff format check
        entry: uv run ruff format --check .
        language: system
        pass_filenames: false
      - id: ruff-check
        name: Ruff lint
        entry: uv run ruff check .
        language: system
        pass_filenames: false
      - id: ty-check
        name: ty type check
        entry: uv run ty check
        language: system
        pass_filenames: false
      - id: focused-tests
        name: Focused tests
        entry: uv run pytest -q tests/unit tests/contract
        language: system
        pass_filenames: false
```

- [ ] **Step 4: Add strict MkDocs Material configuration**

Create `mkdocs.yml`:

```yaml
site_name: OSM Polygon Sentence Classifier
site_description: A landuse-first sentence classifier for OSM polygon descriptions.
site_url: https://noeflandre.github.io/osm-polygon-sentence-classifier/
repo_name: NoeFlandre/osm-polygon-sentence-classifier
repo_url: https://github.com/NoeFlandre/osm-polygon-sentence-classifier
edit_uri: edit/main/docs/

theme:
  name: material
  language: en
  features:
    - navigation.sections
    - navigation.top
    - content.code.copy
    - search.highlight

markdown_extensions:
  - admonition
  - attr_list
  - pymdownx.details
  - pymdownx.highlight
  - pymdownx.superfences
  - tables
  - toc:
      permalink: true

exclude_docs: |
  superpowers/*
  **/.DS_Store

nav:
  - Home: index.md
  - Guides:
      - Getting started: guides/getting-started.md
      - Development: guides/development.md
      - Data policy: guides/data-policy.md
  - Architecture:
      - Overview: architecture/overview.md
  - Reference:
      - Grid'5000 boundary: reference/grid5000-boundary.md
```

- [ ] **Step 5: Add the public documentation pages**

Create `docs/index.md`:

````markdown
# OSM Polygon Sentence Classifier

This project will train a sentence classifier for OSM polygon descriptions,
starting with landuse classification.

The current milestone is the safe local foundation: package boundaries,
managed data paths, Trackio settings, tests, tooling, and documentation. It
does not process the dataset or submit remote jobs.

Start with [Getting started](guides/getting-started.md), then read the
[data policy](guides/data-policy.md) and the
[Grid'5000 boundary](reference/grid5000-boundary.md).
````

Create `docs/guides/getting-started.md`:

````markdown
# Getting started

## Prerequisites

- Python 3.12 or newer
- `uv`
- Git

The Seagate volume is not required for this foundation because no command
creates project data.

## Install the locked environment

```bash
uv sync --locked --all-extras --dev
```

## Run the local gate

```bash
just check
just docs
```

The equivalent commands are:

```bash
uv run pytest -q
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run mkdocs build --strict --site-dir site
```

No command on this page downloads the Hugging Face dataset, authenticates, or
submits a Grid'5000 job.
````

Create `docs/guides/development.md`:

````markdown
# Development

Keep modules small and single-purpose. Do not add a dataset loader, model
class, label schema, or Grid'5000 operator until its contract has a reviewed
design.

## RED -> GREEN -> REFACTOR

For each Python behavior:

1. write one focused failing test;
2. run it and confirm the expected failure;
3. implement the smallest passing behavior;
4. run the focused test and the complete gate;
5. refactor only while tests remain green.

Use `uv` for dependency and command execution, Ruff for formatting/linting,
ty for static typing, pytest for tests, and the Just recipes for repeatable
local gates.
````

Create `docs/guides/data-policy.md`:

````markdown
# Data policy

The only approved local project-data root is:

```text
/Volumes/Seagate M3/projects/osm-polygon-sentence-classifier
```

Future datasets, checkpoints, models, Trackio logs, and resumable run state
must be placed beneath this root. The repository contains no project data.
The runtime path policy rejects absolute paths, traversal paths, and symlink
escapes before future reads or writes.

The training source is the read-only Hugging Face dataset
`NoeFlandre/osm-polygon-wikidata-sentence-relevance`. The eventual model
repository is `NoeFlandre/osm-polygon-sentence-classifier`.

This foundation does not download, transform, upload, or publish any of them.
Credentials must be supplied through the supported external CLI/runtime
authentication mechanisms and must never be written to the repository or
documentation.
````

Create `docs/architecture/overview.md`:

````markdown
# Architecture overview

The initial package has three boundaries:

- `config.py` owns immutable non-secret project and task identity;
- `paths.py` owns containment-checked paths beneath the approved external
  root;
- `tracking.py` owns Trackio directory/project settings without starting a
  tracking run.

Dataset schemas, labels, models, metrics, checkpoints, and Grid'5000/OAR
operations are intentionally deferred until their contracts are designed.
````

Create `docs/reference/grid5000-boundary.md`:

````markdown
# Grid'5000 boundary

Training computation will run on Grid'5000 in a later milestone. This page
records the operational boundary learned from the read-only
`osm-polygon-wikidata-sentence-relevance` project; it is not an implementation
or a submission command.

The later operator must:

- use an immutable run identity and durable job/checkpoint state;
- validate the staged runtime before production allocation;
- perform live usage-policy and home-quota checks before submission;
- use bounded allocations with scheduler margin;
- preserve checkpoints across allocation boundaries;
- resume the exact recorded run after interruption rather than submitting a
  duplicate;
- leave remote work and checkpoints intact when local monitoring is stopped;
- publish only after complete output validation.

The foundation makes no SSH, OAR, Grid'5000, dataset, model, Trackio remote,
or Hugging Face calls.
````

- [ ] **Step 6: Add CI and Pages workflows with least privilege**

Create `.github/workflows/ci.yml`:

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

permissions:
  contents: read

concurrency:
  group: ci-${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2
      - name: Install uv
        uses: astral-sh/setup-uv@08807647e7069bb48b6ef5acd8ec9567f424441b # v8.1.0
        with:
          enable-cache: true
      - name: Set up Python
        run: uv python install 3.12
      - name: Sync locked environment
        run: uv sync --locked --all-extras --dev
      - name: Run tests
        run: uv run pytest -q
      - name: Check formatting
        run: uv run ruff format --check .
      - name: Run Ruff
        run: uv run ruff check .
      - name: Run ty
        run: uv run ty check
      - name: Build strict documentation
        run: uv run mkdocs build --strict --site-dir site
```

Create `.github/workflows/docs.yml`:

```yaml
name: Documentation

on:
  push:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read

concurrency:
  group: pages
  cancel-in-progress: true

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - uses: actions/checkout@11bd71901bbe5b1630ceea73d27597364c9af683 # v4.2.2
      - name: Install uv
        uses: astral-sh/setup-uv@08807647e7069bb48b6ef5acd8ec9567f424441b # v8.1.0
        with:
          enable-cache: true
      - name: Set up Python
        run: uv python install 3.12
      - name: Sync documentation environment
        run: uv sync --locked --extra docs
      - name: Build strict site
        run: uv run mkdocs build --strict --site-dir site
      - name: Upload Pages artifact
        uses: actions/upload-pages-artifact@56afc609e74202658d3ffba0e8f6dda462b719fa # v3.0.1
        with:
          path: site

  deploy:
    needs: build
    runs-on: ubuntu-latest
    permissions:
      pages: write
      id-token: write
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - name: Deploy documentation
        id: deployment
        uses: actions/deploy-pages@d6db90164ac5ed86f2b6aed7e0febac5b3c0c03e # v4.0.5
```

- [ ] **Step 7: Run the contract tests and verify GREEN**

Run:

```bash
uv run pytest -q tests/contract
```

Expected: all contract tests pass. If a workflow assertion fails, fix the
workflow/configuration file to satisfy the stated observable contract; do not
weaken the test.

- [ ] **Step 8: Build the documentation locally**

Run:

```bash
uv run mkdocs build --strict --site-dir site
test -f site/index.html
test -f site/guides/data-policy/index.html
test -f site/reference/grid5000-boundary/index.html
```

Expected: strict build exits 0 and all three pages exist. `site/` is ignored
and contains documentation output only.

- [ ] **Step 9: Run the complete local gate**

Run:

```bash
uv run pre-commit run --all-files
just check
just docs
git diff --check
```

Expected: every hook and command exits 0. The output must contain no token,
dataset-download, remote-job, or upload activity.

- [ ] **Step 10: Commit tooling and documentation**

Run:

```bash
git add .github .pre-commit-config.yaml docs/index.md docs/guides docs/architecture docs/reference mkdocs.yml justfile tests/contract
git commit -m "chore: add project tooling and documentation"
```

## Task 6: Final independent verification and handoff

**Files:**

- Inspect all tracked files; modify only a file named by a failing verification
  command if a correction is required.

- [ ] **Step 1: Recreate the locked environment from the repository**

Run:

```bash
uv sync --locked --all-extras --dev
```

Expected: exit code 0, with `uv.lock` unchanged.

- [ ] **Step 2: Run every acceptance command fresh**

Run:

```bash
uv run pytest -q
uv run ruff format --check .
uv run ruff check .
uv run ty check
uv run mkdocs build --strict --site-dir site
uv run pre-commit run --all-files
git diff --check
```

Expected: all commands exit 0. Record the actual test count and command
outputs; do not claim success from an earlier task-level run.

- [ ] **Step 3: Verify repository and remote boundaries**

Run:

```bash
git status --short --branch
git log --oneline --decorate -5
git remote -v
git ls-files | rg '(^|/)(data|models|checkpoints|runs|outputs|tracking)(/|$)|\.parquet$|\.sqlite(3)?$|\.jsonl$' || true
```

Expected: the worktree is clean except for intentionally ignored `site/`
output, tracked files contain no project data artifacts, the branch is
`main`, and `origin` is the supplied GitHub repository. Do not run `git push`.

- [ ] **Step 4: Review the final diff and confirm no verification fix is needed**

Run:

```bash
git diff 39dce72..HEAD --stat
git diff 39dce72..HEAD --check
```

Expected: the diff contains only the planned foundation files and has no
whitespace errors. If a command fails, stop the handoff, fix only the named
file with a focused RED -> GREEN cycle, rerun the complete gate, and create a
focused Conventional Commit for that fix before continuing.

- [ ] **Step 5: Report the safe stop boundary**

Report:

- the absolute repository path;
- the current branch and latest commit SHA;
- the exact local validation commands and observed results;
- that the Seagate data root was not populated by this milestone;
- that no Hugging Face authentication, upload, Trackio remote sync, or
  Grid'5000/OAR operation occurred;
- that the next safe step is to design the landuse dataset/label contract
  before implementing training.
