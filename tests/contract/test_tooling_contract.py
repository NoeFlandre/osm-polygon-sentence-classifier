from pathlib import Path

ROOT = Path(__file__).parents[2]


def _text(relative_path: str) -> str:
    path = ROOT / relative_path
    return path.read_text(encoding="utf-8") if path.exists() else ""


def test_justfile_exposes_the_documented_quality_recipes() -> None:
    content = _text("justfile")
    for recipe in (
        "format",
        "format-check",
        "lint",
        "typecheck",
        "test",
        "docs",
        "check",
    ):
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
