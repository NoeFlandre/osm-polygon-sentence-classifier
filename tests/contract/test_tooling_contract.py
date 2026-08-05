import re
from pathlib import Path

import pytest

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


_USES_LINE = re.compile(r"^\s*(?:-\s*)?uses:")
_ACTION_LINE = re.compile(
    r"^\s*(?:-\s*)?uses:\s+(?P<action>[^\s@]+)@"
    r"(?P<sha>[0-9a-f]{40})(?:\s+#.*)?\s*$"
)


def _action_references(content: str) -> list[tuple[str, str]]:
    references = []
    for line in content.splitlines():
        if _USES_LINE.match(line):
            match = _ACTION_LINE.fullmatch(line)
            assert match is not None, f"invalid action reference: {line!r}"
            references.append((match["action"], match["sha"]))
    return references


def _assert_action_pin(content: str, action: str, sha: str) -> None:
    references = _action_references(content)
    assert references.count((action, sha)) == 1


def _assert_workflow_action_references(
    content: str, expected: tuple[tuple[str, str], ...]
) -> None:
    assert sorted(_action_references(content)) == sorted(expected)


def test_action_pin_rejects_malformed_prefix_and_extra_unpinned_reference() -> None:
    sha = "11bd71901bbe5b1630ceea73d27597364c9af683"
    invalid_workflows = (
        f"uses: actions/checkout@{sha}trailing\n",
        f"uses: actions/checkout@{sha}\nuses: example/action@main\n",
    )

    for content in invalid_workflows:
        with pytest.raises(AssertionError):
            _assert_action_pin(content, "actions/checkout", sha)


def test_workflows_use_full_sha_action_pins_and_never_publish() -> None:
    ci = _text(".github/workflows/ci.yml")
    pages = _text(".github/workflows/docs.yml")

    expected_ci = (
        (
            "actions/checkout",
            "11bd71901bbe5b1630ceea73d27597364c9af683",
        ),
        (
            "astral-sh/setup-uv",
            "08807647e7069bb48b6ef5acd8ec9567f424441b",
        ),
    )
    expected_pages = expected_ci + (
        (
            "actions/upload-pages-artifact",
            "56afc609e74202658d3ffba0e8f6dda462b719fa",
        ),
        (
            "actions/deploy-pages",
            "d6db90164ac5ed86f2b6aed7e0febac5b3c0c03e",
        ),
    )

    _assert_workflow_action_references(ci, expected_ci)
    _assert_workflow_action_references(pages, expected_pages)

    for workflow in (ci, pages):
        _assert_action_pin(
            workflow,
            "actions/checkout",
            "11bd71901bbe5b1630ceea73d27597364c9af683",
        )
        _assert_action_pin(
            workflow,
            "astral-sh/setup-uv",
            "08807647e7069bb48b6ef5acd8ec9567f424441b",
        )
    _assert_action_pin(
        pages,
        "actions/upload-pages-artifact",
        "56afc609e74202658d3ffba0e8f6dda462b719fa",
    )
    _assert_action_pin(
        pages,
        "actions/deploy-pages",
        "d6db90164ac5ed86f2b6aed7e0febac5b3c0c03e",
    )

    for command in ("hf upload", "huggingface_hub", "push_to_hub"):
        assert command not in ci + pages


def test_pages_workflow_builds_strict_mkdocs_and_deploys_pages_artifact() -> None:
    content = _text(".github/workflows/docs.yml")
    assert "uv run mkdocs build --strict" in content
    assert "actions/upload-pages-artifact@" in content
    assert "actions/deploy-pages@" in content
    assert "pages: write" in content
    assert "id-token: write" in content
    assert "    permissions:\n      contents: read" in content
    deploy = content.split("\n  deploy:\n", maxsplit=1)[1]
    assert "    permissions:\n      pages: write\n      id-token: write" in deploy
    assert "      contents:" not in deploy


def test_training_dependency_declares_trackio() -> None:
    content = _text("pyproject.toml")
    assert "training = [" in content
    assert "accelerate" in content
    assert "datasets" in content
    assert "trackio" in content
    assert "torch" in content
    assert "transformers" in content


def test_typed_package_declares_and_contains_the_pep561_marker() -> None:
    pyproject = _text("pyproject.toml")
    marker = ROOT / "src/osm_polygon_sentence_classifier/py.typed"

    assert "[tool.setuptools.package-data]" in pyproject
    assert 'osm_polygon_sentence_classifier = ["py.typed"]' in pyproject
    assert marker.is_file()


def test_mkdocs_uses_material_and_excludes_internal_superpowers_docs() -> None:
    content = _text("mkdocs.yml")
    assert "name: material" in content
    assert "superpowers/*" in content
    assert "reference/grid5000-boundary.md" in content


def test_getting_started_lists_just_as_a_prerequisite() -> None:
    content = _text("docs/guides/getting-started.md")
    prerequisites = content.split("## Local quality gates", maxsplit=1)[0]
    assert "- just" in prerequisites
