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
    paths = [ROOT / "README.md", ROOT / "CONTRIBUTING.md"]
    docs_root = ROOT / "docs"
    paths.extend(
        path
        for path in sorted(docs_root.rglob("*.md"))
        if "superpowers" not in path.relative_to(docs_root).parts
    )

    for path in paths:
        content = path.read_text(encoding="utf-8")
        assert "hf_" not in content, path
