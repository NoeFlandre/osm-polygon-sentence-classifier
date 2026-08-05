from pathlib import Path
from types import SimpleNamespace

import pytest

from osm_polygon_sentence_classifier.publication import (
    ModelPublicationError,
    publish_model_directory,
)


def _model_directory(tmp_path: Path) -> Path:
    directory = tmp_path / "model"
    directory.mkdir()
    (directory / "config.json").write_text("{}")
    (directory / "model.safetensors").write_bytes(b"weights")
    (directory / "tokenizer.json").write_text("{}")
    return directory


def test_model_publication_commits_only_final_root_files(tmp_path: Path) -> None:
    directory = _model_directory(tmp_path)
    (directory / "checkpoint-10").mkdir()
    (directory / "checkpoint-10" / "model.safetensors").write_bytes(b"old")
    operations: list[dict[str, object]] = []
    commits: list[dict[str, object]] = []

    class FakeHub:
        def create_commit(self, **kwargs: object) -> SimpleNamespace:
            commits.append(kwargs)
            return SimpleNamespace(
                oid="c" * 40,
                commit_url="https://huggingface.co/test/commit/" + "c" * 40,
            )

    def operation_factory(**kwargs: object) -> dict[str, object]:
        operations.append(kwargs)
        return kwargs

    result = publish_model_directory(
        directory,
        "NoeFlandre/osm-polygon-sentence-classifier",
        hub_api=FakeHub(),
        operation_factory=operation_factory,
    )

    assert result.commit_id == "c" * 40
    assert result.files == ("config.json", "model.safetensors", "tokenizer.json")
    assert [item["path_in_repo"] for item in operations] == [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
    ]
    assert commits[0]["repo_type"] == "model"
    assert commits[0]["revision"] == "main"
    assert commits[0]["operations"] == operations


def test_model_publication_rejects_an_incomplete_directory_before_hub_call(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "model"
    directory.mkdir()
    (directory / "config.json").write_text("{}")
    hub_called = False

    class FakeHub:
        def create_commit(self, **kwargs: object) -> None:
            del kwargs
            nonlocal hub_called
            hub_called = True
            return None

    with pytest.raises(ModelPublicationError, match="weights"):
        publish_model_directory(directory, "owner/model", hub_api=FakeHub())

    assert not hub_called


def test_model_publication_rejects_a_non_directory() -> None:
    with pytest.raises(ModelPublicationError, match="directory"):
        publish_model_directory("/tmp/missing-model", "owner/model")
