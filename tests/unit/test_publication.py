from pathlib import Path
from types import SimpleNamespace

import pytest

from osm_polygon_sentence_classifier.checkpointing import write_checkpoint_manifest
from osm_polygon_sentence_classifier.publication import (
    ModelPublicationError,
    ensure_model_repository,
    publish_checkpoint_directory,
    publish_model_directory,
)

CHECKPOINT_IDENTITY = {"run_id": "a" * 20, "model_revision": "b" * 40}


def _model_directory(tmp_path: Path) -> Path:
    directory = tmp_path / "model"
    directory.mkdir()
    (directory / "config.json").write_text("{}")
    (directory / "model.safetensors").write_bytes(b"weights")
    (directory / "tokenizer.json").write_text("{}")
    return directory


def _checkpoint_directory(tmp_path: Path, step: int = 10) -> Path:
    directory = tmp_path / f"checkpoint-{step}"
    directory.mkdir()
    for filename in (
        "model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    ):
        (directory / filename).write_bytes(b"checkpoint")
    (directory / "trainer_state.json").write_text(
        f'{{"global_step": {step}}}', encoding="utf-8"
    )
    write_checkpoint_manifest(
        directory,
        identity=CHECKPOINT_IDENTITY,
        global_step=step,
    )
    return directory


def test_model_repository_setup_is_idempotent() -> None:
    calls: list[dict[str, object]] = []

    class FakeHub:
        def create_repo(self, **kwargs: object) -> None:
            calls.append(kwargs)

    ensure_model_repository(
        "NoeFlandre/osm-polygon-sentence-classifier",
        hub_api=FakeHub(),
    )

    assert calls == [
        {
            "repo_id": "NoeFlandre/osm-polygon-sentence-classifier",
            "repo_type": "model",
            "exist_ok": True,
        }
    ]


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


def test_checkpoint_publication_uploads_a_complete_latest_snapshot(
    tmp_path: Path,
) -> None:
    directory = _checkpoint_directory(tmp_path)
    (directory / ".env").write_text("HF_TOKEN=should-not-upload", encoding="utf-8")
    operations: list[dict[str, object]] = []
    commits: list[dict[str, object]] = []

    class FakeHub:
        def create_commit(self, **kwargs: object) -> SimpleNamespace:
            commits.append(kwargs)
            return SimpleNamespace(
                oid="e" * 40,
                commit_url="https://huggingface.co/test/commit/" + "e" * 40,
            )

    def operation_factory(**kwargs: object) -> dict[str, object]:
        operations.append(kwargs)
        return kwargs

    result = publish_checkpoint_directory(
        directory,
        "NoeFlandre/osm-polygon-sentence-classifier",
        identity=CHECKPOINT_IDENTITY,
        hub_api=FakeHub(),
        operation_factory=operation_factory,
    )

    assert result.commit_id == "e" * 40
    assert result.files == tuple(
        f"checkpoints/last-checkpoint/{name}"
        for name in (
            "checkpoint-manifest.json",
            "model.safetensors",
            "optimizer.pt",
            "rng_state.pth",
            "scheduler.pt",
            "trainer_state.json",
        )
    )
    assert [item["path_in_repo"] for item in operations] == list(result.files)
    assert commits[0]["commit_message"] == "Publish checkpoint checkpoint-10"


def test_checkpoint_publication_rejects_an_incomplete_snapshot_before_hub_call(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "checkpoint-10"
    directory.mkdir()
    (directory / "trainer_state.json").write_text(
        '{"global_step": 10}', encoding="utf-8"
    )
    hub_called = False

    class FakeHub:
        def create_commit(self, **kwargs: object) -> None:
            del kwargs
            nonlocal hub_called
            hub_called = True

    with pytest.raises(ModelPublicationError, match="complete checkpoint"):
        publish_checkpoint_directory(
            directory,
            "owner/model",
            identity=CHECKPOINT_IDENTITY,
            hub_api=FakeHub(),
        )

    assert not hub_called


def test_checkpoint_publication_accepts_a_complete_checkpoint_that_is_not_latest(
    tmp_path: Path,
) -> None:
    directory = _checkpoint_directory(tmp_path, step=10)
    _checkpoint_directory(tmp_path, step=20)
    commits: list[dict[str, object]] = []

    class FakeHub:
        def create_commit(self, **kwargs: object) -> SimpleNamespace:
            commits.append(kwargs)
            return SimpleNamespace(
                oid="f" * 40,
                commit_url="https://huggingface.co/test/commit/" + "f" * 40,
            )

    result = publish_checkpoint_directory(
        directory,
        "owner/model",
        identity=CHECKPOINT_IDENTITY,
        hub_api=FakeHub(),
    )

    assert result.commit_id == "f" * 40
    assert commits
