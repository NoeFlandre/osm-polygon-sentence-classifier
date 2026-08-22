from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from osm_polygon_sentence_classifier import publication
from osm_polygon_sentence_classifier.checkpointing import write_checkpoint_manifest
from osm_polygon_sentence_classifier.publication import (
    ModelPublicationError,
    _commit_publication,
    ensure_model_repository,
    publish_checkpoint_directory,
    publish_model_directory,
    publish_study_documents,
    render_model_card,
    render_repository_readme,
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


def test_commit_publication_returns_the_validated_commit_result() -> None:
    calls: list[dict[str, object]] = []

    class FakeHub:
        def create_commit(self, **kwargs: object) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace(
                oid="c" * 40,
                commit_url="https://huggingface.co/test/commit/" + "c" * 40,
            )

    result = _commit_publication(
        api=FakeHub(),
        repository="owner/model",
        operations=["operation"],
        commit_message="Publish artifacts",
        published_paths=["artifacts/model.safetensors"],
        failure_message="publication failed",
    )

    assert result.repository_id == "owner/model"
    assert result.commit_id == "c" * 40
    assert result.commit_url.endswith("/" + "c" * 40)
    assert result.files == ("artifacts/model.safetensors",)
    assert calls == [
        {
            "repo_id": "owner/model",
            "repo_type": "model",
            "operations": ["operation"],
            "commit_message": "Publish artifacts",
            "revision": "main",
        }
    ]


def test_publication_declares_the_typed_hub_dependency_boundary() -> None:
    protocol = getattr(publication, "_HubApiProtocol", None)
    assert protocol is not None
    assert callable(getattr(protocol, "create_repo", None))
    assert callable(getattr(protocol, "create_commit", None))


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
            "private": False,
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


def test_model_publication_rejects_a_symlinked_root_file_before_hub_call(
    tmp_path: Path,
) -> None:
    directory = _model_directory(tmp_path)
    target = tmp_path / "outside-model.safetensors"
    target.write_bytes(b"weights")
    (directory / "model.safetensors").unlink()
    (directory / "model.safetensors").symlink_to(target)
    hub_called = False

    class FakeHub:
        def create_commit(self, **kwargs: object) -> None:
            del kwargs
            nonlocal hub_called
            hub_called = True

    with pytest.raises(ModelPublicationError) as error:
        publish_model_directory(directory, "owner/model", hub_api=FakeHub())

    assert str(error.value) == "model output cannot contain symlinks"
    assert not hub_called


def test_model_publication_groups_final_files_by_experiment_and_run(
    tmp_path: Path,
) -> None:
    directory = _model_directory(tmp_path)
    operations: list[dict[str, object]] = []

    class FakeHub:
        def create_commit(self, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                oid="c" * 40,
                commit_url="https://huggingface.co/test/commit/" + "c" * 40,
            )

    def operation_factory(**kwargs: object) -> dict[str, object]:
        operations.append(kwargs)
        return kwargs

    identity = {
        "run_id": "r" * 20,
        "task_name": "landuse",
        "training_config": {"run_name": "landuse-baseline"},
    }

    result = publish_model_directory(
        directory,
        "owner/model",
        identity=identity,
        repository_readme="# repository guide\n",
        hub_api=FakeHub(),
        operation_factory=operation_factory,
    )

    assert result.files == (
        "README.md",
        "experiments/landuse-baseline/run-rrrrrrrrrrrrrrrrrrrr/final/config.json",
        "experiments/landuse-baseline/run-rrrrrrrrrrrrrrrrrrrr/final/model.safetensors",
        "experiments/landuse-baseline/run-rrrrrrrrrrrrrrrrrrrr/final/tokenizer.json",
    )
    assert [item["path_in_repo"] for item in operations] == list(result.files)
    assert operations[0]["path_or_fileobj"] == b"# repository guide\n"


def test_model_publication_uses_the_explicit_study_namespace(
    tmp_path: Path,
) -> None:
    directory = _model_directory(tmp_path)
    operations: list[dict[str, object]] = []

    class FakeHub:
        def create_commit(self, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                oid="c" * 40,
                commit_url="https://huggingface.co/test/commit/" + "c" * 40,
            )

    identity = {
        "run_id": "r" * 20,
        "training_config": {
            "run_name": "ablation-run",
            "artifact_namespace": "studies/landuse-v1/a01-head-128",
        },
    }

    result = publish_model_directory(
        directory,
        "owner/model",
        identity=identity,
        hub_api=FakeHub(),
        operation_factory=lambda **kwargs: operations.append(kwargs) or kwargs,
    )

    assert result.files[0].startswith(
        "studies/landuse-v1/a01-head-128/run-rrrrrrrrrrrrrrrrrrrr/final/"
    )


def test_study_documents_are_committed_only_under_declared_public_paths() -> None:
    operations: list[dict[str, object]] = []

    class FakeHub:
        def create_commit(self, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                oid="d" * 40,
                commit_url="https://huggingface.co/test/commit/" + "d" * 40,
            )

    result = publish_study_documents(
        "owner/model",
        {
            "README.md": "# study\n",
            "studies/landuse-v1/study.json": "{}\n",
        },
        hub_api=FakeHub(),
        operation_factory=lambda **kwargs: operations.append(kwargs) or kwargs,
    )

    assert result.files == ("README.md", "studies/landuse-v1/study.json")
    assert all(".." not in str(operation["path_in_repo"]) for operation in operations)


def test_model_publication_accepts_a_one_line_repository_readme(
    tmp_path: Path,
) -> None:
    directory = _model_directory(tmp_path)
    operations: list[dict[str, object]] = []

    class FakeHub:
        def create_commit(self, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                oid="c" * 40,
                commit_url="https://huggingface.co/test/commit/" + "c" * 40,
            )

    result = publish_model_directory(
        directory,
        "owner/model",
        identity={
            "run_id": "r" * 20,
            "training_config": {"run_name": "landuse"},
        },
        repository_readme="# guide",
        hub_api=FakeHub(),
        operation_factory=lambda **kwargs: operations.append(kwargs) or kwargs,
    )

    assert result.files[0] == "README.md"
    assert operations[0]["path_or_fileobj"] == b"# guide"


def test_model_publication_includes_the_generated_model_card(tmp_path: Path) -> None:
    directory = _model_directory(tmp_path)
    (directory / "README.md").write_text("generated card", encoding="utf-8")
    operations: list[dict[str, object]] = []

    class FakeHub:
        def create_commit(self, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                oid="c" * 40,
                commit_url="https://huggingface.co/test/commit/" + "c" * 40,
            )

    def operation_factory(**kwargs: object) -> dict[str, object]:
        operations.append(kwargs)
        return kwargs

    result = publish_model_directory(
        directory,
        "owner/model",
        hub_api=FakeHub(),
        operation_factory=operation_factory,
    )

    assert "README.md" in result.files
    assert "README.md" in [item["path_in_repo"] for item in operations]


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


def test_publication_model_output_rejects_a_regular_file(tmp_path: Path) -> None:
    output = tmp_path / "model"
    output.write_text("not a directory", encoding="utf-8")

    with pytest.raises(ModelPublicationError) as error:
        publication._model_output_files(output)

    assert str(error.value) == "model output must be a real directory"


def test_publication_model_directory_entries_wraps_read_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cause = OSError("cannot read")

    def failing_iterdir(path: Path) -> object:
        del path
        raise cause

    monkeypatch.setattr(Path, "iterdir", failing_iterdir)

    with pytest.raises(ModelPublicationError) as error:
        publication._model_directory_entries(tmp_path)

    assert str(error.value) == "model output cannot be read"
    assert error.value.__cause__ is cause


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
        f"experiments/landuse/run-aaaaaaaaaaaaaaaaaaaa/checkpoints/step-10/{name}"
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


def test_checkpoint_publication_includes_the_generated_model_card(
    tmp_path: Path,
) -> None:
    directory = _checkpoint_directory(tmp_path)
    (directory / "README.md").write_text("checkpoint card", encoding="utf-8")
    operations: list[dict[str, object]] = []

    class FakeHub:
        def create_commit(self, **kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(
                oid="e" * 40,
                commit_url="https://huggingface.co/test/commit/" + "e" * 40,
            )

    def operation_factory(**kwargs: object) -> dict[str, object]:
        operations.append(kwargs)
        return kwargs

    result = publish_checkpoint_directory(
        directory,
        "owner/model",
        identity=CHECKPOINT_IDENTITY,
        hub_api=FakeHub(),
        operation_factory=operation_factory,
    )

    assert (
        "experiments/landuse/run-aaaaaaaaaaaaaaaaaaaa/checkpoints/step-10/README.md"
        in result.files
    )
    assert (
        "experiments/landuse/run-aaaaaaaaaaaaaaaaaaaa/checkpoints/step-10/README.md"
        in [item["path_in_repo"] for item in operations]
    )
    assert all(
        str(item["path_in_repo"]).startswith(
            "experiments/landuse/run-aaaaaaaaaaaaaaaaaaaa/checkpoints/step-10/"
        )
        for item in operations
    )


def test_checkpoint_publication_keeps_different_steps_in_different_directories(
    tmp_path: Path,
) -> None:
    published_paths: list[str] = []

    class FakeHub:
        def create_commit(self, **kwargs: object) -> SimpleNamespace:
            raw_operations = kwargs.get("operations")
            if not isinstance(raw_operations, list):
                raise AssertionError("operations were not supplied")
            for operation in raw_operations:
                if isinstance(operation, dict):
                    published_paths.append(str(operation.get("path_in_repo")))
            return SimpleNamespace(
                oid="e" * 40,
                commit_url="https://huggingface.co/test/commit/" + "e" * 40,
            )

    for step in (10, 20):
        publish_checkpoint_directory(
            _checkpoint_directory(tmp_path, step=step),
            "owner/model",
            identity=CHECKPOINT_IDENTITY,
            hub_api=FakeHub(),
            operation_factory=lambda **kwargs: kwargs,
        )

    assert any(
        path.startswith(
            "experiments/landuse/run-aaaaaaaaaaaaaaaaaaaa/checkpoints/step-10/"
        )
        for path in published_paths
    )
    assert any(
        path.startswith(
            "experiments/landuse/run-aaaaaaaaaaaaaaaaaaaa/checkpoints/step-20/"
        )
        for path in published_paths
    )


def test_model_card_contains_only_safe_training_metadata() -> None:
    card = render_model_card(
        identity={
            "task_name": "landuse",
            "source_commit": "a" * 40,
            "dataset_revision": "b" * 40,
            "model_name_or_path": "test-model",
            "model_revision": "c" * 40,
            "training_config": {
                "max_steps": 1000,
                "learning_rate": 0.0003,
                "HF_TOKEN": "must-not-appear",
            },
        },
        training_metrics={"eval_loss": 0.25, "train/global_step": 100},
        checkpoint_step=100,
        trackio_space_id="owner/trackio-live",
    )

    assert "landuse" in card
    assert "b" * 40 in card
    assert "test-model" in card
    assert '"max_steps": 1000' in card
    assert "checkpoint at step 100" in card
    assert "https://huggingface.co/spaces/owner/trackio-live" in card
    assert "static snapshots" in card
    assert "HF_TOKEN" not in card
    assert "must-not-appear" not in card


def test_worldwide_v2_model_card_describes_place_relevance_not_landuse() -> None:
    card = render_model_card(
        identity={
            "task_name": "place-relevance-v2",
            "dataset_revision": "b" * 40,
            "model_name_or_path": "jhu-clsp/mmBERT-small",
            "model_revision": "c" * 40,
            "training_config": {"trainable_layers": "head"},
        }
    )

    assert "place relevance" in card
    assert "- place-relevance" in card
    assert "landuse" not in card


def test_repository_readme_documents_the_organized_public_layout() -> None:
    readme = render_repository_readme(
        identity={
            "run_id": "a" * 20,
            "task_name": "landuse",
            "model_name_or_path": "test-model",
            "model_revision": "b" * 40,
            "training_config": {"run_name": "baseline"},
        },
        trackio_space_id="owner/trackio",
    )

    assert "experiments/baseline/run-aaaaaaaaaaaaaaaaaaaa/final/" in readme
    assert "experiments/baseline/run-aaaaaaaaaaaaaaaaaaaa/checkpoints/step-N/" in readme
    assert "no model files are stored at the repository root" in readme
    assert "studies/landuse-v1/README.md" in readme
    assert "https://huggingface.co/spaces/owner/trackio" in readme


def test_repository_readme_registers_the_worldwide_v2_study() -> None:
    readme = render_repository_readme(
        identity={
            "task_name": "place-relevance-v2",
            "run_id": "a" * 20,
            "model_name_or_path": "jhu-clsp/mmBERT-small",
            "model_revision": "b" * 40,
            "training_config": {
                "run_name": "place-relevance-v2|baseline|seed-42",
                "artifact_namespace": "studies/place-relevance-v2/baseline",
            },
        }
    )

    assert "studies/place-relevance-v2/README.md" in readme
    assert "place-relevance-v2" in readme


def test_repository_readme_registers_separate_v1_and_v2_trackio_dashboards() -> None:
    readme = render_repository_readme(
        identity={
            "task_name": "place-relevance-v2",
            "run_id": "a" * 20,
            "model_name_or_path": "jhu-clsp/mmBERT-small",
            "model_revision": "b" * 40,
            "training_config": {
                "run_name": "place-relevance-v2|baseline|seed-42",
                "artifact_namespace": "studies/place-relevance-v2/baseline",
            },
        }
    )

    assert (
        "https://huggingface.co/spaces/NoeFlandre/"
        "osm-polygon-sentence-classifier-trackio"
    ) in readme
    assert (
        "https://huggingface.co/spaces/NoeFlandre/"
        "osm-polygon-sentence-classifier-v2-trackio"
    ) in readme
    assert "V1 landuse" in readme
    assert "V2 place relevance" in readme


def test_model_card_renders_every_public_provenance_field() -> None:
    card = render_model_card(
        identity={
            "task_name": "landuse",
            "model_name_or_path": "jhu-clsp/mmBERT-small",
            "model_revision": "b" * 40,
            "dataset_revision": "d" * 40,
            "source_commit": "s" * 40,
            "training_config": {
                "artifact_namespace": "studies/landuse-v1",
                "learning_rate": 0.0003,
                "max_steps": 100,
                "run_name": "landuse-baseline",
                "seed": 42,
            },
        },
        training_metrics={"accuracy": 0.8, "f1": 0.75},
        checkpoint_step=100,
        trackio_space_id="owner/trackio",
    )

    for expected in (
        "# OSM Polygon Landuse Sentence Classifier",
        "relevant to landuse",
        "checkpoint at step 100",
        "Dataset revision: `dddddddddddddddddddddddddddddddddddddddd`",
        "- Task: `landuse`",
        "- Labels: `no` (0), `yes` (1)",
        "- Base model: `jhu-clsp/mmBERT-small`",
        "- Base-model revision: `bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb`",
        "- Source-code commit: `ssssssssssssssssssssssssssssssssssssssss`",
        '"artifact_namespace": "studies/landuse-v1"',
        '"learning_rate": 0.0003',
        '"max_steps": 100',
        '"run_name": "landuse-baseline"',
        '"seed": 42',
        '"accuracy": 0.8',
        '"f1": 0.75',
        "https://huggingface.co/spaces/owner/trackio",
        "Metrics are published as static snapshots after complete checkpoints",
    ):
        assert expected in card

    assert card.splitlines() == [
        "---",
        "library_name: transformers",
        "pipeline_tag: text-classification",
        "tags:",
        "- landuse",
        "- text-classification",
        "---",
        "",
        "# OSM Polygon Landuse Sentence Classifier",
        "",
        "This model classifies whether a sentence is relevant to landuse. "
        "The recorded training status is **checkpoint at step 100**.",
        "",
        "## Training data",
        "",
        "- Dataset: [NoeFlandre/osm-polygon-wikidata-sentence-relevance]"
        "(https://huggingface.co/datasets/"
        "NoeFlandre/osm-polygon-wikidata-sentence-relevance)",
        "- Dataset revision: `dddddddddddddddddddddddddddddddddddddddd`",
        "- Task: `landuse`",
        "- Labels: `no` (0), `yes` (1)",
        "",
        "## Model and provenance",
        "",
        "- Base model: `jhu-clsp/mmBERT-small`",
        "- Base-model revision: `bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb`",
        "- Source-code commit: `ssssssssssssssssssssssssssssssssssssssss`",
        "- Model repository: [NoeFlandre/osm-polygon-sentence-classifier]"
        "(https://huggingface.co/NoeFlandre/osm-polygon-sentence-classifier)",
        "",
        "## Training configuration",
        "",
        "```json",
        "{",
        '  "artifact_namespace": "studies/landuse-v1",',
        '  "learning_rate": 0.0003,',
        '  "max_steps": 100,',
        '  "run_name": "landuse-baseline",',
        '  "seed": 42',
        "}",
        "```",
        "",
        "## Metrics",
        "",
        "```json",
        "{",
        '  "accuracy": 0.8,',
        '  "f1": 0.75',
        "}",
        "```",
        "",
        "## Experiment tracking",
        "",
        "[Open the Trackio dashboard](https://huggingface.co/spaces/"
        "owner/trackio). Metrics are published as static snapshots after "
        "complete checkpoints and final publication.",
    ]


def test_model_card_uses_safe_defaults_for_an_unpopulated_run() -> None:
    card = render_model_card(identity={})

    for expected in (
        "# OSM Polygon Landuse Sentence Classifier",
        "final model",
        "Dataset revision: `not recorded`",
        "- Task: `landuse`",
        "- Base model: `not recorded`",
        "- Base-model revision: `not pinned`",
        "- Source-code commit: `not recorded`",
        "{}",
        "Trackio was not enabled for this run.",
    ):
        assert expected in card


def test_repository_readme_renders_the_run_layout_and_tracking_links() -> None:
    readme = render_repository_readme(
        identity={
            "task_name": "landuse",
            "model_name_or_path": "jhu-clsp/mmBERT-small",
            "model_revision": "b" * 40,
            "run_id": "r" * 20,
            "training_config": {
                "artifact_namespace": "studies/landuse-v1",
                "run_name": "landuse-baseline",
            },
        },
        trackio_space_id="owner/trackio",
    )

    for expected in (
        "- landuse\n- text-classification",
        "studies/landuse-v1/run-rrrrrrrrrrrrrrrrrrrr/final/",
        "studies/landuse-v1/run-rrrrrrrrrrrrrrrrrrrr/checkpoints/step-N/",
        "no model files are stored at the repository root",
        "studies/landuse-v1/README.md",
        "studies/place-relevance-v2/README.md",
        "studies/place-relevance-v2-ablations/README.md",
        "NoeFlandre/osm-polygon-sentence-classifier-trackio-data",
        "NoeFlandre/osm-polygon-sentence-classifier-v2-trackio-data",
        "- Task: `landuse`",
        "- Task description: landuse",
        "- Base model: `jhu-clsp/mmBERT-small`",
        "- Base-model revision: `bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb`",
        "- Current run Trackio dashboard: [owner/trackio]",
        "The generated model card inside the final directory",
    ):
        assert expected in readme

    assert readme.splitlines()[:7] == [
        "---",
        "library_name: transformers",
        "pipeline_tag: text-classification",
        "tags:",
        "- landuse",
        "- text-classification",
        "---",
    ]
    assert readme.endswith(
        "The generated model card inside the final directory contains the "
        "recorded configuration and evaluation metrics."
    )
    assert readme.splitlines() == [
        "---",
        "library_name: transformers",
        "pipeline_tag: text-classification",
        "tags:",
        "- landuse",
        "- text-classification",
        "---",
        "",
        "# OSM Polygon Sentence Classifier",
        "",
        "This public repository contains organized, immutable outputs for the "
        "OSM polygon sentence-classification studies.",
        "",
        "## Repository layout",
        "",
        "- Final model: `studies/landuse-v1/run-rrrrrrrrrrrrrrrrrrrr/final/` "
        "(load with the Transformers `subfolder` argument).",
        "- Complete checkpoints: "
        "`studies/landuse-v1/run-rrrrrrrrrrrrrrrrrrrr/checkpoints/step-N/`.",
        "- Each run has its own experiment and run directory; no model files "
        "are stored at the repository root.",
        "",
        "## Study registry",
        "",
        "- Completed landuse ablations: "
        "[`studies/landuse-v1/README.md`](studies/landuse-v1/README.md).",
        "- Protocol: [`studies/landuse-v1/study.json`](studies/landuse-v1/study.json).",
        "- Results: [`studies/landuse-v1/results.json`]"
        "(studies/landuse-v1/results.json).",
        "",
        "- Worldwide V2 baseline: "
        "[`studies/place-relevance-v2/README.md`]"
        "(studies/place-relevance-v2/README.md).",
        "- Protocol: [`studies/place-relevance-v2/study.json`]"
        "(studies/place-relevance-v2/study.json).",
        "- Results: [`studies/place-relevance-v2/results.json`]"
        "(studies/place-relevance-v2/results.json).",
        "",
        "- Worldwide V2 ablations: "
        "[`studies/place-relevance-v2-ablations/README.md`]"
        "(studies/place-relevance-v2-ablations/README.md).",
        "- Ablation protocol: "
        "[`studies/place-relevance-v2-ablations/study.json`]"
        "(studies/place-relevance-v2-ablations/study.json).",
        "- Ablation results: "
        "[`studies/place-relevance-v2-ablations/results.json`]"
        "(studies/place-relevance-v2-ablations/results.json).",
        "",
        "## Experiment tracking",
        "",
        "- V1 landuse: [NoeFlandre/osm-polygon-sentence-classifier-trackio]"
        "(https://huggingface.co/spaces/"
        "NoeFlandre/osm-polygon-sentence-classifier-trackio).",
        "  Static data bucket: "
        "`NoeFlandre/osm-polygon-sentence-classifier-trackio-data`.",
        "- V2 place relevance: "
        "[NoeFlandre/osm-polygon-sentence-classifier-v2-trackio]"
        "(https://huggingface.co/spaces/"
        "NoeFlandre/osm-polygon-sentence-classifier-v2-trackio).",
        "  Static data bucket: "
        "`NoeFlandre/osm-polygon-sentence-classifier-v2-trackio-data`.",
        "",
        "## Training identity",
        "",
        "- Task: `landuse`",
        "- Task description: landuse",
        "- Base model: `jhu-clsp/mmBERT-small`",
        "- Base-model revision: `bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb`",
        "- Run directory: `studies/landuse-v1/run-rrrrrrrrrrrrrrrrrrrr`",
        "- Current run Trackio dashboard: [owner/trackio]"
        "(https://huggingface.co/spaces/owner/trackio)",
        "",
        "The generated model card inside the final directory contains the "
        "recorded configuration and evaluation metrics.",
    ]


def test_repository_readme_uses_safe_defaults_without_current_tracking() -> None:
    readme = render_repository_readme(identity={})

    for expected in (
        "- Final model: `experiments/landuse/run-",
        "- Task: `landuse`",
        "- Task description: landuse",
        "- Base model: `not recorded`",
        "- Base-model revision: `not pinned`",
    ):
        assert expected in readme
    assert "- Current run Trackio dashboard" not in readme
    assert "XXXX" not in readme


@pytest.mark.parametrize(
    "value",
    [None, 1, "", "   ", "line\nbreak", "line\rbreak"],
)
def test_publication_rejects_non_single_line_repository_values(value: object) -> None:
    with pytest.raises(ModelPublicationError, match="repository_id"):
        publication._require_non_blank(value, "repository_id")


def test_publication_preserves_a_valid_repository_value() -> None:
    assert publication._require_non_blank(" owner/model ", "repository_id") == (
        " owner/model "
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, True),
        (False, True),
        (3, True),
        ("value", True),
        (0.5, True),
        (float("nan"), False),
        (float("inf"), False),
        ([], False),
    ],
)
def test_publication_safe_scalar_contract(value: object, expected: bool) -> None:
    assert publication._safe_scalar(value) is expected


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        ("accuracy", 0.8, ("accuracy", 0.8)),
        ("HF_TOKEN", "secret", None),
        ("password_hash", "secret", None),
        ("nested", {"value": 1}, None),
        (1, "value", None),
        ("loss", float("nan"), None),
    ],
)
def test_publication_metadata_filter_is_credential_safe(
    key: object,
    value: object,
    expected: tuple[str, object] | None,
) -> None:
    assert publication._safe_metadata_item(key, value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("  value  ", "value"),
        ("", "fallback"),
        (None, "fallback"),
        ("line\nbreak", "fallback"),
        ("line\rbreak", "fallback"),
    ],
)
def test_publication_safe_line_contract(value: object, expected: str) -> None:
    assert publication._safe_line(value, "fallback") == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("  Hello world / v2...  ", "Hello-world-v2"),
        ("XhelloX", "XhelloX"),
        ("...---...", "fallback"),
        ("", "fallback"),
        (None, "fallback"),
        ("a" * 100, "a" * 80),
    ],
)
def test_publication_slug_contract(value: object, expected: str) -> None:
    assert publication._slug(value, "fallback") == expected


@pytest.mark.parametrize(
    ("identity", "expected"),
    [
        ({}, ("landuse", "landuse", "landuse")),
        (
            {"task_name": "place-relevance-v2"},
            ("place-relevance-v2", "place relevance", "place-relevance"),
        ),
        ({"task_name": "custom-task"}, ("custom-task", "custom task", "custom-task")),
        ({"task_name": "..."}, ("...", "...", "task")),
    ],
)
def test_publication_task_metadata_contract(
    identity: dict[str, object],
    expected: tuple[str, str, str],
) -> None:
    assert publication._task_metadata(identity) == expected


def test_publication_run_id_accepts_valid_ids_and_hashes_invalid_ids() -> None:
    assert publication._publication_run_id({"run_id": "r" * 20}) == "r" * 20
    assert publication._publication_run_id({"run_id": "short"}) == (
        "17ff69dec54f6d77682c"
    )
    assert publication._publication_run_id({"run_id": "bad\nvalue"}) == (
        "0c55fa553830442147fa"
    )


def test_publication_run_id_uses_strict_canonical_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    original_dumps = cast(Any, publication.json.dumps)

    def dumps(value: object, *args: object, **kwargs: object) -> str:
        observed.update(kwargs)
        return original_dumps(value, *args, **kwargs)

    monkeypatch.setattr(publication.json, "dumps", dumps)

    assert publication._publication_run_id({"z": "é", "a": 1}) == (
        "160c52d506c747530bfe"
    )
    assert observed == {
        "allow_nan": False,
        "ensure_ascii": False,
        "separators": (",", ":"),
        "sort_keys": True,
    }


def test_publication_run_id_rejects_non_finite_identity_values() -> None:
    with pytest.raises(
        ValueError,
        match=r"\AOut of range float values are not JSON compliant: nan\Z",
    ):
        publication._publication_run_id({"run_id": "short", "metric": float("nan")})


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (" studies/landuse-v1/ ", " studies/landuse-v1/ "),
        ("Xstudies/X", "Xstudies/X"),
        ("", None),
        (None, None),
        ("studies//landuse", None),
        ("studies/./landuse", None),
        ("studies/../landuse", None),
    ],
)
def test_publication_artifact_namespace_contract(
    value: object,
    expected: str | None,
) -> None:
    assert publication._valid_artifact_namespace(value) == expected


def test_publication_study_path_parts_reject_each_unsafe_component() -> None:
    assert (
        publication._safe_study_path_parts(
            cast(
                Any,
                SimpleNamespace(is_absolute=lambda: False, parts=("a", ".", "b")),
            )
        )
        is False
    )
    assert (
        publication._safe_study_path_parts(
            cast(
                Any,
                SimpleNamespace(is_absolute=lambda: False, parts=("a", "..", "b")),
            )
        )
        is False
    )
    assert (
        publication._safe_study_path_parts(
            cast(
                Any,
                SimpleNamespace(is_absolute=lambda: False, parts=("a", "", "b")),
            )
        )
        is False
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, False), (Path("README.md"), False), ("safe/path", True), ("a\rb", False)],
)
def test_publication_study_path_text_contract(value: object, expected: bool) -> None:
    assert publication._safe_study_path_text(value) is expected


@pytest.mark.parametrize(
    ("step", "expected"),
    [
        (0, "checkpoint at step 0"),
        (12, "checkpoint at step 12"),
        (False, "final model"),
        (-1, "final model"),
        (None, "final model"),
    ],
)
def test_publication_progress_contract(step: object, expected: str) -> None:
    assert publication._progress_text(cast(Any, step)) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (" owner/trackio ", "https://huggingface.co/spaces/owner/trackio"),
        ("", None),
        (None, None),
        ("owner/trackio\nother", None),
        ("owner/trackio\rother", None),
    ],
)
def test_publication_trackio_link_contract(value: object, expected: str | None) -> None:
    assert publication._trackio_link(value) == expected


def test_publication_tracking_section_distinguishes_enabled_and_disabled_runs() -> None:
    assert publication._tracking_section(None) == (
        "Trackio was not enabled for this run."
    )
    assert publication._tracking_section("owner/trackio") == (
        "[Open the Trackio dashboard](https://huggingface.co/spaces/owner/trackio). "
        "Metrics are published as static snapshots after complete checkpoints "
        "and final publication."
    )


@pytest.mark.parametrize(
    "filename",
    [
        "tokenizer.json",
        "tokenizer_config.json",
        "spiece.model",
        "sentencepiece.bpe.model",
        "tokenizer.model",
        "vocab.json",
        "vocab.txt",
    ],
)
def test_publication_accepts_each_supported_tokenizer_file(filename: str) -> None:
    assert publication._has_tokenizer_files({filename}) is True


def test_publication_rejects_a_set_without_tokenizer_files() -> None:
    assert (
        publication._has_tokenizer_files({"config.json", "model.safetensors"}) is False
    )


@pytest.mark.parametrize(
    ("files", "message"),
    [
        (
            [Path("model.safetensors"), Path("tokenizer.json")],
            "model output is missing config.json",
        ),
        (
            [Path("config.json"), Path("tokenizer.json")],
            "model output is missing model weights",
        ),
        (
            [Path("config.json"), Path("model.safetensors")],
            "model output is missing tokenizer files",
        ),
    ],
)
def test_publication_validates_all_required_model_outputs(
    files: list[Path],
    message: str,
) -> None:
    with pytest.raises(ModelPublicationError) as error:
        publication._validate_model_output_files(files)
    assert str(error.value) == message


def test_publication_accepts_a_configured_model_output() -> None:
    publication._validate_model_output_files(
        [Path("config.json"), Path("model.safetensors"), Path("tokenizer.json")]
    )


def test_publication_rejects_a_set_without_publishable_files() -> None:
    with pytest.raises(ModelPublicationError) as error:
        publication._publishable_model_files([Path(".env")])
    assert str(error.value) == "model output contains no publishable files"


@pytest.mark.parametrize(
    ("info", "message"),
    [
        (
            SimpleNamespace(commit_url="https://example.test"),
            "Hugging Face returned an invalid model commit ID",
        ),
        (
            SimpleNamespace(oid="c" * 40),
            "Hugging Face returned an invalid model commit URL",
        ),
        (
            SimpleNamespace(oid="", commit_url="https://example.test"),
            "Hugging Face returned an invalid model commit ID",
        ),
        (
            SimpleNamespace(oid="c" * 40, commit_url=""),
            "Hugging Face returned an invalid model commit URL",
        ),
    ],
)
def test_publication_rejects_incomplete_hub_commit_facts(
    info: object,
    message: str,
) -> None:
    with pytest.raises(ModelPublicationError) as error:
        publication._commit_facts(info)
    assert str(error.value) == message


def test_publication_preserves_valid_hub_commit_facts() -> None:
    assert publication._commit_facts(
        SimpleNamespace(oid="c" * 40, commit_url=" https://example.test ")
    ) == ("c" * 40, " https://example.test ")


def test_publication_repository_readme_operation_is_optional_and_utf8_safe() -> None:
    calls: list[dict[str, object]] = []

    def factory(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return kwargs

    assert publication._repository_readme_operation(None, factory) is None
    operation = publication._repository_readme_operation("café", factory)

    assert operation is not None
    assert operation[1] == "README.md"
    assert calls == [{"path_in_repo": "README.md", "path_or_fileobj": "café".encode()}]


def test_publication_model_operations_keep_each_operation_and_source_path() -> None:
    calls: list[dict[str, object]] = []

    def factory(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return kwargs

    operations, paths = publication._model_operations(
        [Path("/tmp/config.json"), Path("/tmp/model.safetensors")],
        factory=factory,
        artifact_prefix="experiments/study/run-abc",
        repository_readme="# guide",
    )

    assert operations == calls
    assert paths == [
        "README.md",
        "experiments/study/run-abc/final/config.json",
        "experiments/study/run-abc/final/model.safetensors",
    ]
    assert calls == [
        {"path_in_repo": "README.md", "path_or_fileobj": b"# guide"},
        {
            "path_in_repo": "experiments/study/run-abc/final/config.json",
            "path_or_fileobj": "/tmp/config.json",
        },
        {
            "path_in_repo": "experiments/study/run-abc/final/model.safetensors",
            "path_or_fileobj": "/tmp/model.safetensors",
        },
    ]


@pytest.mark.parametrize("content", ["", "   "])
def test_publication_rejects_an_empty_repository_readme(content: object) -> None:
    with pytest.raises(ModelPublicationError) as error:
        publication._repository_readme_operation(
            cast(Any, content), lambda **kwargs: kwargs
        )
    assert str(error.value) == "repository README must be non-empty"


@pytest.mark.parametrize(
    "raw_path", ["../README.md", "/README.md", "a\\b.md", "a\nb.md"]
)
def test_publication_rejects_unsafe_study_document_paths(raw_path: str) -> None:
    with pytest.raises(ModelPublicationError) as error:
        publication._study_document_operation(
            raw_path,
            "content",
            lambda **kwargs: kwargs,
        )
    assert str(error.value) == "study document path is unsafe"


@pytest.mark.parametrize("content", [None, "", "   "])
def test_publication_rejects_empty_study_document_content(content: object) -> None:
    with pytest.raises(ModelPublicationError) as error:
        publication._study_document_operation(
            "studies/landuse-v1/README.md",
            cast(Any, content),
            lambda **kwargs: kwargs,
        )
    assert str(error.value) == "study document content is empty"


def test_publication_normalizes_and_encodes_study_document_operations() -> None:
    calls: list[dict[str, object]] = []

    def factory(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return kwargs

    operation, path = publication._study_document_operation(
        "studies//landuse-v1/README.md",
        "café",
        factory,
    )

    assert path == "studies/landuse-v1/README.md"
    assert operation == calls[0]
    assert calls == [
        {
            "path_in_repo": "studies/landuse-v1/README.md",
            "path_or_fileobj": "café".encode(),
        }
    ]


def test_publication_study_operations_keep_sorted_operations_and_paths() -> None:
    calls: list[dict[str, object]] = []

    def factory(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        return kwargs

    operations, paths = publication._study_operations(
        {
            "studies/z/results.json": "z",
            "studies/a/README.md": "a",
        },
        factory,
    )

    assert operations == calls
    assert paths == ["studies/a/README.md", "studies/z/results.json"]


def test_publication_study_operations_wrap_factory_errors() -> None:
    cause = RuntimeError("factory failed")

    def failing_factory(**kwargs: object) -> object:
        del kwargs
        raise cause

    with pytest.raises(ModelPublicationError) as error:
        publication._safe_study_operations({"README.md": "# study"}, failing_factory)

    assert str(error.value) == "study document operations could not be constructed"
    assert error.value.__cause__ is cause


def test_publication_checkpoint_output_keeps_only_allowed_regular_files(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "checkpoint-10"
    directory.mkdir()
    for filename in (
        "config.json",
        "model.safetensors",
        "trainer_state.json",
        ".env",
    ):
        (directory / filename).write_text("content", encoding="utf-8")
    (directory / "subdirectory").mkdir()

    assert [path.name for path in publication._checkpoint_output_files(directory)] == [
        "config.json",
        "model.safetensors",
        "trainer_state.json",
    ]


def test_publication_checkpoint_output_rejects_an_empty_directory(
    tmp_path: Path,
) -> None:
    with pytest.raises(ModelPublicationError) as error:
        publication._checkpoint_output_files(tmp_path)
    assert str(error.value) == "checkpoint output contains no files"


def test_publication_checkpoint_output_wraps_read_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cause = OSError("cannot read")

    def failing_iterdir(path: Path) -> object:
        del path
        raise cause

    monkeypatch.setattr(Path, "iterdir", failing_iterdir)

    with pytest.raises(ModelPublicationError) as error:
        publication._checkpoint_output_files(tmp_path)
    assert str(error.value) == "checkpoint output cannot be read"
    assert error.value.__cause__ is cause


def test_publication_complete_checkpoint_wraps_checkpoint_evidence_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cause = publication.CheckpointError("invalid evidence")

    def failing_find(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise cause

    monkeypatch.setattr(publication, "find_complete_checkpoint", failing_find)

    with pytest.raises(ModelPublicationError) as error:
        publication._require_complete_checkpoint(tmp_path, identity={})
    assert str(error.value) == "checkpoint evidence is invalid"
    assert error.value.__cause__ is cause


def test_publication_complete_checkpoint_requires_a_selected_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(publication, "find_complete_checkpoint", lambda *a, **k: None)

    with pytest.raises(ModelPublicationError) as error:
        publication._require_complete_checkpoint(tmp_path, identity={})
    assert str(error.value) == "model output is not a complete checkpoint"


def test_publication_checkpoint_path_requires_the_standard_directory_name() -> None:
    with pytest.raises(ModelPublicationError) as error:
        publication._checkpoint_path_in_repo(
            Path("model.safetensors"),
            Path("not-a-checkpoint"),
            identity=CHECKPOINT_IDENTITY,
        )
    assert str(error.value) == "checkpoint directory name is invalid"
    assert publication._checkpoint_path_in_repo(
        Path("model.safetensors"),
        Path("checkpoint-12"),
        identity=CHECKPOINT_IDENTITY,
    ) == (
        "experiments/landuse/run-aaaaaaaaaaaaaaaaaaaa/checkpoints/step-12/"
        "model.safetensors"
    )


def test_publication_model_commit_has_a_stable_public_contract(
    tmp_path: Path,
) -> None:
    commits: list[dict[str, object]] = []

    class FakeHub:
        def create_commit(self, **kwargs: object) -> SimpleNamespace:
            commits.append(kwargs)
            return SimpleNamespace(
                oid="c" * 40,
                commit_url="https://huggingface.co/test/commit/" + "c" * 40,
            )

    result = publish_model_directory(
        _model_directory(tmp_path),
        "owner/model",
        hub_api=FakeHub(),
        operation_factory=lambda **kwargs: kwargs,
    )

    assert result.repository_id == "owner/model"
    assert commits[0]["repo_id"] == "owner/model"
    assert commits[0]["repo_type"] == "model"
    assert commits[0]["revision"] == "main"
    assert commits[0]["commit_message"] == "Publish completed classifier model"
    assert commits[0]["operations"]


def test_publication_model_operations_failure_is_wrapped(
    tmp_path: Path,
) -> None:
    cause = ValueError("operation factory failed")

    def failing_factory(**kwargs: object) -> object:
        del kwargs
        raise cause

    with pytest.raises(ModelPublicationError) as error:
        publish_model_directory(
            _model_directory(tmp_path),
            "owner/model",
            operation_factory=failing_factory,
        )

    assert str(error.value) == "Hugging Face model operations could not be constructed"
    assert error.value.__cause__ is cause


def test_publication_model_commit_failure_is_wrapped(tmp_path: Path) -> None:
    cause = RuntimeError("network failed")

    class FailingHub:
        def create_commit(self, **kwargs: object) -> None:
            del kwargs
            raise cause

    with pytest.raises(ModelPublicationError) as error:
        publish_model_directory(
            _model_directory(tmp_path),
            "owner/model",
            hub_api=FailingHub(),
            operation_factory=lambda **kwargs: kwargs,
        )

    assert str(error.value) == "Hugging Face model publication failed"
    assert error.value.__cause__ is cause


def test_publication_study_commit_has_a_stable_public_contract() -> None:
    commits: list[dict[str, object]] = []

    class FakeHub:
        def create_commit(self, **kwargs: object) -> SimpleNamespace:
            commits.append(kwargs)
            return SimpleNamespace(
                oid="d" * 40,
                commit_url="https://huggingface.co/test/commit/" + "d" * 40,
            )

    result = publish_study_documents(
        "owner/model",
        {"studies/landuse-v1/README.md": "# study\n"},
        hub_api=FakeHub(),
        operation_factory=lambda **kwargs: kwargs,
    )

    assert result.repository_id == "owner/model"
    assert commits[0]["repo_id"] == "owner/model"
    assert commits[0]["repo_type"] == "model"
    assert commits[0]["revision"] == "main"
    assert commits[0]["commit_message"] == "Update classifier study report"
    assert commits[0]["operations"]


def test_publication_study_rejects_empty_documents_and_wraps_factory_failure() -> None:
    with pytest.raises(ModelPublicationError) as empty_error:
        publish_study_documents("owner/model", {})
    assert str(empty_error.value) == "study documents cannot be empty"

    cause = ValueError("operation factory failed")

    def failing_factory(**kwargs: object) -> object:
        del kwargs
        raise cause

    with pytest.raises(ModelPublicationError) as error:
        publish_study_documents(
            "owner/model",
            {"README.md": "# study\n"},
            operation_factory=failing_factory,
        )

    assert str(error.value) == "study document operations could not be constructed"
    assert error.value.__cause__ is cause


def test_publication_study_commit_failure_is_wrapped() -> None:
    cause = RuntimeError("network failed")

    class FailingHub:
        def create_commit(self, **kwargs: object) -> None:
            del kwargs
            raise cause

    with pytest.raises(ModelPublicationError) as error:
        publish_study_documents(
            "owner/model",
            {"README.md": "# study\n"},
            hub_api=FailingHub(),
            operation_factory=lambda **kwargs: kwargs,
        )

    assert str(error.value) == "study documentation publication failed"
    assert error.value.__cause__ is cause


def test_publication_checkpoint_commit_has_a_stable_public_contract(
    tmp_path: Path,
) -> None:
    commits: list[dict[str, object]] = []

    class FakeHub:
        def create_commit(self, **kwargs: object) -> SimpleNamespace:
            commits.append(kwargs)
            return SimpleNamespace(
                oid="e" * 40,
                commit_url="https://huggingface.co/test/commit/" + "e" * 40,
            )

    result = publish_checkpoint_directory(
        _checkpoint_directory(tmp_path),
        "owner/model",
        identity=CHECKPOINT_IDENTITY,
        hub_api=FakeHub(),
        operation_factory=lambda **kwargs: kwargs,
    )

    assert result.repository_id == "owner/model"
    assert commits[0]["repo_id"] == "owner/model"
    assert commits[0]["repo_type"] == "model"
    assert commits[0]["revision"] == "main"
    assert commits[0]["commit_message"] == "Publish checkpoint checkpoint-10"
    assert commits[0]["operations"]


def test_publication_checkpoint_operation_failure_is_wrapped(tmp_path: Path) -> None:
    cause = ValueError("operation factory failed")

    def failing_factory(**kwargs: object) -> object:
        del kwargs
        raise cause

    with pytest.raises(ModelPublicationError) as error:
        publish_checkpoint_directory(
            _checkpoint_directory(tmp_path),
            "owner/model",
            identity=CHECKPOINT_IDENTITY,
            operation_factory=failing_factory,
        )

    assert (
        str(error.value)
        == "Hugging Face checkpoint operations could not be constructed"
    )
    assert error.value.__cause__ is cause


def test_publication_checkpoint_commit_failure_is_wrapped(tmp_path: Path) -> None:
    cause = RuntimeError("network failed")

    class FailingHub:
        def create_commit(self, **kwargs: object) -> None:
            del kwargs
            raise cause

    with pytest.raises(ModelPublicationError) as error:
        publish_checkpoint_directory(
            _checkpoint_directory(tmp_path),
            "owner/model",
            identity=CHECKPOINT_IDENTITY,
            hub_api=FailingHub(),
            operation_factory=lambda **kwargs: kwargs,
        )

    assert str(error.value) == "Hugging Face checkpoint publication failed"
    assert error.value.__cause__ is cause


@pytest.mark.parametrize("repository_id", [None, "", "   ", "owner\nmodel"])
def test_publication_model_rejects_an_invalid_repository_id(
    tmp_path: Path,
    repository_id: object,
) -> None:
    with pytest.raises(ModelPublicationError) as error:
        publish_model_directory(_model_directory(tmp_path), repository_id)
    assert str(error.value) == "repository_id must be a non-blank single-line string"


@pytest.mark.parametrize("repository_id", [None, "", "   ", "owner\nmodel"])
def test_publication_study_rejects_an_invalid_repository_id(
    repository_id: object,
) -> None:
    with pytest.raises(ModelPublicationError) as error:
        publish_study_documents(repository_id, {"README.md": "# study\n"})
    assert str(error.value) == "repository_id must be a non-blank single-line string"


@pytest.mark.parametrize("repository_id", [None, "", "   ", "owner\nmodel"])
def test_publication_checkpoint_rejects_an_invalid_repository_id(
    tmp_path: Path,
    repository_id: object,
) -> None:
    with pytest.raises(ModelPublicationError) as error:
        publish_checkpoint_directory(
            _checkpoint_directory(tmp_path),
            repository_id,
            identity=CHECKPOINT_IDENTITY,
        )
    assert str(error.value) == "repository_id must be a non-blank single-line string"


def test_publication_default_operation_factory_wraps_dependency_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cause = ImportError("missing hub dependency")

    def failing_import(name: str) -> object:
        del name
        raise cause

    monkeypatch.setattr(publication, "import_module", failing_import)

    with pytest.raises(ModelPublicationError) as error:
        publication._default_operation_factory()

    assert str(error.value) == (
        "Hugging Face publication requires the training dependencies"
    )
    assert error.value.__cause__ is cause


def test_publication_default_hub_api_wraps_dependency_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cause = ImportError("missing hub dependency")
    calls: list[str] = []

    def configure() -> None:
        calls.append("configure")

    def failing_import(name: str) -> object:
        calls.append(name)
        raise cause

    monkeypatch.setattr(publication, "configure_huggingface_http", configure)
    monkeypatch.setattr(publication, "import_module", failing_import)

    with pytest.raises(ModelPublicationError) as error:
        publication._default_hub_api()

    assert str(error.value) == (
        "Hugging Face publication requires the training dependencies"
    )
    assert calls == ["configure", "huggingface_hub"]
    assert error.value.__cause__ is cause


def test_publication_commit_wraps_hub_failure() -> None:
    cause = RuntimeError("network failed")

    class FailingHub:
        def create_commit(self, **kwargs: object) -> None:
            del kwargs
            raise cause

    with pytest.raises(ModelPublicationError, match="publication failed") as error:
        publication._commit_publication(
            api=FailingHub(),
            repository="owner/model",
            operations=[],
            commit_message="Publish artifacts",
            published_paths=[],
            failure_message="publication failed",
        )

    assert error.value.__cause__ is cause


def test_publication_model_repository_setup_rejects_and_wraps_failures() -> None:
    with pytest.raises(ModelPublicationError) as invalid_error:
        ensure_model_repository(None)
    assert str(invalid_error.value) == (
        "repository_id must be a non-blank single-line string"
    )

    cause = RuntimeError("network failed")

    class FailingHub:
        def create_repo(self, **kwargs: object) -> None:
            del kwargs
            raise cause

    with pytest.raises(ModelPublicationError) as error:
        ensure_model_repository("owner/model", hub_api=FailingHub())

    assert str(error.value) == "Hugging Face model repository setup failed"
    assert error.value.__cause__ is cause


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
