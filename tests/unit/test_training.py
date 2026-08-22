import logging
import os
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from osm_polygon_sentence_classifier import training_runtime
from osm_polygon_sentence_classifier.checkpointing import (
    CheckpointError,
    find_latest_complete_checkpoint,
    write_checkpoint_manifest,
)
from osm_polygon_sentence_classifier.config import ProjectConfig
from osm_polygon_sentence_classifier.dataset_contract import (
    LANDUSE_DATASET_CONTRACT,
    WORLDWIDE_PLACE_RELEVANCE_V2_CONTRACT,
    DatasetContract,
)
from osm_polygon_sentence_classifier.dataset_loader import split_for_polygon
from osm_polygon_sentence_classifier.tracking import (
    TRACKIO_STATIC_SPACE_ID,
    TrackingError,
    TrackioSettings,
)
from osm_polygon_sentence_classifier.training import (
    PLACE_RELEVANCE_V2_DEFAULT_MAX_STEPS,
    TrainingConfig,
    TrainingError,
    TrainingRecord,
    TrainingResult,
    _evaluate_test_dataset,
    _evaluate_validation_dataset,
    _is_clean_relative_path,
    _is_finite_fraction,
    _load_hugging_face_token,
    _make_tokenized_dataset,
    _model_card_identity,
    _training_argument_values,
    _training_config_payload,
    _validate_checkpoint_resume_arguments,
    _validate_learning_rate,
    _validate_model_name,
    _validate_positive_integer,
    _validate_run_name,
    iter_split_training_records,
    place_relevance_v2_training_config,
    train_landuse_classifier,
    train_place_relevance_classifier,
)


def _row(
    *,
    sentence_id: str = "sentence-1",
    polygon_id: str = "polygon-1",
    text: str = "Normalized sentence.",
    label: str = "yes",
    content_hash: str | None = None,
) -> dict[str, object]:
    row = dict.fromkeys(LANDUSE_DATASET_CONTRACT.required_columns)
    row.update(
        {
            "sentence_id": sentence_id,
            "polygon_id": polygon_id,
            "region": "afghanistan",
            "sentence_text_normalized": text,
            "sentence_content_hash": content_hash,
            "landuse_relevance": label,
        }
    )
    return row


def _worldwide_row(
    *,
    sentence_id: str = "sentence-1",
    polygon_id: str = "polygon-1",
    text: str = "A physical description.",
    label: str = "yes",
    content_hash: str | None = None,
) -> dict[str, object]:
    row = dict.fromkeys(WORLDWIDE_PLACE_RELEVANCE_V2_CONTRACT.required_columns)
    row.update(
        {
            "sentence_id": sentence_id,
            "polygon_id": polygon_id,
            "region": "us-colorado",
            "sentence_text_normalized": text,
            "sentence_content_hash": content_hash,
            "place_relevance": label,
        }
    )
    return row


def _polygon_for_split(split: str) -> str:
    for index in range(100):
        polygon_id = f"polygon-{index}"
        if split_for_polygon(polygon_id, validation_fraction=0.5) == split:
            return polygon_id
    raise AssertionError(f"no polygon found for {split}")


def test_training_argument_values_preserves_the_complete_trainer_contract(
    tmp_path: Path,
) -> None:
    config = TrainingConfig(
        per_device_train_batch_size=3,
        per_device_eval_batch_size=4,
        learning_rate=0.012,
        max_steps=17,
        seed=9,
        logging_steps=2,
        eval_steps=5,
        save_steps=5,
        save_total_limit=7,
        run_name="payload-contract",
    )
    output_directory = tmp_path / "output"

    assert _training_argument_values(
        config,
        output_directory=output_directory,
        tracking_project="tracking-project",
        trackio_space_id="space-id",
        trackio_bucket_id="bucket-id",
    ) == {
        "output_dir": str(output_directory),
        "per_device_train_batch_size": 3,
        "per_device_eval_batch_size": 4,
        "learning_rate": 0.012,
        "max_steps": 17,
        "seed": 9,
        "logging_steps": 2,
        "eval_strategy": "steps",
        "eval_steps": 5,
        "save_strategy": "steps",
        "save_steps": 5,
        "save_total_limit": 7,
        "report_to": ["trackio"],
        "project": "tracking-project",
        "run_name": "payload-contract",
        "trackio_space_id": "space-id",
        "trackio_bucket_id": "bucket-id",
        "trackio_static_space_id": False,
        "remove_unused_columns": False,
    }


def test_load_hugging_face_token_prefers_the_trimmed_environment_token(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HF_TOKEN", "  environment-token  ")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))

    assert _load_hugging_face_token() == "environment-token"


def test_load_hugging_face_token_reads_the_configured_utf8_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    token_path = tmp_path / "hf-home" / "token"
    token_path.parent.mkdir(parents=True)
    token_path.write_text("  fichier-éclair  ", encoding="utf-8")
    monkeypatch.setenv("HF_HOME", str(token_path.parent))
    read_calls: list[tuple[Path, tuple[object, ...], dict[str, object]]] = []
    original_read_text = cast(Any, Path.read_text)

    def recording_read_text(path: Path, *args: object, **kwargs: object) -> str:
        read_calls.append((path, args, kwargs))
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", recording_read_text)

    assert _load_hugging_face_token() == "fichier-éclair"
    assert read_calls == [(token_path, (), {"encoding": "utf-8"})]


def test_load_hugging_face_token_uses_the_default_home_when_unconfigured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_HOME", raising=False)
    token_path = tmp_path / ".cache" / "huggingface" / "token"
    token_path.parent.mkdir(parents=True)
    token_path.write_text("default-token", encoding="utf-8")
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    read_paths: list[Path] = []
    original_read_text = cast(Any, Path.read_text)

    def recording_read_text(path: Path, *args: object, **kwargs: object) -> str:
        read_paths.append(path)
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", recording_read_text)

    assert _load_hugging_face_token() == "default-token"
    assert read_paths == [token_path]


def test_load_hugging_face_token_returns_empty_for_a_missing_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "missing-hf-home"))

    assert _load_hugging_face_token() == ""


def test_train_classifier_preserves_the_complete_orchestration_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from osm_polygon_sentence_classifier import training

    config = TrainingConfig(
        model_name_or_path="test-model",
        publish_to_hub=True,
        sync_trackio=True,
        tracking_project="tracked-project",
        run_name="orchestration-contract",
    )
    project_config = ProjectConfig()

    def rows_factory() -> Iterator[Mapping[str, object]]:
        return iter(())

    checkpoint = tmp_path / "checkpoint-100"
    checkpoint_identity = {"checkpoint": 100}
    tracking = TrackioSettings(
        project="tracked-project",
        directory=tmp_path / "tracking",
        static_space_id="static-space",
    )
    context = training._TrainingContext(
        config=config,
        project_config=project_config,
        contract=LANDUSE_DATASET_CONTRACT,
        rows_factory=rows_factory,
        output_directory=tmp_path / "output",
        model_cache_directory=tmp_path / "cache",
        tracking=tracking,
        resume_from_checkpoint=checkpoint,
        checkpoint_identity=checkpoint_identity,
    )
    calls: dict[str, object] = {}
    dependencies = object()
    tokenizer = object()
    datasets = (object(), object(), object())
    model = object()
    trainer = object()
    train_output = object()
    final_metrics: dict[str, object] = {"eval_accuracy": 0.8}
    model_card_identity: dict[str, object] = {"run_id": "run-identity"}
    model_publication = cast(Any, object())

    def record_context(received_rows_factory: object, **kwargs: object) -> object:
        calls["context"] = (received_rows_factory, kwargs)
        return context

    def record_environment(
        received_project_config: object, *, tracking_project: object
    ) -> object:
        calls["environment"] = (received_project_config, tracking_project)
        return nullcontext()

    def record_datasets(
        received_dependencies: object,
        received_context: object,
        received_tokenizer: object,
    ) -> tuple[object, object, object]:
        calls["datasets"] = (
            received_dependencies,
            received_context,
            received_tokenizer,
        )
        return datasets

    def record_trainer(received_dependencies: object, **kwargs: object) -> object:
        calls["trainer"] = (received_dependencies, kwargs)
        return trainer

    def record_finalize(
        received_trainer: object, **kwargs: object
    ) -> tuple[object, dict[str, object], dict[str, object]]:
        calls["finalize"] = (received_trainer, kwargs)
        return train_output, final_metrics, model_card_identity

    def record_publish(output_directory: Path, **kwargs: object) -> object:
        calls["publish"] = (output_directory, kwargs)
        return model_publication

    def record_sync(settings: TrackioSettings, **kwargs: object) -> None:
        calls["sync"] = (settings, kwargs)

    monkeypatch.setattr(training, "_training_context", record_context)
    monkeypatch.setattr(training, "_managed_training_environment", record_environment)
    monkeypatch.setattr(
        training,
        "_restore_tracking_snapshot_if_needed",
        lambda received: calls.update(snapshot=received),
    )
    monkeypatch.setattr(
        training._training_runtime,
        "load_training_dependencies",
        lambda: dependencies,
    )
    monkeypatch.setattr(
        training,
        "_load_tokenizer",
        lambda received_dependencies, received_context: calls.update(
            tokenizer=(received_dependencies, received_context)
        )
        or tokenizer,
    )
    monkeypatch.setattr(training, "_training_datasets", record_datasets)
    monkeypatch.setattr(
        training,
        "_load_model",
        lambda received_dependencies, received_context: calls.update(
            model=(received_dependencies, received_context)
        )
        or model,
    )
    checkpoint_hub_api = object()
    monkeypatch.setattr(
        training,
        "_checkpoint_publication_api",
        lambda received_context: calls.update(checkpoint_api=received_context)
        or checkpoint_hub_api,
    )
    monkeypatch.setattr(training, "_build_training_trainer", record_trainer)
    monkeypatch.setattr(training, "_finalize_training_outputs", record_finalize)
    monkeypatch.setattr(training, "_publish_completed_model", record_publish)
    monkeypatch.setattr(training, "_sync_static_trackio", record_sync)

    result = training._train_classifier(
        rows_factory,
        config=config,
        project_config=project_config,
        contract=LANDUSE_DATASET_CONTRACT,
        resume_from_checkpoint=checkpoint,
        checkpoint_identity=checkpoint_identity,
    )

    assert calls["context"] == (
        rows_factory,
        {
            "config": config,
            "project_config": project_config,
            "contract": LANDUSE_DATASET_CONTRACT,
            "resume_from_checkpoint": checkpoint,
            "checkpoint_identity": checkpoint_identity,
        },
    )
    assert calls["environment"] == (project_config, "tracked-project")
    assert calls["snapshot"] is context
    assert calls["tokenizer"] == (dependencies, context)
    assert calls["datasets"] == (dependencies, context, tokenizer)
    assert calls["model"] == (dependencies, context)
    assert calls["checkpoint_api"] is context
    assert calls["trainer"] == (
        dependencies,
        {
            "context": context,
            "model": model,
            "tokenizer": tokenizer,
            "train_dataset": datasets[0],
            "validation_dataset": datasets[1],
            "checkpoint_hub_api": checkpoint_hub_api,
        },
    )
    assert calls["finalize"] == (
        trainer,
        {
            "context": context,
            "tokenizer": tokenizer,
            "validation_dataset": datasets[1],
            "test_dataset": datasets[2],
        },
    )
    assert calls["publish"] == (
        context.output_directory,
        {
            "project_config": project_config,
            "config": config,
            "contract": context.contract,
            "identity": model_card_identity,
            "metrics": final_metrics,
            "tracking": context.tracking,
            "checkpoint_hub_api": checkpoint_hub_api,
        },
    )
    assert calls["sync"] == (
        context.tracking,
        {"failure_message": "Trackio static snapshot failed", "finalize": True},
    )
    assert result == training.TrainingResult(
        output_directory=context.output_directory,
        train_output=train_output,
        model_publication=model_publication,
        tracking_space_id="static-space",
        metrics=final_metrics,
    )


def test_training_config_is_frozen_and_uses_a_managed_relative_output() -> None:
    config = TrainingConfig()

    assert config.model_name_or_path == "jhu-clsp/mmBERT-small"
    assert config.learning_rate == 3e-4
    assert config.run_name == "landuse-mmbert-small-frozen-head"
    assert config.publish_to_hub is False
    assert config.sync_trackio is False
    assert config.output_subdirectory == Path("models/landuse")
    assert not config.output_subdirectory.is_absolute()
    assert config.max_steps > 0
    assert config.max_length > 0
    assert config.save_total_limit == 5

    with pytest.raises(AttributeError):
        config.max_steps = 1  # type: ignore[misc]  # ty: ignore[invalid-assignment]


def test_training_config_separates_local_and_hub_checkpoint_intervals() -> None:
    config = TrainingConfig()

    assert config.save_steps == 100
    assert config.hub_checkpoint_steps == 1_000


@pytest.mark.parametrize("value", [0.1, 1, 3e-4])
def test_learning_rate_accepts_positive_finite_numbers(value: float) -> None:
    assert _validate_learning_rate(value) is None


@pytest.mark.parametrize(
    "value", [True, False, 0, -0.1, float("nan"), float("inf"), "0.1"]
)
def test_learning_rate_rejects_non_positive_or_non_finite_values(value: object) -> None:
    with pytest.raises(TrainingError) as error:
        _validate_learning_rate(value)
    assert str(error.value) == "learning_rate must be a positive finite number"


def test_training_config_rejects_misaligned_checkpoint_intervals() -> None:
    with pytest.raises(TrainingError, match="multiple of save_steps"):
        TrainingConfig(save_steps=300, hub_checkpoint_steps=1_000)


def test_training_config_supports_epoch_evaluation_and_a_held_out_test_fraction() -> (
    None
):
    config = TrainingConfig(eval_strategy="epoch", test_fraction=0.1)

    assert config.eval_strategy == "epoch"
    assert config.test_fraction == 0.1


def test_training_config_reports_the_exact_validation_fraction_error() -> None:
    with pytest.raises(TrainingError) as error:
        TrainingConfig(validation_fraction=1.1)

    assert str(error.value) == (
        "validation_fraction must be a finite number between 0 and 1"
    )


def test_training_config_rejects_an_invalid_test_fraction_even_when_the_sum_is_safe() -> (
    None
):
    with pytest.raises(TrainingError) as error:
        TrainingConfig(validation_fraction=0.2, test_fraction=-0.1)

    assert str(error.value) == (
        "validation and test fractions must be finite, non-negative, "
        "and sum to at most 1"
    )


def test_training_config_rejects_fractions_that_sum_above_one() -> None:
    with pytest.raises(TrainingError) as error:
        TrainingConfig(validation_fraction=0.7, test_fraction=0.4)

    assert str(error.value) == (
        "validation and test fractions must be finite, non-negative, "
        "and sum to at most 1"
    )


def test_training_config_allows_fractions_that_sum_to_one() -> None:
    config = TrainingConfig(validation_fraction=0.6, test_fraction=0.4)

    assert config.validation_fraction + config.test_fraction == 1


def test_worldwide_v2_config_uses_the_audited_single_epoch_budget() -> None:
    config = place_relevance_v2_training_config()

    assert PLACE_RELEVANCE_V2_DEFAULT_MAX_STEPS == 17_661
    assert config.max_steps == 17_661
    assert config.max_steps == PLACE_RELEVANCE_V2_DEFAULT_MAX_STEPS
    assert config.eval_strategy == "epoch"
    assert config.trainable_layers == "head"
    assert config.run_name == "place-relevance-v2|baseline|seed-42"
    assert config.tracking_project == "place-relevance-v2"
    assert config.artifact_namespace == "studies/place-relevance-v2/baseline"
    assert config.publish_to_hub is False
    assert config.sync_trackio is False


def test_training_config_payload_omits_only_unset_optional_controls() -> None:
    optional_names = (
        "trainable_layers",
        "class_weight_mode",
        "tracking_project",
        "artifact_namespace",
    )
    default_payload = _training_config_payload(TrainingConfig())

    assert all(name not in default_payload for name in optional_names)

    configured_payload = _training_config_payload(
        TrainingConfig(
            trainable_layers="head",
            class_weight_mode="balanced",
            tracking_project="public-project",
            artifact_namespace="studies/example",
        )
    )

    assert {name: configured_payload[name] for name in optional_names} == {
        "trainable_layers": "head",
        "class_weight_mode": "balanced",
        "tracking_project": "public-project",
        "artifact_namespace": "studies/example",
    }


@pytest.mark.parametrize(
    ("contract", "task_name"),
    [
        (LANDUSE_DATASET_CONTRACT, "landuse"),
        (WORLDWIDE_PLACE_RELEVANCE_V2_CONTRACT, "place-relevance-v2"),
    ],
)
def test_model_card_identity_fills_the_task_and_pinned_training_identity(
    contract: DatasetContract,
    task_name: str,
) -> None:
    config = TrainingConfig(
        model_name_or_path="test-model",
        model_revision="a" * 40,
        run_name="test-run",
    )

    identity = _model_card_identity(None, config=config, contract=contract)

    assert identity["task_name"] == task_name
    assert identity["dataset_revision"] == contract.provenance.repository_revision
    assert identity["model_name_or_path"] == "test-model"
    assert identity["model_revision"] == "a" * 40
    assert identity["training_config"] == {
        "model_name_or_path": "test-model",
        "model_revision": "a" * 40,
        "learning_rate": config.learning_rate,
        "max_length": config.max_length,
        "max_steps": config.max_steps,
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "eval_strategy": config.eval_strategy,
        "logging_steps": config.logging_steps,
        "eval_steps": config.eval_steps,
        "save_steps": config.save_steps,
        "hub_checkpoint_steps": config.hub_checkpoint_steps,
        "save_total_limit": config.save_total_limit,
        "seed": config.seed,
        "validation_fraction": config.validation_fraction,
        "test_fraction": config.test_fraction,
        "run_name": config.run_name,
        "output_subdirectory": "models/landuse",
        "publish_to_hub": config.publish_to_hub,
        "sync_trackio": config.sync_trackio,
    }


def test_model_card_identity_preserves_existing_fields_and_replaces_bad_config() -> (
    None
):
    config = TrainingConfig(model_name_or_path="test-model", model_revision="a" * 40)
    identity = {
        "task_name": "custom-task",
        "dataset_revision": "custom-dataset",
        "model_name_or_path": "custom-model",
        "model_revision": "custom-revision",
        "training_config": "not-a-mapping",
    }

    result = _model_card_identity(
        identity,
        config=config,
        contract=LANDUSE_DATASET_CONTRACT,
    )

    assert result["task_name"] == "custom-task"
    assert result["dataset_revision"] == "custom-dataset"
    assert result["model_name_or_path"] == "custom-model"
    assert result["model_revision"] == "custom-revision"
    assert result["training_config"] != "not-a-mapping"
    assert isinstance(result["training_config"], Mapping)


def test_model_card_identity_preserves_an_existing_training_config_mapping() -> None:
    config = TrainingConfig()
    training_config = {"source": "existing"}

    result = _model_card_identity(
        {"training_config": training_config},
        config=config,
        contract=LANDUSE_DATASET_CONTRACT,
    )

    assert result["training_config"] is training_config


@pytest.mark.parametrize("output_subdirectory", [Path("."), Path(".."), Path("/tmp")])
def test_training_config_rejects_an_unsafe_output_subdirectory(
    output_subdirectory: Path,
) -> None:
    with pytest.raises(TrainingError) as error:
        TrainingConfig(output_subdirectory=output_subdirectory)

    assert (
        str(error.value)
        == "output_subdirectory must be a clean relative path beneath the managed data root"
    )


@pytest.mark.parametrize("model_revision", ["unpinned", "A" * 40, 42])
def test_training_config_rejects_an_invalid_model_revision(
    model_revision: object,
) -> None:
    with pytest.raises(TrainingError, match="model_revision"):
        cast(Any, TrainingConfig)(model_revision=model_revision)


def test_training_config_reports_the_model_revision_error_exactly() -> None:
    with pytest.raises(TrainingError) as error:
        cast(Any, TrainingConfig)(model_revision="unpinned")

    assert (
        str(error.value)
        == "model_revision must be exactly 40 lowercase hexadecimal characters"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_steps", 0),
        ("max_length", 0),
        ("per_device_train_batch_size", 0),
        ("per_device_eval_batch_size", 0),
        ("logging_steps", 0),
        ("eval_steps", 0),
        ("save_steps", 0),
    ],
)
def test_training_config_rejects_non_positive_integer_settings(
    field: str, value: int
) -> None:
    with pytest.raises(TrainingError, match=field):
        cast(Any, TrainingConfig)(**{field: value})


def test_training_config_supports_ablation_controls() -> None:
    config = TrainingConfig(
        trainable_layers="last2",
        class_weight_mode="balanced",
        tracking_project="landuse-ablation-study-v1",
        artifact_namespace="studies/landuse-v1/a06-last2-256",
    )

    assert config.trainable_layers == "last2"
    assert config.class_weight_mode == "balanced"
    assert config.tracking_project == "landuse-ablation-study-v1"
    assert config.artifact_namespace == "studies/landuse-v1/a06-last2-256"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "trainable_layers",
            "encoder",
            "trainable_layers must be head or last2",
        ),
        (
            "class_weight_mode",
            "weighted",
            "class_weight_mode must be none or balanced",
        ),
        (
            "eval_strategy",
            "never",
            "eval_strategy must be steps or epoch",
        ),
    ],
)
def test_training_config_reports_invalid_training_modes_exactly(
    field: str, value: str, message: str
) -> None:
    with pytest.raises(TrainingError) as error:
        cast(Any, TrainingConfig)(**{field: value})

    assert str(error.value) == message


def test_training_config_rejects_an_unsafe_artifact_namespace() -> None:
    with pytest.raises(TrainingError) as error:
        TrainingConfig(artifact_namespace="../outside")

    assert str(error.value) == "artifact_namespace must be a clean relative path"


@pytest.mark.parametrize(
    "field",
    ["tracking_project", "artifact_namespace"],
)
@pytest.mark.parametrize("value", [" ", "line\nbreak", "line\rbreak", 42])
def test_training_config_rejects_invalid_tracking_names(
    field: str, value: object
) -> None:
    with pytest.raises(TrainingError) as error:
        cast(Any, TrainingConfig)(**{field: value})

    assert str(error.value) == f"{field} must be a non-empty single-line string"


@pytest.mark.parametrize("field", ["publish_to_hub", "sync_trackio"])
def test_training_config_reports_invalid_boolean_settings_exactly(field: str) -> None:
    with pytest.raises(TrainingError) as error:
        cast(Any, TrainingConfig)(**{field: 1})

    assert str(error.value) == f"{field} must be a boolean"


def test_training_config_reports_checkpoint_interval_error_exactly() -> None:
    with pytest.raises(TrainingError) as error:
        TrainingConfig(save_steps=300, hub_checkpoint_steps=1_000)

    assert str(error.value) == "hub_checkpoint_steps must be a multiple of save_steps"


def test_checkpoint_resume_validation_reports_the_missing_identity_exactly() -> None:
    with pytest.raises(TrainingError) as error:
        _validate_checkpoint_resume_arguments(
            resume_from_checkpoint=Path("checkpoint"),
            checkpoint_identity=None,
        )

    assert str(error.value) == "checkpoint identity is required for resume"


@pytest.mark.parametrize("path", [Path("."), Path(".."), Path("nested/../output")])
def test_clean_relative_path_rejects_dot_segments(path: Path) -> None:
    assert _is_clean_relative_path(path) is False


def test_clean_relative_path_rejects_a_preserved_dot_component() -> None:
    path = SimpleNamespace(
        parts=("nested", ".", "output"),
        is_absolute=lambda: False,
    )

    assert _is_clean_relative_path(cast(Any, path)) is False


@pytest.mark.parametrize("validator", [_validate_model_name, _validate_run_name])
@pytest.mark.parametrize("value", [42, "", " "])
def test_training_identity_validators_reject_invalid_values(
    validator: Callable[[object], None], value: object
) -> None:
    with pytest.raises(TrainingError):
        validator(value)


def test_training_identity_validators_report_exact_messages() -> None:
    with pytest.raises(TrainingError) as model_error:
        _validate_model_name(42)
    assert str(model_error.value) == "model_name_or_path must be a non-empty string"

    with pytest.raises(TrainingError) as run_error:
        _validate_run_name(42)
    assert str(run_error.value) == "run_name must be a non-empty string"


@pytest.mark.parametrize("value", [0, 0.5, 1])
def test_fraction_validator_accepts_the_closed_unit_interval(value: object) -> None:
    assert _is_finite_fraction(value) is True


@pytest.mark.parametrize("value", [True, False, -0.1, 1.1, 2, "0.5"])
def test_fraction_validator_rejects_non_finite_or_out_of_range_values(
    value: object,
) -> None:
    assert _is_finite_fraction(value) is False


def test_positive_integer_validator_rejects_booleans() -> None:
    with pytest.raises(TrainingError) as error:
        _validate_positive_integer("steps", True)

    assert str(error.value) == "steps must be a positive integer"


def test_positive_integer_validator_accepts_one() -> None:
    _validate_positive_integer("steps", 1)


def test_split_records_are_lazy_clean_and_label_mapped() -> None:
    train_polygon = _polygon_for_split("train")
    validation_polygon = _polygon_for_split("validation")
    rows = [
        _row(
            sentence_id="negative",
            polygon_id=train_polygon,
            text="Negative sentence.",
            label="no",
            content_hash="negative-hash",
        ),
        _row(
            sentence_id="validation",
            polygon_id=validation_polygon,
            text="Validation sentence.",
            label="yes",
            content_hash="validation-hash",
        ),
        _row(
            sentence_id="contradictory",
            polygon_id=train_polygon,
            text="Contradictory sentence.",
            label="yes",
            content_hash="conflict",
        ),
        _row(
            sentence_id="contradictory-no",
            polygon_id=validation_polygon,
            text="Contradictory sentence.",
            label="no",
            content_hash="conflict",
        ),
    ]
    factory_calls = 0

    def rows_factory() -> Iterable[Mapping[str, object]]:
        nonlocal factory_calls
        factory_calls += 1
        return iter(rows)

    records = iter_split_training_records(
        rows_factory,
        split="train",
        validation_fraction=0.5,
    )

    assert factory_calls == 0
    assert list(records) == [{"text": "Negative sentence.", "labels": 0}]
    assert factory_calls == 2


@pytest.mark.parametrize("split", ["validation", "test"])
def test_split_records_forwards_all_controls_and_keeps_later_matches(
    monkeypatch: pytest.MonkeyPatch,
    split: str,
) -> None:
    from osm_polygon_sentence_classifier import training

    contract = WORLDWIDE_PLACE_RELEVANCE_V2_CONTRACT
    calls: list[dict[str, object]] = []
    examples = [
        SimpleNamespace(split="train", text="ignored", label="no"),
        SimpleNamespace(split=split, text="first", label="yes"),
        SimpleNamespace(split=split, text="second", label="no"),
    ]

    def record_clean_examples(
        rows_factory: Callable[[], Iterable[Mapping[str, object]]],
        **kwargs: object,
    ) -> Iterator[object]:
        calls.append({"rows_factory": rows_factory, **kwargs})
        return iter(examples)

    monkeypatch.setattr(training, "iter_clean_training_examples", record_clean_examples)

    def rows_factory() -> Iterator[Mapping[str, object]]:
        return iter(())

    records = list(
        iter_split_training_records(
            rows_factory,
            split=cast(Any, split),
            validation_fraction=0.3,
            test_fraction=0.1,
            seed=7,
            contract=contract,
        )
    )

    assert records == [
        {"text": "first", "labels": 1},
        {"text": "second", "labels": 0},
    ]
    assert calls == [
        {
            "rows_factory": rows_factory,
            "validation_fraction": 0.3,
            "test_fraction": 0.1,
            "seed": 7,
            "contract": contract,
        }
    ]


def test_split_records_preserves_its_stable_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_classifier import training

    calls: list[dict[str, object]] = []

    def record_clean_examples(
        _rows_factory: Callable[[], Iterable[Mapping[str, object]]],
        **kwargs: object,
    ) -> Iterator[object]:
        calls.append(kwargs)
        return iter(())

    monkeypatch.setattr(training, "iter_clean_training_examples", record_clean_examples)

    assert list(iter_split_training_records(lambda: iter(()), split="train")) == []
    assert calls == [
        {
            "validation_fraction": 0.2,
            "test_fraction": 0.0,
            "seed": 42,
            "contract": LANDUSE_DATASET_CONTRACT,
        }
    ]


def test_validation_evaluation_forwards_the_eval_prefix_and_copies_metrics() -> None:
    dataset = object()
    calls: list[tuple[object, str]] = []
    metrics = {"eval_accuracy": 0.9}

    class Trainer:
        def evaluate(
            self, received_dataset: object, *, metric_key_prefix: str
        ) -> Mapping[str, object]:
            calls.append((received_dataset, metric_key_prefix))
            return metrics

    result = _evaluate_validation_dataset(Trainer(), dataset)

    assert result == metrics
    assert result is not metrics
    assert calls == [(dataset, "eval")]


def test_validation_evaluation_rejects_a_trainer_without_a_callable_evaluate() -> None:
    with pytest.raises(TrainingError) as error:
        _evaluate_validation_dataset(SimpleNamespace(evaluate=None), object())

    assert str(error.value) == "Trainer does not expose validation evaluation"


def test_validation_evaluation_handles_a_trainer_without_an_evaluate_attribute() -> (
    None
):
    with pytest.raises(TrainingError) as error:
        _evaluate_validation_dataset(SimpleNamespace(), object())

    assert str(error.value) == "Trainer does not expose validation evaluation"


def test_validation_evaluation_preserves_the_evaluation_failure_as_the_cause() -> None:
    cause = RuntimeError("backend failed")

    class Trainer:
        def evaluate(self, dataset: object, *, metric_key_prefix: str) -> object:
            del dataset, metric_key_prefix
            raise cause

    with pytest.raises(TrainingError) as error:
        _evaluate_validation_dataset(Trainer(), object())

    assert str(error.value) == "validation evaluation failed"
    assert error.value.__cause__ is cause


def test_validation_evaluation_rejects_non_mapping_metrics() -> None:
    class Trainer:
        def evaluate(self, dataset: object, *, metric_key_prefix: str) -> object:
            del dataset, metric_key_prefix
            return [("eval_accuracy", 0.9)]

    with pytest.raises(TrainingError) as error:
        _evaluate_validation_dataset(Trainer(), object())

    assert str(error.value) == "validation metrics are invalid"


def test_test_evaluation_forwards_the_test_prefix_and_copies_metrics() -> None:
    dataset = object()
    calls: list[tuple[object, str]] = []
    metrics = {"test_accuracy": 0.9}

    class Trainer:
        def evaluate(
            self, received_dataset: object, *, metric_key_prefix: str
        ) -> Mapping[str, object]:
            calls.append((received_dataset, metric_key_prefix))
            return metrics

    result = _evaluate_test_dataset(Trainer(), dataset)

    assert result == metrics
    assert result is not metrics
    assert calls == [(dataset, "test")]


def test_test_evaluation_rejects_a_trainer_without_a_callable_evaluate() -> None:
    with pytest.raises(TrainingError) as error:
        _evaluate_test_dataset(SimpleNamespace(evaluate=None), object())

    assert str(error.value) == "Trainer does not expose held-out test evaluation"


def test_test_evaluation_handles_a_trainer_without_an_evaluate_attribute() -> None:
    with pytest.raises(TrainingError) as error:
        _evaluate_test_dataset(SimpleNamespace(), object())

    assert str(error.value) == "Trainer does not expose held-out test evaluation"


def test_test_evaluation_preserves_the_evaluation_failure_as_the_cause() -> None:
    cause = RuntimeError("backend failed")

    class Trainer:
        def evaluate(self, dataset: object, *, metric_key_prefix: str) -> object:
            del dataset, metric_key_prefix
            raise cause

    with pytest.raises(TrainingError) as error:
        _evaluate_test_dataset(Trainer(), object())

    assert str(error.value) == "held-out test evaluation failed"
    assert error.value.__cause__ is cause


def test_test_evaluation_rejects_non_mapping_metrics() -> None:
    class Trainer:
        def evaluate(self, dataset: object, *, metric_key_prefix: str) -> object:
            del dataset, metric_key_prefix
            return [("test_accuracy", 0.9)]

    with pytest.raises(TrainingError) as error:
        _evaluate_test_dataset(Trainer(), object())

    assert str(error.value) == "held-out test metrics are invalid"


def test_split_records_rejects_an_unsupported_split_with_exact_error() -> None:
    with pytest.raises(TrainingError) as error:
        list(
            iter_split_training_records(
                lambda: iter(()),
                split=cast(Any, "holdout"),
            )
        )

    assert str(error.value) == "unsupported dataset split: 'holdout'"


class _FakeDataset:
    created: list["_FakeDataset"] = []

    def __init__(self, generator: Callable[[], Iterator[Mapping[str, object]]]) -> None:
        self.generator = generator
        self.map_calls: list[dict[str, object]] = []

    @classmethod
    def from_generator(
        cls, generator: Callable[[], Iterator[Mapping[str, object]]]
    ) -> "_FakeDataset":
        dataset = cls(generator)
        cls.created.append(dataset)
        return dataset

    def map(self, function: Callable[..., object], **kwargs: object) -> "_FakeDataset":
        self.map_calls.append({"function": function, **kwargs})
        return self


class _FakeTokenizer:
    from_pretrained_calls: list[dict[str, object]] = []
    save_pretrained_calls: list[str] = []

    @classmethod
    def from_pretrained(cls, *args: object, **kwargs: object) -> "_FakeTokenizer":
        cls.from_pretrained_calls.append({"args": args, **kwargs})
        return cls()

    def save_pretrained(self, output_directory: str) -> None:
        self.save_pretrained_calls.append(output_directory)


def test_make_tokenized_dataset_preserves_stream_and_tokenizer_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_classifier import training

    calls: list[dict[str, object]] = []
    generated: list[Callable[[], Iterator[Mapping[str, object]]]] = []
    split_calls: list[dict[str, object]] = []
    mapped = object()

    class RecordingDataset:
        @classmethod
        def from_generator(
            cls, generator: Callable[[], Iterator[Mapping[str, object]]]
        ) -> "RecordingDataset":
            generated.append(generator)
            return cls()

        def map(self, function: Callable[..., object], **kwargs: object) -> object:
            calls.append({"function": function, **kwargs})
            function({"text": ["A sentence to tokenize."]})
            return mapped

    class RecordingTokenizer:
        def __call__(self, texts: object, **kwargs: object) -> Mapping[str, object]:
            calls.append({"texts": texts, **kwargs})
            return {"input_ids": [[1, 2, 3]]}

    def factory() -> Iterator[Mapping[str, object]]:
        return iter([_row(polygon_id="polygon-1")])

    def recording_split_records(
        rows_factory: Callable[[], Iterable[Mapping[str, object]]],
        **kwargs: object,
    ) -> Iterator[TrainingRecord]:
        split_calls.append({"rows_factory": rows_factory, **kwargs})
        return iter([{"text": "Normalized sentence.", "labels": 1}])

    monkeypatch.setattr(
        training, "iter_split_training_records", recording_split_records
    )

    config = TrainingConfig(
        validation_fraction=0.5,
        max_length=32,
        seed=7,
    )
    dependencies = training_runtime.TrainingDependencies(
        iterable_dataset=RecordingDataset,
        auto_tokenizer=object(),
        auto_model_for_sequence_classification=object(),
        data_collator_with_padding=object(),
        training_arguments=object(),
        trainer=object(),
    )

    result = _make_tokenized_dataset(
        dependencies,
        factory,
        split="train",
        config=config,
        contract=LANDUSE_DATASET_CONTRACT,
        tokenizer=RecordingTokenizer(),
    )

    assert result is mapped
    assert len(generated) == 1
    assert list(generated[0]()) == [{"text": "Normalized sentence.", "labels": 1}]
    assert split_calls == [
        {
            "rows_factory": factory,
            "split": "train",
            "validation_fraction": 0.5,
            "test_fraction": 0.0,
            "seed": 7,
            "contract": LANDUSE_DATASET_CONTRACT,
        }
    ]
    assert calls[0]["batched"] is True
    assert calls[0]["remove_columns"] == ["text"]
    assert calls[1] == {
        "texts": ["A sentence to tokenize."],
        "truncation": True,
        "max_length": 32,
    }


def test_training_datasets_builds_each_requested_split_in_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from osm_polygon_sentence_classifier import training

    config = TrainingConfig(test_fraction=0.1)

    def rows_factory() -> Iterator[Mapping[str, object]]:
        return iter([_row()])

    context = training._TrainingContext(
        config=config,
        project_config=ProjectConfig(),
        contract=LANDUSE_DATASET_CONTRACT,
        rows_factory=rows_factory,
        output_directory=tmp_path / "output",
        model_cache_directory=tmp_path / "cache",
        tracking=TrackioSettings(project="test", directory=tmp_path / "tracking"),
        resume_from_checkpoint=None,
        checkpoint_identity=None,
    )
    calls: list[dict[str, object]] = []

    def recording_dataset(
        dependencies: object,
        received_rows_factory: Callable[[], Iterable[Mapping[str, object]]],
        *,
        split: str,
        config: TrainingConfig,
        contract: DatasetContract,
        tokenizer: object,
    ) -> str:
        calls.append(
            {
                "dependencies": dependencies,
                "rows_factory": received_rows_factory,
                "split": split,
                "config": config,
                "contract": contract,
                "tokenizer": tokenizer,
            }
        )
        return f"{split}-dataset"

    monkeypatch.setattr(training, "_make_tokenized_dataset", recording_dataset)
    dependencies = object()
    tokenizer = object()

    result = training._training_datasets(
        cast(Any, dependencies), context, cast(Any, tokenizer)
    )

    assert result == ("train-dataset", "validation-dataset", "test-dataset")
    assert calls == [
        {
            "dependencies": dependencies,
            "rows_factory": rows_factory,
            "split": split,
            "config": config,
            "contract": LANDUSE_DATASET_CONTRACT,
            "tokenizer": tokenizer,
        }
        for split in ("train", "validation", "test")
    ]


def test_training_datasets_omits_the_test_dataset_without_a_test_fraction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from osm_polygon_sentence_classifier import training

    config = TrainingConfig()
    calls: list[str] = []
    context = training._TrainingContext(
        config=config,
        project_config=ProjectConfig(),
        contract=LANDUSE_DATASET_CONTRACT,
        rows_factory=lambda: iter([_row()]),
        output_directory=tmp_path / "output",
        model_cache_directory=tmp_path / "cache",
        tracking=TrackioSettings(project="test", directory=tmp_path / "tracking"),
        resume_from_checkpoint=None,
        checkpoint_identity=None,
    )

    def recording_dataset(*_args: object, split: str, **_kwargs: object) -> str:
        calls.append(split)
        return f"{split}-dataset"

    monkeypatch.setattr(training, "_make_tokenized_dataset", recording_dataset)

    assert training._training_datasets(
        cast(Any, object()), context, cast(Any, object())
    ) == (
        "train-dataset",
        "validation-dataset",
        None,
    )
    assert calls == ["train", "validation"]


def test_publish_completed_model_passes_the_complete_publication_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from osm_polygon_sentence_classifier import training

    config = TrainingConfig(
        model_name_or_path="test-model",
        publish_to_hub=True,
    )
    project_config = ProjectConfig()
    contract = LANDUSE_DATASET_CONTRACT
    identity: Mapping[str, object] = {"run_id": "a" * 20}
    tracking = TrackioSettings(project="test", directory=tmp_path / "tracking")
    publication = object()
    readme_calls: list[dict[str, object]] = []
    publish_calls: list[dict[str, object]] = []

    def readme(**kwargs: object) -> str:
        readme_calls.append(kwargs)
        return "generated README"

    def publish(
        output_directory: Path,
        repository_id: str,
        **kwargs: object,
    ) -> object:
        publish_calls.append(
            {
                "output_directory": output_directory,
                "repository_id": repository_id,
                **kwargs,
            }
        )
        return publication

    monkeypatch.setattr(training, "_repository_readme", readme)
    monkeypatch.setattr(training, "publish_model_directory", publish)

    result = training._publish_completed_model(
        tmp_path / "model",
        project_config=project_config,
        config=config,
        contract=contract,
        identity=identity,
        metrics={"eval_f1": 0.8},
        tracking=tracking,
        checkpoint_hub_api=None,
    )

    assert result is publication
    assert readme_calls == [
        {
            "config": config,
            "contract": contract,
            "identity": identity,
            "tracking": tracking,
        }
    ]
    assert publish_calls == [
        {
            "output_directory": tmp_path / "model",
            "repository_id": project_config.target_model_repository_id,
            "identity": identity,
            "repository_readme": "generated README",
        }
    ]


def test_publish_completed_model_publishes_worldwide_baseline_documents(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from osm_polygon_sentence_classifier import training

    config = TrainingConfig(
        model_name_or_path="test-model",
        publish_to_hub=True,
        artifact_namespace="studies/place-relevance-v2/baseline",
    )
    project_config = ProjectConfig()
    identity: Mapping[str, object] = {"run_id": "b" * 20}
    metrics: Mapping[str, object] = {"test_f1": 0.7}
    tracking = TrackioSettings(project="place-relevance-v2", directory=tmp_path)
    publication = object()
    publish_calls: list[dict[str, object]] = []
    document_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        training,
        "_repository_readme",
        lambda **_kwargs: "generated README",
    )
    monkeypatch.setattr(
        training,
        "publish_model_directory",
        lambda output_directory, repository_id, **kwargs: publish_calls.append(
            {
                "output_directory": output_directory,
                "repository_id": repository_id,
                **kwargs,
            }
        )
        or publication,
    )
    monkeypatch.setattr(
        training,
        "_publish_worldwide_v2_documents",
        lambda project_config, **kwargs: document_calls.append(
            {"project_config": project_config, **kwargs}
        ),
    )
    hub_api = object()

    result = training._publish_completed_model(
        tmp_path / "model",
        project_config=project_config,
        config=config,
        contract=WORLDWIDE_PLACE_RELEVANCE_V2_CONTRACT,
        identity=identity,
        metrics=metrics,
        tracking=tracking,
        checkpoint_hub_api=hub_api,
    )

    assert result is publication
    assert publish_calls == [
        {
            "output_directory": tmp_path / "model",
            "repository_id": project_config.target_model_repository_id,
            "identity": identity,
            "repository_readme": "generated README",
        }
    ]
    assert document_calls == [
        {
            "project_config": project_config,
            "config": config,
            "identity": identity,
            "metrics": metrics,
            "tracking": tracking,
            "checkpoint_hub_api": hub_api,
        }
    ]


class _FakeModel:
    from_pretrained_calls: list[dict[str, object]] = []
    instances: list["_FakeModel"] = []

    def __init__(self) -> None:
        self.encoder_parameters = [_FakeParameter(), _FakeParameter()]
        self.head = _FakeClassifier()
        self.classifier = _FakeClassifier()
        self.instances.append(self)

    @classmethod
    def from_pretrained(cls, *args: object, **kwargs: object) -> "_FakeModel":
        cls.from_pretrained_calls.append({"args": args, **kwargs})
        return cls()

    def parameters(self) -> Iterable["_FakeParameter"]:
        return [
            *self.encoder_parameters,
            *self.head.parameters(),
            *self.classifier.parameters(),
        ]


class _FakeLayer:
    def __init__(self) -> None:
        self.parameters_list = [_FakeParameter()]

    def parameters(self) -> Iterable["_FakeParameter"]:
        return self.parameters_list


class _FakeOrderedLayers:
    def __init__(self, layers: Iterable[_FakeLayer]) -> None:
        self.layers = list(layers)

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, index: int | slice) -> _FakeLayer | list[_FakeLayer]:
        return self.layers[index]


class _FakeLayeredModel(_FakeModel):
    def __init__(self) -> None:
        super().__init__()
        self.base_model = SimpleNamespace(layers=[_FakeLayer() for _ in range(4)])


class _FakeParameter:
    def __init__(self) -> None:
        self.requires_grad = True


class _FakeClassifier:
    def __init__(self) -> None:
        self.parameters_list = [_FakeParameter()]

    def parameters(self) -> Iterable[_FakeParameter]:
        return self.parameters_list


def test_last2_training_unfreezes_only_the_last_two_encoder_layers() -> None:
    from osm_polygon_sentence_classifier import training_freezing

    model = _FakeLayeredModel()

    training_freezing.configure_trainable_layers(model, "last2")

    assert all(
        parameter.requires_grad is False
        for parameter in model.base_model.layers[0].parameters()
    )
    assert all(
        parameter.requires_grad is True
        for layer in model.base_model.layers[-2:]
        for parameter in layer.parameters()
    )
    assert all(
        parameter.requires_grad is False for parameter in model.encoder_parameters
    )
    assert all(
        parameter.requires_grad is True for parameter in model.classifier.parameters()
    )


def test_last2_training_accepts_module_list_like_encoder_layers() -> None:
    from osm_polygon_sentence_classifier import training_freezing

    model = _FakeLayeredModel()
    ordered_layers = _FakeOrderedLayers(_FakeLayer() for _ in range(4))
    model.base_model.layers = ordered_layers

    training_freezing.configure_trainable_layers(model, "last2")

    assert all(
        parameter.requires_grad is False
        for parameter in ordered_layers.layers[0].parameters()
    )
    assert all(
        parameter.requires_grad is True
        for layer in ordered_layers.layers[-2:]
        for parameter in layer.parameters()
    )


def test_head_training_freezes_encoder_and_enables_classifier_heads() -> None:
    from osm_polygon_sentence_classifier import training_freezing

    model = _FakeModel()

    training_freezing.configure_trainable_layers(model, "head")

    assert all(
        parameter.requires_grad is False for parameter in model.encoder_parameters
    )
    assert all(parameter.requires_grad is True for parameter in model.head.parameters())
    assert all(
        parameter.requires_grad is True for parameter in model.classifier.parameters()
    )


def test_trainable_layer_configuration_rejects_an_unknown_mode() -> None:
    from osm_polygon_sentence_classifier import training_freezing

    with pytest.raises(TrainingError) as error:
        training_freezing.configure_trainable_layers(_FakeModel(), cast(Any, "all"))
    assert str(error.value) == "unsupported trainable layer mode"


@pytest.mark.parametrize(
    "model",
    [
        SimpleNamespace(classifier=_FakeClassifier()),
        SimpleNamespace(parameters=lambda: (), classifier=None),
        SimpleNamespace(
            parameters=lambda: (),
            classifier=SimpleNamespace(parameters=object()),
        ),
    ],
)
def test_head_training_rejects_models_without_a_classifier_contract(
    model: object,
) -> None:
    from osm_polygon_sentence_classifier import training_freezing

    with pytest.raises(TrainingError) as error:
        training_freezing.configure_trainable_layers(model, "head")
    assert (
        str(error.value)
        == "model must expose a parameters() method and classifier head"
    )


def test_head_training_rejects_a_missing_classifier_attribute() -> None:
    from osm_polygon_sentence_classifier import training_freezing

    model = SimpleNamespace(parameters=lambda: (), head=_FakeClassifier())

    with pytest.raises(TrainingError) as error:
        training_freezing.configure_trainable_layers(model, "head")
    assert (
        str(error.value)
        == "model must expose a parameters() method and classifier head"
    )


def test_head_training_allows_a_model_without_a_separate_head() -> None:
    from osm_polygon_sentence_classifier import training_freezing

    encoder_parameters = [_FakeParameter(), _FakeParameter()]
    classifier = _FakeClassifier()
    model = cast(
        Any,
        SimpleNamespace(
            parameters=lambda: [*encoder_parameters, *classifier.parameters()],
            head=None,
            classifier=classifier,
        ),
    )

    training_freezing.configure_trainable_layers(model, "head")

    assert all(parameter.requires_grad for parameter in classifier.parameters())
    assert all(not parameter.requires_grad for parameter in encoder_parameters)


@pytest.mark.parametrize("layout", ["encoder", "base_model", "model", "top_level"])
def test_last2_training_accepts_supported_encoder_layouts(layout: str) -> None:
    from osm_polygon_sentence_classifier import training_freezing

    model = cast(Any, _FakeModel())
    layers = [_FakeLayer() for _ in range(3)]
    if layout == "encoder":
        model.base_model = SimpleNamespace(encoder=SimpleNamespace(layer=layers))
    elif layout == "base_model":
        model.base_model = SimpleNamespace(model=SimpleNamespace(layers=layers))
    elif layout == "model":
        model.model = SimpleNamespace(layers=layers)
    else:
        model.layers = layers

    training_freezing.configure_trainable_layers(model, "last2")

    assert all(not parameter.requires_grad for parameter in layers[0].parameters())
    assert all(
        parameter.requires_grad
        for layer in layers[1:]
        for parameter in layer.parameters()
    )


@pytest.mark.parametrize(
    "model",
    [
        _FakeModel(),
        SimpleNamespace(
            parameters=lambda: (),
            base_model=SimpleNamespace(layers={"layer": _FakeLayer()}),
        ),
        SimpleNamespace(
            parameters=lambda: (),
            base_model=SimpleNamespace(layers="not-layers"),
        ),
    ],
)
def test_last2_training_rejects_models_without_ordered_encoder_layers(
    model: object,
) -> None:
    from osm_polygon_sentence_classifier import training_freezing

    with pytest.raises(TrainingError) as error:
        training_freezing.configure_trainable_layers(model, "last2")
    assert str(error.value) == "model does not expose ordered encoder layers"


def test_last2_training_rejects_an_encoder_with_fewer_than_two_layers() -> None:
    from osm_polygon_sentence_classifier import training_freezing

    model = cast(Any, _FakeModel())
    model.layers = [_FakeLayer()]

    with pytest.raises(TrainingError) as error:
        training_freezing.configure_trainable_layers(model, "last2")
    assert str(error.value) == "model must expose at least two encoder layers"


def test_last2_training_rejects_a_layer_without_parameters() -> None:
    from osm_polygon_sentence_classifier import training_freezing

    model = cast(Any, _FakeModel())
    model.layers = [object(), object()]

    with pytest.raises(TrainingError) as error:
        training_freezing.configure_trainable_layers(model, "last2")
    assert str(error.value) == "encoder layer does not expose parameters()"


def test_last2_training_rejects_a_model_without_parameters() -> None:
    from osm_polygon_sentence_classifier import training_freezing

    model = SimpleNamespace(layers=[_FakeLayer(), _FakeLayer()])

    with pytest.raises(TrainingError) as error:
        training_freezing.configure_trainable_layers(model, "last2")
    assert str(error.value) == "model must expose a parameters() method"


def test_last2_training_ignores_classifier_modules_without_parameters() -> None:
    from osm_polygon_sentence_classifier import training_freezing

    layers = [_FakeLayer(), _FakeLayer()]
    model = cast(
        Any,
        SimpleNamespace(
            parameters=lambda: (),
            head=object(),
            classifier=object(),
            layers=layers,
        ),
    )

    training_freezing.configure_trainable_layers(model, "last2")

    assert all(
        parameter.requires_grad is True
        for layer in layers
        for parameter in layer.parameters()
    )


def test_last2_training_enables_a_head_without_a_classifier() -> None:
    from osm_polygon_sentence_classifier import training_freezing

    head = _FakeClassifier()
    layers = [_FakeLayer(), _FakeLayer()]
    encoder_parameters = [_FakeParameter()]
    model = cast(
        Any,
        SimpleNamespace(
            parameters=lambda: [*encoder_parameters, *head.parameters()],
            head=head,
            layers=layers,
        ),
    )

    training_freezing.configure_trainable_layers(model, "last2")

    assert all(parameter.requires_grad is True for parameter in head.parameters())


def test_last2_training_enables_a_classifier_without_a_head() -> None:
    from osm_polygon_sentence_classifier import training_freezing

    classifier = _FakeClassifier()
    layers = [_FakeLayer(), _FakeLayer()]
    encoder_parameters = [_FakeParameter()]
    model = cast(
        Any,
        SimpleNamespace(
            parameters=lambda: [*encoder_parameters, *classifier.parameters()],
            classifier=classifier,
            layers=layers,
        ),
    )

    training_freezing.configure_trainable_layers(model, "last2")

    assert all(parameter.requires_grad is True for parameter in classifier.parameters())


class _LengthOnlyLayers:
    def __len__(self) -> int:
        return 2


def test_last2_training_rejects_an_object_without_layer_indexing() -> None:
    from osm_polygon_sentence_classifier import training_freezing

    model = SimpleNamespace(
        parameters=lambda: (),
        base_model=SimpleNamespace(layers=_LengthOnlyLayers()),
    )

    with pytest.raises(TrainingError) as error:
        training_freezing.configure_trainable_layers(model, "last2")
    assert str(error.value) == "model does not expose ordered encoder layers"


def test_balanced_class_weights_use_the_pinned_training_label_counts() -> None:
    assert training_runtime.balanced_class_weights() == pytest.approx(
        (44208 / (2 * 35560), 44208 / (2 * 8648))
    )


class _FakeTrainingArguments:
    calls: list[dict[str, object]] = []

    def __init__(self, **kwargs: object) -> None:
        self.calls.append(kwargs)


class _FakeDataCollator:
    calls: list[dict[str, object]] = []

    def __init__(self, **kwargs: object) -> None:
        self.calls.append(kwargs)


class _FakeTrainer:
    init_calls: list[dict[str, object]] = []
    save_model_calls: list[str] = []
    train_calls: list[str | None] = []
    evaluate_calls: list[tuple[object, str]] = []
    environment_during_train: dict[str, str | None] = {}
    train_output = object()

    def __init__(self, **kwargs: object) -> None:
        self.init_calls.append(kwargs)

    def train(self, resume_from_checkpoint: str | None = None) -> object:
        self.train_calls.append(resume_from_checkpoint)
        _FakeTrainer.environment_during_train = {
            "HF_HOME": os.environ.get("HF_HOME"),
            "TRACKIO_DIR": os.environ.get("TRACKIO_DIR"),
        }
        return self.train_output

    def evaluate(
        self, dataset: object, metric_key_prefix: str = "eval"
    ) -> dict[str, float]:
        self.evaluate_calls.append((dataset, metric_key_prefix))
        return {f"{metric_key_prefix}_accuracy": 0.9}

    def save_model(self, output_directory: str) -> None:
        self.save_model_calls.append(output_directory)


class _FakeTrainerCallback:
    def on_train_begin(
        self, args: object, state: object, control: object, **kwargs: object
    ) -> object:
        del args, state, kwargs
        return control


def test_training_wires_managed_streams_tokenizer_trainer_and_tracking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_classifier import training

    _FakeDataset.created.clear()
    _FakeTokenizer.from_pretrained_calls.clear()
    _FakeTokenizer.save_pretrained_calls.clear()
    _FakeModel.from_pretrained_calls.clear()
    _FakeModel.instances.clear()
    _FakeTrainingArguments.calls.clear()
    _FakeDataCollator.calls.clear()
    _FakeTrainer.init_calls.clear()
    _FakeTrainer.save_model_calls.clear()
    _FakeTrainer.train_calls.clear()
    _FakeTrainer.evaluate_calls.clear()
    _FakeTrainer.environment_during_train.clear()

    dependencies = training_runtime.TrainingDependencies(
        iterable_dataset=_FakeDataset,
        auto_tokenizer=_FakeTokenizer,
        auto_model_for_sequence_classification=_FakeModel,
        data_collator_with_padding=_FakeDataCollator,
        training_arguments=_FakeTrainingArguments,
        trainer=_FakeTrainer,
    )
    monkeypatch.setattr(
        training_runtime, "load_training_dependencies", lambda: dependencies
    )
    monkeypatch.setattr(training, "_write_model_card", lambda *args, **kwargs: None)

    result = train_landuse_classifier(
        rows_factory=lambda: iter([_row()]),
        config=TrainingConfig(
            model_name_or_path="test-model",
            max_steps=12,
            run_name="test-run",
        ),
    )

    assert isinstance(result, TrainingResult)
    assert result.output_directory == ProjectConfig().data_root / "models/landuse"
    assert result.train_output is _FakeTrainer.train_output
    assert len(_FakeDataset.created) == 2
    assert all(
        dataset.map_calls[0]["remove_columns"] == ["text"]
        for dataset in _FakeDataset.created
    )
    assert _FakeTokenizer.from_pretrained_calls[0] == {
        "args": ("test-model",),
        "cache_dir": str(ProjectConfig().data_root / "cache/huggingface/models"),
        "use_fast": True,
    }
    assert _FakeModel.from_pretrained_calls[0]["args"] == ("test-model",)
    assert _FakeModel.from_pretrained_calls[0]["num_labels"] == 2
    assert _FakeModel.from_pretrained_calls[0]["id2label"] == {0: "no", 1: "yes"}
    assert _FakeModel.from_pretrained_calls[0]["label2id"] == {"no": 0, "yes": 1}
    assert all(
        not parameter.requires_grad
        for parameter in _FakeModel.instances[0].encoder_parameters
    )
    assert all(
        parameter.requires_grad
        for parameter in _FakeModel.instances[0].head.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in _FakeModel.instances[0].classifier.parameters()
    )
    arguments = _FakeTrainingArguments.calls[0]
    assert arguments["output_dir"] == str(ProjectConfig().data_root / "models/landuse")
    assert arguments["max_steps"] == 12
    assert arguments["eval_strategy"] == "steps"
    assert arguments["report_to"] == ["trackio"]
    assert arguments["project"] == "osm-polygon-sentence-classifier"
    assert arguments["run_name"] == "test-run"
    assert arguments["save_total_limit"] == 5
    assert arguments["remove_unused_columns"] is False
    assert arguments["trackio_static_space_id"] is False
    assert arguments["trackio_space_id"] is None
    assert arguments["trackio_bucket_id"] is None
    assert _FakeTrainer.init_calls[0]["train_dataset"] is _FakeDataset.created[0]
    assert _FakeTrainer.init_calls[0]["eval_dataset"] is _FakeDataset.created[1]
    assert callable(_FakeTrainer.init_calls[0]["compute_metrics"])
    assert _FakeTrainer.save_model_calls == [
        str(ProjectConfig().data_root / "models/landuse")
    ]
    assert _FakeTokenizer.save_pretrained_calls == [
        str(ProjectConfig().data_root / "models/landuse")
    ]
    assert _FakeTrainer.environment_during_train == {
        "HF_HOME": str(ProjectConfig().data_root / "cache/huggingface"),
        "TRACKIO_DIR": str(ProjectConfig().data_root / "tracking"),
    }


def test_worldwide_training_evaluates_the_held_out_test_set_after_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_classifier import training

    dependencies = training_runtime.TrainingDependencies(
        iterable_dataset=_FakeDataset,
        auto_tokenizer=_FakeTokenizer,
        auto_model_for_sequence_classification=_FakeModel,
        data_collator_with_padding=_FakeDataCollator,
        training_arguments=_FakeTrainingArguments,
        trainer=_FakeTrainer,
    )
    monkeypatch.setattr(
        training_runtime, "load_training_dependencies", lambda: dependencies
    )
    monkeypatch.setattr(training, "_write_model_card", lambda *args, **kwargs: None)
    _FakeDataset.created.clear()
    _FakeTrainer.init_calls.clear()
    _FakeTrainer.evaluate_calls.clear()
    _FakeTrainingArguments.calls.clear()

    result = train_place_relevance_classifier(
        rows_factory=lambda: iter([_worldwide_row(content_hash="worldwide-hash")]),
        config=TrainingConfig(
            model_name_or_path="test-model",
            eval_strategy="epoch",
            test_fraction=0.1,
            run_name="place-relevance-v2|baseline|seed-42",
            output_subdirectory=Path("studies/place-relevance-v2/baseline/models"),
            tracking_project="place-relevance-v2",
            artifact_namespace="studies/place-relevance-v2/baseline",
        ),
    )

    assert len(_FakeDataset.created) == 3
    assert _FakeTrainingArguments.calls[0]["eval_strategy"] == "epoch"
    assert _FakeTrainer.evaluate_calls == [
        (_FakeDataset.created[1], "eval"),
        (_FakeDataset.created[2], "test"),
    ]
    assert result.metrics == {
        "eval_accuracy": 0.9,
        "test_accuracy": 0.9,
    }


def test_worldwide_training_entry_point_forwards_its_complete_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_classifier import training

    def rows_factory() -> Iterator[Mapping[str, object]]:
        return iter(())

    config = TrainingConfig(model_name_or_path="test-model")
    project_config = ProjectConfig()
    checkpoint = Path("checkpoint")
    checkpoint_identity = {"run_id": "a" * 20}
    expected = cast(Any, object())
    received: dict[str, object] = {}

    def fake_train(*args: object, **kwargs: object) -> object:
        received["args"] = args
        received["kwargs"] = kwargs
        return expected

    monkeypatch.setattr(training, "_train_classifier", fake_train)

    result = train_place_relevance_classifier(
        rows_factory,
        config=config,
        project_config=project_config,
        resume_from_checkpoint=checkpoint,
        checkpoint_identity=checkpoint_identity,
    )

    assert result is expected
    assert received["args"] == (rows_factory,)
    assert received["kwargs"] == {
        "config": config,
        "project_config": project_config,
        "contract": WORLDWIDE_PLACE_RELEVANCE_V2_CONTRACT,
        "resume_from_checkpoint": checkpoint,
        "checkpoint_identity": checkpoint_identity,
    }


def test_worldwide_training_entry_point_builds_the_default_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_classifier import training

    default_config = TrainingConfig(model_name_or_path="default-model")
    received: dict[str, object] = {}

    def fake_train(*args: object, **kwargs: object) -> object:
        del args
        received.update(kwargs)
        return cast(Any, object())

    monkeypatch.setattr(
        training, "place_relevance_v2_training_config", lambda: default_config
    )
    monkeypatch.setattr(training, "_train_classifier", fake_train)

    train_place_relevance_classifier()

    assert received["config"] is default_config
    assert received["contract"] is WORLDWIDE_PLACE_RELEVANCE_V2_CONTRACT


def test_v2_ablation_publication_does_not_replace_the_baseline_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_classifier import training
    from osm_polygon_sentence_classifier.publication import ModelPublicationResult
    from osm_polygon_sentence_classifier.tracking import TrackioSettings

    study_publications: list[object] = []
    monkeypatch.setattr(
        training,
        "publish_model_directory",
        lambda *args, **kwargs: ModelPublicationResult(
            repository_id="owner/model",
            commit_id="a" * 40,
            commit_url="https://huggingface.co/owner/model/commit/" + "a" * 40,
            files=(),
        ),
    )
    monkeypatch.setattr(
        training,
        "publish_study_documents",
        lambda *args, **kwargs: study_publications.append((args, kwargs)),
    )
    config = TrainingConfig(
        model_name_or_path="test-model",
        model_revision="b" * 40,
        run_name="place-relevance-v2-ablations|a01-head-128|seed-42",
        output_subdirectory=Path(
            "studies/place-relevance-v2-ablations/a01-head-128/models/place-relevance-v2"
        ),
        tracking_project="place-relevance-v2-ablations",
        artifact_namespace="studies/place-relevance-v2-ablations/a01-head-128",
        publish_to_hub=True,
    )

    training._publish_completed_model(
        tmp_path,
        project_config=ProjectConfig(),
        config=config,
        contract=WORLDWIDE_PLACE_RELEVANCE_V2_CONTRACT,
        identity={
            "task_name": "place-relevance-v2",
            "run_id": "c" * 20,
            "model_revision": "b" * 40,
            "training_config": {"artifact_namespace": config.artifact_namespace},
        },
        metrics={},
        tracking=TrackioSettings(
            project="place-relevance-v2-ablations",
            directory=tmp_path / "tracking",
        ),
        checkpoint_hub_api=None,
    )

    assert study_publications == []


def test_training_uses_the_ablation_trackio_project_and_artifact_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_classifier import training

    dependencies = training_runtime.TrainingDependencies(
        iterable_dataset=_FakeDataset,
        auto_tokenizer=_FakeTokenizer,
        auto_model_for_sequence_classification=_FakeModel,
        data_collator_with_padding=_FakeDataCollator,
        training_arguments=_FakeTrainingArguments,
        trainer=_FakeTrainer,
    )
    monkeypatch.setattr(
        training_runtime, "load_training_dependencies", lambda: dependencies
    )
    monkeypatch.setattr(training, "_write_model_card", lambda *args, **kwargs: None)
    _FakeTrainingArguments.calls.clear()

    result = train_landuse_classifier(
        rows_factory=lambda: iter([_row()]),
        config=TrainingConfig(
            model_name_or_path="test-model",
            run_name="landuse-v1|a01-head-128|seed-42",
            output_subdirectory=Path("studies/landuse-v1/a01-head-128/models/landuse"),
            tracking_project="osm-polygon-sentence-classifier",
            artifact_namespace="studies/landuse-v1/a01-head-128",
        ),
    )

    assert result.output_directory == ProjectConfig().data_root / (
        "studies/landuse-v1/a01-head-128/models/landuse"
    )
    assert _FakeTrainingArguments.calls[0]["project"] == (
        "osm-polygon-sentence-classifier"
    )


def test_weighted_trainer_is_used_for_balanced_loss() -> None:
    dependencies = training_runtime.TrainingDependencies(
        iterable_dataset=_FakeDataset,
        auto_tokenizer=_FakeTokenizer,
        auto_model_for_sequence_classification=_FakeModel,
        data_collator_with_padding=_FakeDataCollator,
        training_arguments=_FakeTrainingArguments,
        trainer=_FakeTrainer,
    )

    trainer = training_runtime.build_trainer(
        dependencies,
        model=object(),
        training_arguments=object(),
        train_dataset=object(),
        validation_dataset=object(),
        data_collator=object(),
        checkpoint_identity=None,
        class_weight_mode="balanced",
    )

    assert isinstance(trainer, _FakeTrainer)
    assert type(trainer) is not _FakeTrainer


def test_training_resumes_from_a_checkpoint_and_registers_identity_callback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_classifier import training

    dependencies = training_runtime.TrainingDependencies(
        iterable_dataset=_FakeDataset,
        auto_tokenizer=_FakeTokenizer,
        auto_model_for_sequence_classification=_FakeModel,
        data_collator_with_padding=_FakeDataCollator,
        training_arguments=_FakeTrainingArguments,
        trainer=_FakeTrainer,
    )
    monkeypatch.setattr(
        training_runtime, "load_training_dependencies", lambda: dependencies
    )
    monkeypatch.setattr(training, "_write_model_card", lambda *args, **kwargs: None)
    _FakeTrainer.init_calls.clear()
    _FakeTrainer.train_calls.clear()

    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    project_config = ProjectConfig.for_remote_root(tmp_path / "data")
    output = project_config.data_root / "models/landuse"
    checkpoint = output / "checkpoint-42"
    checkpoint.mkdir(parents=True)
    for filename in (
        "model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    ):
        (checkpoint / filename).write_bytes(b"checkpoint")
    (checkpoint / "trainer_state.json").write_text(
        '{"global_step": 42}', encoding="utf-8"
    )
    identity = {"run_id": "a" * 20, "model_revision": "b" * 40}
    write_checkpoint_manifest(
        checkpoint,
        identity=identity,
        global_step=42,
    )

    train_landuse_classifier(
        rows_factory=lambda: iter([_row()]),
        config=TrainingConfig(model_name_or_path="test-model"),
        project_config=project_config,
        resume_from_checkpoint=checkpoint,
        checkpoint_identity=identity,
    )

    assert _FakeTrainer.train_calls == [str(checkpoint)]
    callbacks = _FakeTrainer.init_calls[0]["callbacks"]
    assert isinstance(callbacks, list)
    assert len(callbacks) == 1


def test_requested_checkpoint_wraps_checkpoint_errors_with_the_public_message(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from osm_polygon_sentence_classifier import training

    cause = CheckpointError("invalid manifest")

    def fail_find(*_args: object, **_kwargs: object) -> None:
        raise cause

    monkeypatch.setattr(training, "find_latest_complete_checkpoint", fail_find)

    with pytest.raises(TrainingError) as error:
        training._validate_requested_checkpoint(
            tmp_path,
            resume_from_checkpoint=tmp_path / "checkpoint-1",
            identity={"run_id": "a" * 20},
        )

    assert str(error.value) == "checkpoint evidence is invalid"
    assert error.value.__cause__ is cause


def test_checkpoint_callback_writes_identity_after_a_save(tmp_path: Path) -> None:
    from osm_polygon_sentence_classifier import training_publication

    checkpoint = tmp_path / "checkpoint-7"
    checkpoint.mkdir()
    for filename in (
        "model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    ):
        (checkpoint / filename).write_bytes(b"checkpoint")
    (checkpoint / "trainer_state.json").write_text(
        '{"global_step": 7}', encoding="utf-8"
    )
    identity = {"run_id": "a" * 20, "model_revision": "b" * 40}

    training_publication.CheckpointManifestCallback(identity).on_save(
        args=SimpleNamespace(output_dir=str(tmp_path)),
        state=SimpleNamespace(global_step=7),
        control=object(),
    )

    result = find_latest_complete_checkpoint(tmp_path, identity=identity)
    assert result is not None
    assert result.global_step == 7


def test_checkpoint_callback_writes_model_card_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_classifier import training_publication

    checkpoint = tmp_path / "checkpoint-7"
    checkpoint.mkdir()
    for filename in (
        "model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    ):
        (checkpoint / filename).write_bytes(b"checkpoint")
    (checkpoint / "trainer_state.json").write_text(
        '{"global_step": 7}', encoding="utf-8"
    )
    identity = {
        "run_id": "a" * 20,
        "source_commit": "c" * 40,
        "dataset_revision": "d" * 40,
        "model_name_or_path": "test-model",
        "model_revision": "b" * 40,
        "task_name": "landuse",
        "training_config": {"max_steps": 1000},
    }
    publication_calls: list[tuple[Path, str, Mapping[str, object]]] = []
    sync_calls: list[tuple[object, dict[str, object]]] = []

    def publish(
        directory: Path,
        repository_id: str,
        *,
        identity: Mapping[str, object],
    ) -> object:
        assert (directory / "checkpoint-manifest.json").is_file()
        assert (directory / "README.md").is_file()
        publication_calls.append((directory, repository_id, identity))
        return object()

    monkeypatch.setattr(training_publication, "publish_checkpoint_directory", publish)
    monkeypatch.setattr(
        training_publication,
        "sync_project_to_static_space",
        lambda settings, **kwargs: sync_calls.append((settings, kwargs))
        or TRACKIO_STATIC_SPACE_ID,
    )

    training_publication.CheckpointManifestCallback(
        identity,
        model_repository_id="owner/model",
        tracking_settings=cast(Any, object()),
    ).on_save(
        args=SimpleNamespace(output_dir=str(tmp_path)),
        state=SimpleNamespace(global_step=7),
        control=object(),
    )

    assert publication_calls == [(checkpoint, "owner/model", identity)]
    assert len(sync_calls) == 1
    assert sync_calls[0][1] == {"finalize": False}


def test_write_model_card_forwards_metadata_and_uses_the_public_filename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_classifier import training_publication

    identity = {"run_id": "a" * 20}
    metrics = {"eval_accuracy": 0.75}
    render_calls: list[dict[str, object]] = []
    write_calls: list[tuple[Path, str, dict[str, object]]] = []

    def render(**kwargs: object) -> str:
        render_calls.append(kwargs)
        return "Résumé"

    def write_text(path: Path, text: str, **kwargs: object) -> int:
        write_calls.append((path, text, kwargs))
        return len(text)

    monkeypatch.setattr(training_publication, "render_model_card", render)
    monkeypatch.setattr(Path, "write_text", write_text)
    directory = tmp_path / "nested" / "checkpoint-7"

    training_publication.write_model_card(
        directory,
        identity=identity,
        training_metrics=metrics,
        checkpoint_step=7,
        trackio_space_id="owner/trackio",
    )

    assert render_calls == [
        {
            "identity": identity,
            "training_metrics": metrics,
            "checkpoint_step": 7,
            "trackio_space_id": "owner/trackio",
        }
    ]
    assert write_calls == [
        (
            directory / "README.md",
            "Résumé",
            {"encoding": "utf-8"},
        )
    ]


def test_write_model_card_creates_parent_directories_idempotently(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_classifier import training_publication

    directory = tmp_path / "nested" / "checkpoint-7"
    mkdir_calls: list[tuple[Path, dict[str, object]]] = []

    monkeypatch.setattr(training_publication, "render_model_card", lambda **_: "card")
    monkeypatch.setattr(Path, "exists", lambda _path: False)
    monkeypatch.setattr(
        Path,
        "mkdir",
        lambda path, **kwargs: mkdir_calls.append((path, kwargs)),
    )
    monkeypatch.setattr(Path, "write_text", lambda *_args, **_kwargs: 4)

    training_publication.write_model_card(directory, identity={})

    assert mkdir_calls == [(directory, {"parents": True, "exist_ok": True})]


def test_static_trackio_rate_limit_preserves_settings_and_logs_exactly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from osm_polygon_sentence_classifier import training_publication

    settings = TrackioSettings(project="test", directory=tmp_path)
    calls: list[object] = []

    def rate_limited_sync(received: object, *, finalize: bool) -> None:
        calls.append((received, finalize))
        raise TrackingError("429 Too Many Requests: rate limit")

    monkeypatch.setattr(
        training_publication,
        "sync_project_to_static_space",
        rate_limited_sync,
    )
    caplog.set_level(logging.WARNING)

    training_publication._sync_static_trackio(settings, failure_message="failed")

    assert calls == [(settings, False)]
    assert caplog.messages == [
        "Trackio rate limit reached; retaining the local snapshot and "
        "continuing without this checkpoint sync"
    ]


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "1", None])
def test_checkpoint_callback_rejects_invalid_hub_checkpoint_steps(
    value: object,
) -> None:
    from osm_polygon_sentence_classifier import training_publication

    with pytest.raises(TrainingError) as error:
        training_publication.CheckpointManifestCallback(
            {}, hub_checkpoint_steps=cast(Any, value)
        )

    assert str(error.value) == "hub_checkpoint_steps must be a positive integer"


def test_checkpoint_callback_warns_once_when_hub_is_rate_limited(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from osm_polygon_sentence_classifier import training_publication

    callback = training_publication.CheckpointManifestCallback({})
    caplog.set_level(logging.WARNING)

    callback._mark_hub_rate_limited()
    assert caplog.messages == [
        "Hugging Face rate limit reached; retaining local checkpoints "
        "and continuing without further checkpoint commits"
    ]
    callback._mark_hub_rate_limited()

    assert callback._hub_rate_limited is True
    assert caplog.messages == [
        "Hugging Face rate limit reached; retaining local checkpoints "
        "and continuing without further checkpoint commits"
    ]


def test_checkpoint_callback_initializes_publication_state() -> None:
    from osm_polygon_sentence_classifier import training_publication

    callback = training_publication.CheckpointManifestCallback({})

    assert callback._pending_publications == []
    assert callback._hub_rate_limited is False


def test_checkpoint_callback_wraps_future_publication_with_exact_error() -> None:
    from osm_polygon_sentence_classifier import training_publication

    cause = RuntimeError("storage unavailable")

    class Future:
        def result(self) -> object:
            raise cause

    callback = training_publication.CheckpointManifestCallback({})
    callback._pending_publications.append(Future())

    with pytest.raises(TrainingError) as error:
        callback._wait_for_next_publication()

    assert str(error.value) == "checkpoint model publication failed"
    assert error.value.__cause__ is cause
    assert callback._pending_publications == []


def test_checkpoint_callback_wraps_direct_publication_with_exact_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from osm_polygon_sentence_classifier import training_publication
    from osm_polygon_sentence_classifier.publication import ModelPublicationError

    cause = ModelPublicationError("storage unavailable")

    def fail_publish(*_args: object, **_kwargs: object) -> object:
        raise cause

    monkeypatch.setattr(
        training_publication,
        "publish_checkpoint_directory",
        fail_publish,
    )
    callback = training_publication.CheckpointManifestCallback(
        {}, model_repository_id="owner/model"
    )

    with pytest.raises(TrainingError) as error:
        callback._publish_checkpoint(tmp_path / "checkpoint-7")

    assert str(error.value) == "checkpoint model publication failed"
    assert error.value.__cause__ is cause


@pytest.mark.parametrize(
    ("args", "state"),
    [
        (SimpleNamespace(), SimpleNamespace(global_step=7)),
        (SimpleNamespace(output_dir="/tmp/output"), SimpleNamespace()),
        (SimpleNamespace(output_dir="/tmp/output"), SimpleNamespace(global_step="7")),
    ],
)
def test_checkpoint_callback_rejects_incomplete_save_inputs(
    args: object,
    state: object,
) -> None:
    from osm_polygon_sentence_classifier import training_publication

    with pytest.raises(TrainingError) as error:
        training_publication.CheckpointManifestCallback._checkpoint_save_inputs(
            args, state
        )

    assert str(error.value) == "checkpoint save did not expose a valid step"


def test_checkpoint_callback_extracts_valid_save_inputs() -> None:
    from osm_polygon_sentence_classifier import training_publication

    output_directory, global_step = (
        training_publication.CheckpointManifestCallback._checkpoint_save_inputs(
            SimpleNamespace(output_dir="/tmp/output"),
            SimpleNamespace(global_step=7),
        )
    )

    assert output_directory == Path("/tmp/output")
    assert global_step == 7


def test_checkpoint_callback_wraps_manifest_errors_with_the_original_cause(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from osm_polygon_sentence_classifier import training_publication
    from osm_polygon_sentence_classifier.checkpointing import CheckpointError

    cause = CheckpointError("invalid checkpoint")
    monkeypatch.setattr(
        training_publication,
        "write_checkpoint_manifest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(cause),
    )

    with pytest.raises(TrainingError) as error:
        training_publication.CheckpointManifestCallback({})._write_checkpoint_manifest(
            tmp_path / "checkpoint-7", 7
        )

    assert str(error.value) == "checkpoint manifest could not be written"
    assert error.value.__cause__ is cause


def test_checkpoint_callback_writes_the_merged_training_and_evaluation_metrics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from osm_polygon_sentence_classifier import training_publication

    state = object()
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        training_publication._training_metrics,
        "latest_training_metrics",
        lambda received: {"loss": received is state},
    )
    monkeypatch.setattr(
        training_publication._training_metrics,
        "latest_evaluation_metrics",
        lambda received: {"eval_accuracy": received is state},
    )
    monkeypatch.setattr(
        training_publication,
        "write_model_card",
        lambda checkpoint, **kwargs: calls.append({"checkpoint": checkpoint, **kwargs}),
    )

    callback = training_publication.CheckpointManifestCallback(
        {"run_id": "a" * 20},
        trackio_space_id="owner/trackio",
    )
    checkpoint = tmp_path / "checkpoint-7"
    callback._write_checkpoint_card(checkpoint, state=state, global_step=7)

    assert calls == [
        {
            "checkpoint": checkpoint,
            "identity": {"run_id": "a" * 20},
            "training_metrics": {"loss": True, "eval_accuracy": True},
            "checkpoint_step": 7,
            "trackio_space_id": "owner/trackio",
        }
    ]


def test_checkpoint_callback_wraps_model_card_errors_with_the_original_cause(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from osm_polygon_sentence_classifier import training_publication

    cause = OSError("read-only filesystem")
    monkeypatch.setattr(
        training_publication,
        "write_model_card",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(cause),
    )

    with pytest.raises(TrainingError) as error:
        training_publication.CheckpointManifestCallback({})._write_checkpoint_card(
            tmp_path / "checkpoint-7",
            state=SimpleNamespace(),
            global_step=7,
        )

    assert str(error.value) == "checkpoint model card could not be written"
    assert error.value.__cause__ is cause


def test_checkpoint_callback_queues_the_exact_hub_publication_call(
    tmp_path: Path,
) -> None:
    from osm_polygon_sentence_classifier import training_publication

    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class Future:
        def result(self) -> object:
            return object()

    class Hub:
        def run_as_future(self, *args: object, **kwargs: object) -> Future:
            calls.append((args, kwargs))
            return Future()

    hub = Hub()
    callback = training_publication.CheckpointManifestCallback(
        {"run_id": "a" * 20},
        model_repository_id="owner/model",
        hub_api=hub,
    )
    checkpoint = tmp_path / "checkpoint-7"

    callback._queue_checkpoint_publication(
        SimpleNamespace(save_total_limit=None), checkpoint
    )

    assert calls == [
        (
            (
                training_publication.publish_checkpoint_directory,
                checkpoint,
                "owner/model",
            ),
            {"identity": callback.identity, "hub_api": hub},
        )
    ]
    assert len(callback._pending_publications) == 1


def test_checkpoint_callback_requires_a_queue_capable_hub_api(
    tmp_path: Path,
) -> None:
    from osm_polygon_sentence_classifier import training_publication

    callback = training_publication.CheckpointManifestCallback(
        {}, model_repository_id="owner/model", hub_api=object()
    )

    with pytest.raises(TrainingError) as error:
        callback._queue_checkpoint_publication(
            SimpleNamespace(save_total_limit=None), tmp_path / "checkpoint-7"
        )

    assert str(error.value) == (
        "checkpoint publication API cannot queue background work"
    )


@pytest.mark.parametrize(
    ("save_total_limit", "pending_count", "expected"),
    [
        (None, 1, False),
        (True, 1, False),
        (0, 0, False),
        (0, 1, False),
        (-1, 1, False),
        (1, 1, True),
        (2, 1, False),
    ],
)
def test_checkpoint_callback_applies_the_publication_limit_contract(
    save_total_limit: object,
    pending_count: int,
    expected: bool,
) -> None:
    from osm_polygon_sentence_classifier import training_publication

    callback = training_publication.CheckpointManifestCallback({})
    callback._pending_publications = [object()] * pending_count

    result = callback._publication_limit_reached(
        SimpleNamespace(save_total_limit=cast(Any, save_total_limit))
    )

    assert result is expected


def test_checkpoint_callback_forwards_remote_checkpoint_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from osm_polygon_sentence_classifier import training_publication

    args = SimpleNamespace(save_total_limit=None)
    state = object()
    checkpoint = tmp_path / "checkpoint-7"
    card_calls: list[tuple[Path, dict[str, object]]] = []
    queue_calls: list[tuple[object, Path]] = []
    callback = training_publication.CheckpointManifestCallback(
        {"run_id": "a" * 20}, model_repository_id="owner/model", hub_api=object()
    )
    monkeypatch.setattr(
        callback,
        "_write_checkpoint_card",
        lambda received_checkpoint, **kwargs: card_calls.append(
            (received_checkpoint, kwargs)
        ),
    )
    monkeypatch.setattr(
        callback,
        "_queue_checkpoint_publication",
        lambda received_args, received_checkpoint: queue_calls.append(
            (received_args, received_checkpoint)
        ),
    )

    callback._publish_remote_checkpoint(args, checkpoint, state=state, global_step=7)

    assert card_calls == [(checkpoint, {"state": state, "global_step": 7})]
    assert queue_calls == [(args, checkpoint)]


def test_checkpoint_callback_on_save_forwards_all_publication_and_tracking_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from osm_polygon_sentence_classifier import training_publication

    args = SimpleNamespace(output_dir=str(tmp_path), save_total_limit=None)
    state = SimpleNamespace(global_step=7)
    control = object()
    settings = object()
    manifest_calls: list[tuple[Path, int]] = []
    remote_calls: list[dict[str, object]] = []
    sync_calls: list[tuple[object, dict[str, object]]] = []
    callback = training_publication.CheckpointManifestCallback(
        {},
        model_repository_id="owner/model",
        tracking_settings=cast(Any, settings),
    )
    monkeypatch.setattr(
        callback,
        "_write_checkpoint_manifest",
        lambda checkpoint, global_step: manifest_calls.append(
            (checkpoint, global_step)
        ),
    )
    monkeypatch.setattr(
        callback,
        "_publish_remote_checkpoint",
        lambda received_args, checkpoint, **kwargs: remote_calls.append(
            {"args": received_args, "checkpoint": checkpoint, **kwargs}
        ),
    )
    monkeypatch.setattr(
        training_publication,
        "_sync_static_trackio",
        lambda received_settings, **kwargs: sync_calls.append(
            (received_settings, kwargs)
        ),
    )

    result = callback.on_save(args, state, control)

    checkpoint = tmp_path / "checkpoint-7"
    assert result is control
    assert manifest_calls == [(checkpoint, 7)]
    assert remote_calls == [
        {"args": args, "checkpoint": checkpoint, "state": state, "global_step": 7}
    ]
    assert sync_calls == [
        (
            settings,
            {
                "failure_message": "checkpoint Trackio static snapshot failed",
                "finalize": False,
            },
        )
    ]


def test_make_checkpoint_manifest_callback_forwards_options_and_name() -> None:
    from osm_polygon_sentence_classifier import training_publication

    class TrainerCallback:
        pass

    identity = {"run_id": "a" * 20}
    tracking_settings = cast(Any, object())
    hub_api = object()
    callback = training_publication.make_checkpoint_manifest_callback(
        identity,
        TrainerCallback,
        model_repository_id="owner/model",
        trackio_space_id="owner/trackio",
        tracking_settings=tracking_settings,
        hub_api=hub_api,
        hub_checkpoint_steps=7,
    )

    assert type(callback).__name__ == "_BoundCheckpointManifestCallback"
    assert isinstance(callback, TrainerCallback)
    assert callback.identity == identity
    assert callback.model_repository_id == "owner/model"
    assert callback.trackio_space_id == "owner/trackio"
    assert callback.tracking_settings is tracking_settings
    assert callback.hub_api is hub_api
    assert callback.hub_checkpoint_steps == 7


def test_make_checkpoint_manifest_callback_uses_the_default_interval() -> None:
    from osm_polygon_sentence_classifier import training_publication

    callback = training_publication.make_checkpoint_manifest_callback({}, None)

    assert callback.hub_checkpoint_steps == 1


def test_checkpoint_callback_skips_non_interval_hub_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_classifier import training_publication

    checkpoint = tmp_path / "checkpoint-100"
    checkpoint.mkdir()
    for filename in (
        "model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    ):
        (checkpoint / filename).write_bytes(b"checkpoint")
    (checkpoint / "trainer_state.json").write_text(
        '{"global_step": 100}', encoding="utf-8"
    )
    publication_calls: list[Path] = []
    sync_calls: list[object] = []

    monkeypatch.setattr(
        training_publication,
        "publish_checkpoint_directory",
        lambda directory, *args, **kwargs: publication_calls.append(directory)
        or object(),
    )
    monkeypatch.setattr(
        training_publication,
        "_sync_static_trackio",
        lambda settings, **kwargs: sync_calls.append(settings),
    )

    training_publication.CheckpointManifestCallback(
        {"run_id": "a" * 20, "model_revision": "b" * 40},
        model_repository_id="owner/model",
        hub_checkpoint_steps=1_000,
        tracking_settings=cast(Any, object()),
    ).on_save(
        args=SimpleNamespace(output_dir=str(tmp_path)),
        state=SimpleNamespace(global_step=100),
        control=object(),
    )

    assert publication_calls == []
    assert sync_calls == []
    assert (checkpoint / "checkpoint-manifest.json").is_file()


def test_checkpoint_callback_survives_a_hub_rate_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_classifier import training_publication
    from osm_polygon_sentence_classifier.publication import ModelPublicationError

    checkpoint = tmp_path / "checkpoint-1000"
    checkpoint.mkdir()
    for filename in (
        "model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    ):
        (checkpoint / filename).write_bytes(b"checkpoint")
    (checkpoint / "trainer_state.json").write_text(
        '{"global_step": 1000}', encoding="utf-8"
    )

    def rate_limited_publish(*args: object, **kwargs: object) -> object:
        del args, kwargs
        try:
            raise RuntimeError("429 Too Many Requests: rate limit")
        except RuntimeError as cause:
            raise ModelPublicationError(
                "Hugging Face checkpoint publication failed"
            ) from cause

    monkeypatch.setattr(
        training_publication,
        "publish_checkpoint_directory",
        rate_limited_publish,
    )

    control = object()
    result = training_publication.CheckpointManifestCallback(
        {"run_id": "a" * 20, "model_revision": "b" * 40},
        model_repository_id="owner/model",
        hub_checkpoint_steps=1_000,
    ).on_save(
        args=SimpleNamespace(output_dir=str(tmp_path)),
        state=SimpleNamespace(global_step=1000),
        control=control,
    )

    assert result is control


def test_static_trackio_sync_treats_checkpoint_rate_limits_as_recoverable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from osm_polygon_sentence_classifier import training_publication

    settings = TrackioSettings(project="test", directory=tmp_path)
    calls: list[bool] = []

    def rate_limited_sync(_settings: TrackioSettings, *, finalize: bool) -> None:
        calls.append(finalize)
        raise TrackingError("429 Too Many Requests: rate limit")

    monkeypatch.setattr(
        training_publication,
        "sync_project_to_static_space",
        rate_limited_sync,
    )

    training_publication._sync_static_trackio(settings, failure_message="failed")

    assert calls == [False]


def test_static_trackio_sync_wraps_final_and_non_rate_limit_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from osm_polygon_sentence_classifier import training_publication

    settings = TrackioSettings(project="test", directory=tmp_path)

    def fail_sync(_settings: TrackioSettings, *, finalize: bool) -> None:
        del finalize
        raise TrackingError("storage unavailable")

    monkeypatch.setattr(training_publication, "sync_project_to_static_space", fail_sync)

    with pytest.raises(TrainingError, match="failed"):
        training_publication._sync_static_trackio(settings, failure_message="failed")
    with pytest.raises(TrainingError, match="final failed"):
        training_publication._sync_static_trackio(
            settings,
            failure_message="final failed",
            finalize=True,
        )


def test_static_trackio_sync_passes_the_finalize_flag_on_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from osm_polygon_sentence_classifier import training_publication

    settings = TrackioSettings(project="test", directory=tmp_path)
    calls: list[bool] = []
    monkeypatch.setattr(
        training_publication,
        "sync_project_to_static_space",
        lambda _settings, *, finalize: calls.append(finalize),
    )

    training_publication._sync_static_trackio(
        settings,
        failure_message="failed",
        finalize=True,
    )

    assert calls == [True]


@pytest.mark.parametrize(
    "error_text", ["429 Too Many Requests: rate limit", "storage unavailable"]
)
def test_checkpoint_callback_handles_publication_future_failures(
    error_text: str,
) -> None:
    from osm_polygon_sentence_classifier import training_publication

    class Future:
        def result(self) -> object:
            raise RuntimeError(error_text)

    callback = training_publication.CheckpointManifestCallback({})
    callback._pending_publications.append(Future())

    if "rate limit" in error_text:
        callback._wait_for_next_publication()
        assert callback._hub_rate_limited is True
    else:
        with pytest.raises(TrainingError, match="publication failed"):
            callback._wait_for_next_publication()
    assert callback._pending_publications == []


def test_worldwide_v2_publication_writes_the_study_documents(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_classifier import training

    config = TrainingConfig(
        model_name_or_path="test-model",
        model_revision="b" * 40,
        artifact_namespace="studies/place-relevance-v2/baseline",
        sync_trackio=True,
    )
    tracking = TrackioSettings(project="place-relevance-v2", directory=Path("tracking"))
    calls: list[tuple[object, object, object]] = []
    monkeypatch.setattr(
        training,
        "render_place_relevance_study_documents",
        lambda **kwargs: calls.append(("render", kwargs, None))
        or ("README.md", "docs"),
    )
    monkeypatch.setattr(
        training,
        "publish_study_documents",
        lambda repository, documents, *, hub_api: calls.append(
            (repository, documents, hub_api)
        ),
    )

    training._publish_worldwide_v2_documents(
        ProjectConfig(),
        config=config,
        identity={"run_id": "a" * 20},
        metrics={"eval_f1": 0.8},
        tracking=tracking,
        checkpoint_hub_api=object(),
    )

    assert calls[0][0] == "render"
    rendered_kwargs = cast(dict[str, object], calls[0][1])
    assert rendered_kwargs["trackio_space_id"] == tracking.static_space_id
    assert calls[1][0] == ProjectConfig().target_model_repository_id


def test_worldwide_v2_publication_wraps_hub_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_classifier import training
    from osm_polygon_sentence_classifier.publication import ModelPublicationError

    config = TrainingConfig(
        model_name_or_path="test-model",
        model_revision="b" * 40,
        artifact_namespace="studies/place-relevance-v2/baseline",
    )
    monkeypatch.setattr(
        training,
        "publish_study_documents",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ModelPublicationError("hub unavailable")
        ),
    )

    with pytest.raises(TrainingError, match="study documentation"):
        training._publish_worldwide_v2_documents(
            ProjectConfig(),
            config=config,
            identity={"run_id": "a" * 20},
            metrics={},
            tracking=TrackioSettings(
                project="place-relevance-v2", directory=Path("tracking")
            ),
            checkpoint_hub_api=object(),
        )


def test_checkpoint_callback_queues_hub_publication_until_training_end(
    tmp_path: Path,
) -> None:
    from osm_polygon_sentence_classifier import training_publication

    checkpoint = tmp_path / "checkpoint-7"
    checkpoint.mkdir()
    for filename in (
        "model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    ):
        (checkpoint / filename).write_bytes(b"checkpoint")
    (checkpoint / "trainer_state.json").write_text(
        '{"global_step": 7}', encoding="utf-8"
    )
    events: list[str] = []

    class Future:
        def result(self) -> object:
            events.append("wait")
            return object()

    class Hub:
        def run_as_future(
            self, function: object, *args: object, **kwargs: object
        ) -> Future:
            del function, args, kwargs
            events.append("queue")
            return Future()

    callback = training_publication.CheckpointManifestCallback(
        {"run_id": "a" * 20, "model_revision": "b" * 40},
        model_repository_id="owner/model",
        hub_api=Hub(),
    )
    callback.on_save(
        args=SimpleNamespace(output_dir=str(tmp_path)),
        state=SimpleNamespace(global_step=7),
        control=object(),
    )
    assert events == ["queue"]

    callback.on_train_end(
        args=SimpleNamespace(),
        state=SimpleNamespace(),
        control=object(),
    )

    assert events == ["queue", "wait"]


def test_checkpoint_callback_waits_before_retained_checkpoint_rotation(
    tmp_path: Path,
) -> None:
    from osm_polygon_sentence_classifier import training_publication

    events: list[str] = []

    class Future:
        def result(self) -> object:
            events.append("wait")
            return object()

    class Hub:
        def run_as_future(
            self, function: object, *args: object, **kwargs: object
        ) -> Future:
            del function, args, kwargs
            events.append("queue")
            return Future()

    callback = training_publication.CheckpointManifestCallback(
        {"run_id": "a" * 20, "model_revision": "b" * 40},
        model_repository_id="owner/model",
        hub_api=Hub(),
    )
    for step in (7, 8):
        checkpoint = tmp_path / f"checkpoint-{step}"
        checkpoint.mkdir()
        for filename in (
            "model.safetensors",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
        ):
            (checkpoint / filename).write_bytes(b"checkpoint")
        (checkpoint / "trainer_state.json").write_text(
            f'{{"global_step": {step}}}', encoding="utf-8"
        )
        callback.on_save(
            args=SimpleNamespace(output_dir=str(tmp_path), save_total_limit=2),
            state=SimpleNamespace(global_step=step),
            control=object(),
        )

    assert events == ["queue", "queue", "wait"]


def test_checkpoint_callback_supports_trainer_initialization_event() -> None:
    from osm_polygon_sentence_classifier import training_publication

    control = object()

    result = training_publication.CheckpointManifestCallback({}).on_init_end(
        args=SimpleNamespace(),
        state=SimpleNamespace(),
        control=control,
    )

    assert result is control


def test_checkpoint_callback_uses_the_trainer_callback_base() -> None:
    dependencies = training_runtime.TrainingDependencies(
        iterable_dataset=_FakeDataset,
        auto_tokenizer=_FakeTokenizer,
        auto_model_for_sequence_classification=_FakeModel,
        data_collator_with_padding=_FakeDataCollator,
        training_arguments=_FakeTrainingArguments,
        trainer=_FakeTrainer,
        trainer_callback=_FakeTrainerCallback,
    )
    _FakeTrainer.init_calls.clear()

    training_runtime.build_trainer(
        dependencies,
        model=object(),
        training_arguments=object(),
        train_dataset=object(),
        validation_dataset=object(),
        data_collator=object(),
        checkpoint_identity={},
    )

    callbacks = _FakeTrainer.init_calls[0]["callbacks"]
    assert isinstance(callbacks, list)
    callback = callbacks[0]
    assert isinstance(callback, _FakeTrainerCallback)
    control = object()
    assert (
        callback.on_train_begin(SimpleNamespace(), SimpleNamespace(), control)
        is control
    )


def test_training_reports_missing_optional_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_dependency(module_name: str) -> Any:
        raise ModuleNotFoundError(name=module_name)

    monkeypatch.setattr(training_runtime, "import_module", missing_dependency)

    with pytest.raises(TrainingError, match="optional 'training' dependencies"):
        training_runtime.load_training_dependencies()


def test_training_passes_a_pinned_model_revision_to_both_loaders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_classifier import training

    _FakeDataset.created.clear()
    _FakeTokenizer.from_pretrained_calls.clear()
    _FakeTokenizer.save_pretrained_calls.clear()
    _FakeModel.from_pretrained_calls.clear()
    _FakeModel.instances.clear()
    _FakeTrainingArguments.calls.clear()
    _FakeDataCollator.calls.clear()
    _FakeTrainer.init_calls.clear()
    _FakeTrainer.save_model_calls.clear()

    dependencies = training_runtime.TrainingDependencies(
        iterable_dataset=_FakeDataset,
        auto_tokenizer=_FakeTokenizer,
        auto_model_for_sequence_classification=_FakeModel,
        data_collator_with_padding=_FakeDataCollator,
        training_arguments=_FakeTrainingArguments,
        trainer=_FakeTrainer,
    )
    monkeypatch.setattr(
        training_runtime, "load_training_dependencies", lambda: dependencies
    )
    monkeypatch.setattr(training, "_write_model_card", lambda *args, **kwargs: None)
    revision = "d" * 40

    train_landuse_classifier(
        rows_factory=lambda: iter([_row()]),
        config=TrainingConfig(
            model_name_or_path="test-model",
            model_revision=revision,
        ),
    )

    assert _FakeTokenizer.from_pretrained_calls[0]["revision"] == revision
    assert _FakeModel.from_pretrained_calls[0]["revision"] == revision


def test_training_can_publish_the_final_model_and_sync_static_trackio(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_classifier import training

    _FakeDataset.created.clear()
    _FakeTokenizer.from_pretrained_calls.clear()
    _FakeTokenizer.save_pretrained_calls.clear()
    _FakeModel.from_pretrained_calls.clear()
    _FakeModel.instances.clear()
    _FakeTrainingArguments.calls.clear()
    _FakeDataCollator.calls.clear()
    _FakeTrainer.init_calls.clear()
    _FakeTrainer.save_model_calls.clear()

    dependencies = training_runtime.TrainingDependencies(
        iterable_dataset=_FakeDataset,
        auto_tokenizer=_FakeTokenizer,
        auto_model_for_sequence_classification=_FakeModel,
        data_collator_with_padding=_FakeDataCollator,
        training_arguments=_FakeTrainingArguments,
        trainer=_FakeTrainer,
    )
    monkeypatch.setattr(
        training_runtime, "load_training_dependencies", lambda: dependencies
    )
    checkpoint_hub_api = object()
    monkeypatch.setattr(
        training_runtime,
        "load_checkpoint_publication_api",
        lambda: checkpoint_hub_api,
    )
    publication_calls: list[Path] = []
    monkeypatch.setattr(
        training,
        "publish_model_directory",
        lambda directory, repository_id, **kwargs: publication_calls.append(directory)
        or training.ModelPublicationResult(
            repository_id=repository_id,
            commit_id="d" * 40,
            commit_url="https://huggingface.co/test/commit/" + "d" * 40,
            files=("config.json", "model.safetensors"),
        ),
    )
    sync_calls: list[tuple[object, dict[str, object]]] = []
    monkeypatch.setattr(
        training,
        "sync_project_to_static_space",
        lambda settings, **kwargs: sync_calls.append((settings, kwargs))
        or TRACKIO_STATIC_SPACE_ID,
    )
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    result = training.train_landuse_classifier(
        rows_factory=lambda: iter([_row()]),
        config=TrainingConfig(
            model_name_or_path="test-model",
            publish_to_hub=True,
            sync_trackio=True,
        ),
        project_config=ProjectConfig.for_remote_root(tmp_path / "data"),
        checkpoint_identity={"run_id": "a" * 20, "model_revision": "b" * 40},
    )

    project_config = ProjectConfig.for_remote_root(tmp_path / "data")
    assert publication_calls == [project_config.data_root / "models/landuse"]
    arguments = _FakeTrainingArguments.calls[0]
    assert arguments["trackio_space_id"] is None
    assert arguments["trackio_bucket_id"] is None
    callbacks = cast(list[Any], _FakeTrainer.init_calls[0]["callbacks"])
    callback = callbacks[0]
    assert callback.model_repository_id == ProjectConfig().target_model_repository_id
    assert callback.trackio_space_id == TRACKIO_STATIC_SPACE_ID
    assert callback.tracking_settings is not None
    assert callback.tracking_settings.static_space_id == TRACKIO_STATIC_SPACE_ID
    assert callback.hub_api is checkpoint_hub_api
    assert callback.hub_checkpoint_steps == 1_000
    assert result.model_publication is not None
    assert result.model_publication.commit_id == "d" * 40
    assert result.tracking_space_id == TRACKIO_STATIC_SPACE_ID
    assert len(sync_calls) == 1
    assert sync_calls[0][1] == {"finalize": True}


def test_prepare_checkpoint_resume_forwards_the_selected_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from osm_polygon_sentence_classifier import training

    checkpoint = tmp_path / "checkpoint-100"
    identity = {"run_id": "a" * 20}
    calls: dict[str, object] = {}

    def validate(
        output_directory: Path,
        *,
        resume_from_checkpoint: Path,
        identity: Mapping[str, object],
    ) -> None:
        calls.update(
            output_directory=output_directory,
            resume_from_checkpoint=resume_from_checkpoint,
            identity=identity,
        )

    resume_validation_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        training,
        "_validate_checkpoint_resume_arguments",
        lambda **kwargs: resume_validation_calls.append(kwargs),
    )
    monkeypatch.setattr(training, "_validate_requested_checkpoint", validate)

    result = training._prepare_checkpoint_resume(
        tmp_path,
        resume_from_checkpoint=checkpoint,
        checkpoint_identity=identity,
    )

    assert result == (checkpoint, identity)
    assert calls == {
        "output_directory": tmp_path,
        "resume_from_checkpoint": checkpoint,
        "identity": identity,
    }
    assert resume_validation_calls == [
        {
            "resume_from_checkpoint": checkpoint,
            "checkpoint_identity": identity,
        }
    ]


def test_requested_checkpoint_rejects_a_complete_but_different_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from osm_polygon_sentence_classifier import training

    selected = SimpleNamespace(path=tmp_path / "checkpoint-200")
    monkeypatch.setattr(
        training,
        "find_latest_complete_checkpoint",
        lambda *_args, **_kwargs: selected,
    )

    with pytest.raises(TrainingError) as error:
        training._validate_requested_checkpoint(
            tmp_path,
            resume_from_checkpoint=tmp_path / "checkpoint-100",
            identity={"run_id": "a" * 20},
        )

    assert str(error.value) == "requested checkpoint is not a complete identity match"


def test_training_environment_values_forwards_the_tracking_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_classifier import training
    from osm_polygon_sentence_classifier.paths import ManagedPaths

    project_config = ProjectConfig()
    observed: dict[str, object] = {}

    class Settings:
        def environment(self) -> dict[str, str]:
            return {"TRACKIO_DIR": "tracking"}

    def fake_settings(config: ProjectConfig, *, project: str | None) -> Settings:
        observed.update(config=config, project=project)
        return Settings()

    monkeypatch.setattr(training, "settings_for", fake_settings)
    monkeypatch.setattr(training, "_load_hugging_face_token", lambda: "token")

    values = training._training_environment_values(
        project_config,
        paths=ManagedPaths(project_config),
        tracking_project="tracked-project",
    )

    assert observed == {"config": project_config, "project": "tracked-project"}
    assert values == {
        "HF_HOME": str(project_config.data_root / "cache/huggingface"),
        "TRACKIO_DIR": "tracking",
        "HF_TOKEN": "token",
    }


def test_restore_environment_tolerates_a_previously_absent_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_classifier import training

    variable = "OSM_SENTENCE_CLASSIFIER_TEST_ABSENT"
    monkeypatch.delenv(variable, raising=False)

    training._restore_environment({variable: None})

    assert variable not in os.environ


def test_restore_environment_restores_a_previously_present_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_classifier import training

    variable = "OSM_SENTENCE_CLASSIFIER_TEST_PRESENT"
    monkeypatch.setenv(variable, "current")

    training._restore_environment({variable: "previous"})

    assert os.environ[variable] == "previous"


def test_repository_readme_is_available_for_a_v2_namespaced_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_classifier import training

    config = TrainingConfig(
        model_name_or_path="test-model",
        artifact_namespace="studies/place-relevance-v2/baseline",
        sync_trackio=True,
    )
    tracking = TrackioSettings(
        project="project",
        directory=Path("tracking"),
        static_space_id="static-space",
    )
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        training,
        "render_repository_readme",
        lambda **kwargs: observed.update(kwargs) or "README",
    )

    assert (
        training._repository_readme(
            config=config,
            contract=WORLDWIDE_PLACE_RELEVANCE_V2_CONTRACT,
            identity={"run_id": "a" * 20},
            tracking=tracking,
        )
        == "README"
    )
    assert observed["trackio_space_id"] == "static-space"


def test_repository_readme_is_suppressed_for_a_namespaced_landuse_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_classifier import training

    monkeypatch.setattr(
        training,
        "render_repository_readme",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("landuse study README should be omitted")
        ),
    )

    assert (
        training._repository_readme(
            config=TrainingConfig(artifact_namespace="studies/landuse-v1/a01-head-128"),
            contract=LANDUSE_DATASET_CONTRACT,
            identity={"run_id": "a" * 20},
            tracking=TrackioSettings(project="project", directory=Path("tracking")),
        )
        is None
    )


def test_effective_rows_factory_preserves_inputs_and_default_loader_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_classifier import training

    project_config = ProjectConfig()
    contract = WORLDWIDE_PLACE_RELEVANCE_V2_CONTRACT

    def supplied() -> Iterator[Mapping[str, object]]:
        return iter(())

    assert (
        training._effective_rows_factory(
            supplied,
            project_config=project_config,
            contract=contract,
        )
        is supplied
    )

    observed: dict[str, object] = {}

    def fake_load_rows(**kwargs: object) -> Iterable[Mapping[str, object]]:
        observed.update(kwargs)
        return iter(({"sentence": "row"},))

    monkeypatch.setattr(training, "load_streaming_rows", fake_load_rows)
    default_factory = training._effective_rows_factory(
        None,
        project_config=project_config,
        contract=contract,
    )

    assert tuple(default_factory()) == ({"sentence": "row"},)
    assert observed == {"config": project_config, "contract": contract}


def test_training_context_forwards_tracking_and_rows_factory_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_classifier import training

    project_config = ProjectConfig()
    config = TrainingConfig(
        output_subdirectory=Path("models/context"),
        tracking_project="tracked-project",
    )
    tracking = TrackioSettings(project="tracked-project", directory=Path("tracking"))

    def factory() -> Iterator[Mapping[str, object]]:
        return iter(())

    calls: dict[str, object] = {}

    def fake_settings(
        received_config: ProjectConfig, *, project: str | None
    ) -> TrackioSettings:
        calls.update(config=received_config, project=project)
        return tracking

    def fake_rows_factory(
        received_factory: Callable[[], Iterable[Mapping[str, object]]] | None,
        *,
        project_config: ProjectConfig,
        contract: DatasetContract,
    ) -> Callable[[], Iterable[Mapping[str, object]]]:
        calls.update(
            rows_factory=received_factory,
            rows_project_config=project_config,
            rows_contract=contract,
        )
        return factory

    monkeypatch.setattr(training, "settings_for", fake_settings)
    monkeypatch.setattr(training, "_effective_rows_factory", fake_rows_factory)

    context = training._training_context(
        factory,
        config=config,
        project_config=project_config,
        contract=LANDUSE_DATASET_CONTRACT,
        resume_from_checkpoint=None,
        checkpoint_identity=None,
    )

    assert context.rows_factory is factory
    assert context.tracking is tracking
    assert calls == {
        "config": project_config,
        "project": "tracked-project",
        "rows_factory": factory,
        "rows_project_config": project_config,
        "rows_contract": LANDUSE_DATASET_CONTRACT,
    }


def test_restore_tracking_snapshot_forwards_the_context_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_classifier import training

    tracking = TrackioSettings(project="tracked-project", directory=Path("tracking"))
    context = SimpleNamespace(
        config=TrainingConfig(sync_trackio=True, tracking_project="tracked-project"),
        tracking=tracking,
    )
    calls: list[object] = []
    monkeypatch.setattr(
        training,
        "restore_static_project_snapshot",
        lambda received: calls.append(received),
    )

    training._restore_tracking_snapshot_if_needed(cast(Any, context))

    assert calls == [tracking]


@pytest.mark.parametrize(
    ("sync_trackio", "tracking_project"),
    [(False, "tracked-project"), (True, None), (False, None)],
)
def test_restore_tracking_snapshot_requires_both_context_settings(
    monkeypatch: pytest.MonkeyPatch,
    sync_trackio: bool,
    tracking_project: str | None,
) -> None:
    from osm_polygon_sentence_classifier import training

    tracking = object()
    context = SimpleNamespace(
        config=TrainingConfig(
            sync_trackio=sync_trackio,
            tracking_project=tracking_project,
        ),
        tracking=tracking,
    )
    calls: list[object] = []
    monkeypatch.setattr(
        training,
        "restore_static_project_snapshot",
        lambda received: calls.append(received),
    )

    training._restore_tracking_snapshot_if_needed(cast(Any, context))

    assert calls == []


def test_model_kwargs_preserve_the_complete_model_identity_contract() -> None:
    from osm_polygon_sentence_classifier import training

    revision = "b" * 40
    config = TrainingConfig(model_name_or_path="test-model", model_revision=revision)

    assert training._model_kwargs(config, Path("cache")) == {
        "cache_dir": "cache",
        "classifier_dropout": 0.0,
        "num_labels": 2,
        "id2label": {0: "no", 1: "yes"},
        "label2id": {"no": 0, "yes": 1},
        "revision": revision,
    }


def test_load_model_forwards_cache_and_trainable_layer_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_classifier import training

    model = object()
    model_calls: list[tuple[object, dict[str, object]]] = []
    configured: list[tuple[object, object]] = []

    def load_model(name: object, **kwargs: object) -> object:
        model_calls.append((name, kwargs))
        return model

    dependencies = SimpleNamespace(
        auto_model_for_sequence_classification=SimpleNamespace(
            from_pretrained=load_model
        )
    )
    context = SimpleNamespace(
        config=TrainingConfig(model_name_or_path="test-model", trainable_layers="head"),
        model_cache_directory=Path("cache"),
    )
    monkeypatch.setattr(
        training,
        "configure_trainable_layers",
        lambda received_model, layers: configured.append((received_model, layers)),
    )

    assert training._load_model(cast(Any, dependencies), cast(Any, context)) is model
    assert model_calls[0][0] == "test-model"
    assert model_calls[0][1]["cache_dir"] == "cache"
    assert configured == [(model, "head")]


def test_checkpoint_publication_api_is_disabled_without_hub_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_classifier import training

    context = SimpleNamespace(
        checkpoint_identity={"run_id": "a" * 20},
        config=TrainingConfig(publish_to_hub=False),
    )
    monkeypatch.setattr(
        training._training_runtime,
        "load_checkpoint_publication_api",
        lambda: (_ for _ in ()).throw(
            AssertionError("checkpoint publication API should not be loaded")
        ),
    )

    assert training._checkpoint_publication_api(cast(Any, context)) is None


def test_sync_static_trackio_defaults_to_a_non_final_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_classifier import training

    settings = TrackioSettings(project="tracked-project", directory=Path("tracking"))
    calls: list[tuple[object, bool]] = []
    monkeypatch.setattr(
        training,
        "sync_project_to_static_space",
        lambda received, *, finalize: calls.append((received, finalize)),
    )

    training._sync_static_trackio(settings, failure_message="sync failed")

    assert calls == [(settings, False)]


def test_sync_static_trackio_preserves_the_failure_message_and_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_classifier import training

    cause = TrackingError("temporary tracking failure")
    monkeypatch.setattr(
        training,
        "sync_project_to_static_space",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(cause),
    )

    with pytest.raises(TrainingError) as error:
        training._sync_static_trackio(
            TrackioSettings(project="project", directory=Path("tracking")),
            failure_message="sync failed",
        )

    assert str(error.value) == "sync failed"
    assert error.value.__cause__ is cause


def test_build_training_trainer_forwards_the_complete_runtime_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_classifier import training

    config = TrainingConfig(
        publish_to_hub=True,
        sync_trackio=True,
        tracking_project="tracked-project",
        class_weight_mode="balanced",
        hub_checkpoint_steps=1_000,
    )
    tracking = TrackioSettings(
        project="tracked-project",
        directory=Path("tracking"),
        static_space_id="static-space",
    )
    context = training._TrainingContext(
        config=config,
        project_config=ProjectConfig(),
        contract=LANDUSE_DATASET_CONTRACT,
        rows_factory=lambda: iter(()),
        output_directory=Path("output"),
        model_cache_directory=Path("cache"),
        tracking=tracking,
        resume_from_checkpoint=None,
        checkpoint_identity={"run_id": "a" * 20},
    )
    training_arguments = object()
    collator = object()
    model = object()
    tokenizer = object()
    train_dataset = object()
    validation_dataset = object()
    checkpoint_hub_api = object()
    calls: dict[str, object] = {}

    dependencies = SimpleNamespace(
        training_arguments=lambda **kwargs: calls.update(
            training_arguments_kwargs=kwargs
        )
        or training_arguments,
        data_collator_with_padding=lambda *, tokenizer: calls.update(
            collator_tokenizer=tokenizer
        )
        or collator,
    )

    def build(received_dependencies: object, **kwargs: object) -> object:
        calls.update(dependencies=received_dependencies, trainer_kwargs=kwargs)
        return "trainer"

    monkeypatch.setattr(training._training_runtime, "build_trainer", build)

    assert (
        training._build_training_trainer(
            cast(Any, dependencies),
            context=cast(Any, context),
            model=cast(Any, model),
            tokenizer=cast(Any, tokenizer),
            train_dataset=cast(Any, train_dataset),
            validation_dataset=cast(Any, validation_dataset),
            checkpoint_hub_api=cast(Any, checkpoint_hub_api),
        )
        == "trainer"
    )
    assert calls["dependencies"] is dependencies
    assert calls["collator_tokenizer"] is tokenizer
    trainer_kwargs = cast(dict[str, object], calls["trainer_kwargs"])
    assert trainer_kwargs == {
        "model": model,
        "training_arguments": training_arguments,
        "train_dataset": train_dataset,
        "validation_dataset": validation_dataset,
        "data_collator": collator,
        "checkpoint_identity": context.checkpoint_identity,
        "model_repository_id": ProjectConfig().target_model_repository_id,
        "trackio_space_id": "static-space",
        "tracking_settings": tracking,
        "hub_api": checkpoint_hub_api,
        "hub_checkpoint_steps": 1_000,
        "class_weight_mode": "balanced",
    }


def test_finalize_training_outputs_preserves_metrics_and_model_card_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from osm_polygon_sentence_classifier import training

    config = TrainingConfig(
        model_name_or_path="test-model",
        sync_trackio=True,
        tracking_project="tracked-project",
    )
    tracking = TrackioSettings(
        project="tracked-project",
        directory=tmp_path / "tracking",
        static_space_id="static-space",
    )
    context = training._TrainingContext(
        config=config,
        project_config=ProjectConfig(),
        contract=LANDUSE_DATASET_CONTRACT,
        rows_factory=lambda: iter(()),
        output_directory=tmp_path / "output",
        model_cache_directory=tmp_path / "cache",
        tracking=tracking,
        resume_from_checkpoint=tmp_path / "checkpoint-100",
        checkpoint_identity={"checkpoint": 100},
    )
    train_output = object()
    model_card_identity = {"run_id": "a" * 20}
    metrics = {"train_loss": 0.2}
    trainer_calls: list[tuple[object, object]] = []
    save_model_calls: list[str] = []
    save_tokenizer_calls: list[str] = []
    metric_calls: list[tuple[object, object]] = []
    identity_calls: list[tuple[object, object, object]] = []
    model_card_calls: list[dict[str, object]] = []

    trainer = SimpleNamespace(
        save_model=lambda directory: save_model_calls.append(directory),
    )
    tokenizer = SimpleNamespace(
        save_pretrained=lambda directory: save_tokenizer_calls.append(directory),
    )
    monkeypatch.setattr(
        training._training_runtime,
        "run_trainer",
        lambda received_trainer, resume: trainer_calls.append(
            (received_trainer, resume)
        )
        or train_output,
    )
    monkeypatch.setattr(
        training._training_metrics,
        "metrics_for_model_card",
        lambda received_output, received_trainer: metric_calls.append(
            (received_output, received_trainer)
        )
        or dict(metrics),
    )
    monkeypatch.setattr(
        training,
        "_evaluate_validation_dataset",
        lambda received_trainer, dataset: {"eval_accuracy": 0.8},
    )
    monkeypatch.setattr(
        training,
        "_evaluate_test_dataset",
        lambda received_trainer, dataset: {"test_accuracy": 0.7},
    )
    monkeypatch.setattr(
        training,
        "_model_card_identity",
        lambda received_identity, *, config, contract: identity_calls.append(
            (received_identity, config, contract)
        )
        or model_card_identity,
    )
    monkeypatch.setattr(
        training,
        "_write_model_card",
        lambda directory, **kwargs: model_card_calls.append(
            {"directory": directory, **kwargs}
        ),
    )

    result = training._finalize_training_outputs(
        trainer,
        context=context,
        tokenizer=tokenizer,
        validation_dataset="validation",
        test_dataset="test",
    )

    assert result == (
        train_output,
        {"train_loss": 0.2, "eval_accuracy": 0.8, "test_accuracy": 0.7},
        model_card_identity,
    )
    assert trainer_calls == [(trainer, context.resume_from_checkpoint)]
    assert metric_calls == [(train_output, trainer)]
    assert save_model_calls == [str(context.output_directory)]
    assert save_tokenizer_calls == [str(context.output_directory)]
    assert identity_calls == [
        (context.checkpoint_identity, config, LANDUSE_DATASET_CONTRACT)
    ]
    assert model_card_calls == [
        {
            "directory": context.output_directory,
            "identity": model_card_identity,
            "training_metrics": {
                "train_loss": 0.2,
                "eval_accuracy": 0.8,
                "test_accuracy": 0.7,
            },
            "trackio_space_id": "static-space",
        }
    ]


def test_worldwide_v2_document_publication_forwards_identity_metrics_and_hub_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_classifier import training

    config = TrainingConfig(sync_trackio=True)
    tracking = TrackioSettings(project="project", directory=Path("tracking"))
    identity = {"run_id": "a" * 20}
    metrics = {"eval_f1": 0.8}
    hub_api = object()
    rendered: dict[str, object] = {}
    published: dict[str, object] = {}
    monkeypatch.setattr(
        training,
        "render_place_relevance_study_documents",
        lambda **kwargs: rendered.update(kwargs) or ("README.md",),
    )
    monkeypatch.setattr(
        training,
        "publish_study_documents",
        lambda repository, documents, *, hub_api: published.update(
            repository=repository, documents=documents, hub_api=hub_api
        ),
    )

    training._publish_worldwide_v2_documents(
        ProjectConfig(),
        config=config,
        identity=identity,
        metrics=metrics,
        tracking=tracking,
        checkpoint_hub_api=hub_api,
    )

    assert rendered["identity"] is identity
    assert rendered["metrics"] is metrics
    assert published["hub_api"] is hub_api


def test_worldwide_v2_document_publication_keeps_the_exact_error_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_classifier import training
    from osm_polygon_sentence_classifier.publication import ModelPublicationError

    monkeypatch.setattr(
        training,
        "publish_study_documents",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ModelPublicationError("hub unavailable")
        ),
    )

    with pytest.raises(TrainingError) as error:
        training._publish_worldwide_v2_documents(
            ProjectConfig(),
            config=TrainingConfig(),
            identity={"run_id": "a" * 20},
            metrics={},
            tracking=TrackioSettings(project="project", directory=Path("tracking")),
            checkpoint_hub_api=object(),
        )

    assert str(error.value) == "worldwide V2 study documentation publication failed"


def test_landuse_entry_point_forwards_its_complete_classifier_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_classifier import training

    def rows_factory() -> Iterator[Mapping[str, object]]:
        return iter(())

    config = TrainingConfig(model_name_or_path="test-model")
    project_config = ProjectConfig()
    checkpoint = Path("checkpoint")
    identity = {"run_id": "a" * 20}
    expected = object()
    received: dict[str, object] = {}

    def fake_train(*args: object, **kwargs: object) -> object:
        received["args"] = args
        received["kwargs"] = kwargs
        return expected

    monkeypatch.setattr(training, "_train_classifier", fake_train)

    result = training.train_landuse_classifier(
        rows_factory,
        config=config,
        project_config=project_config,
        contract=LANDUSE_DATASET_CONTRACT,
        resume_from_checkpoint=checkpoint,
        checkpoint_identity=identity,
    )

    assert result is expected
    assert received == {
        "args": (rows_factory,),
        "kwargs": {
            "config": config,
            "project_config": project_config,
            "contract": LANDUSE_DATASET_CONTRACT,
            "resume_from_checkpoint": checkpoint,
            "checkpoint_identity": identity,
        },
    }
