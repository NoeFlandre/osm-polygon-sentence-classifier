"""Managed, streaming training orchestration for binary classifiers."""

import math
import os
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Literal, TypedDict

from . import training_metrics as _training_metrics
from . import training_runtime as _training_runtime
from .checkpointing import CheckpointError, find_latest_complete_checkpoint
from .config import ProjectConfig
from .dataset_contract import (
    LANDUSE_DATASET_CONTRACT,
    WORLDWIDE_PLACE_RELEVANCE_V2_CONTRACT,
    DatasetContract,
)
from .dataset_loader import (
    DatasetSplit,
    TrainingLabel,
    iter_clean_training_examples,
    load_streaming_rows,
)
from .paths import ManagedPaths
from .place_relevance_reporting import render_place_relevance_study_documents
from .publication import (
    ModelPublicationError,
    ModelPublicationResult,
    publish_model_directory,
    publish_study_documents,
    render_repository_readme,
)
from .tracking import (
    TrackingError,
    TrackioSettings,
    restore_static_project_snapshot,
    settings_for,
    sync_project_to_static_space,
)
from .training_freezing import (
    TrainableLayers,
    TrainingError,
    configure_trainable_layers,
)
from .training_publication import write_model_card as _write_model_card
from .training_runtime import ClassWeightMode

LabelId = Literal[0, 1]

LABEL_TO_ID: dict[TrainingLabel, LabelId] = {"no": 0, "yes": 1}
ID_TO_LABEL: dict[int, str] = {0: "no", 1: "yes"}
DEFAULT_MODEL_NAME = "jhu-clsp/mmBERT-small"
# One logical streamed epoch over the pinned, audited clean training split:
# ceil(141,283 train representatives / batch size 8).
PLACE_RELEVANCE_V2_DEFAULT_MAX_STEPS = 17_661
PLACE_RELEVANCE_V2_OUTPUT = Path("studies/place-relevance-v2/baseline/models")
PLACE_RELEVANCE_V2_RUN_NAME = "place-relevance-v2|baseline|seed-42"


class TrainingRecord(TypedDict):
    """One clean, split-assigned record before tokenization."""

    text: str
    labels: LabelId


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Small, explicit configuration for one binary training run."""

    model_name_or_path: str = DEFAULT_MODEL_NAME
    output_subdirectory: Path = Path("models/landuse")
    validation_fraction: float = 0.2
    test_fraction: float = 0.0
    seed: int = 42
    max_length: int = 256
    max_steps: int = 1_000
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    learning_rate: float = 3e-4
    logging_steps: int = 10
    eval_steps: int = 100
    eval_strategy: Literal["steps", "epoch"] = "steps"
    save_steps: int = 100
    save_total_limit: int = 5
    hub_checkpoint_steps: int = 1_000
    run_name: str = "landuse-mmbert-small-frozen-head"
    model_revision: str | None = None
    trainable_layers: TrainableLayers | None = None
    class_weight_mode: ClassWeightMode | None = None
    tracking_project: str | None = None
    artifact_namespace: str | None = None
    publish_to_hub: bool = False
    sync_trackio: bool = False

    def __post_init__(self) -> None:
        _validate_training_identity(self)
        _validate_boolean_settings(self)
        _normalize_training_paths(self)
        _validate_numeric_settings(self)


def place_relevance_v2_training_config(
    *,
    model_name_or_path: str = DEFAULT_MODEL_NAME,
    model_revision: str | None = None,
    max_steps: int = PLACE_RELEVANCE_V2_DEFAULT_MAX_STEPS,
    publish_to_hub: bool = False,
    sync_trackio: bool = False,
) -> TrainingConfig:
    """Return the pinned baseline configuration for the worldwide V2 task."""

    return TrainingConfig(
        model_name_or_path=model_name_or_path,
        model_revision=model_revision,
        output_subdirectory=PLACE_RELEVANCE_V2_OUTPUT,
        validation_fraction=0.1,
        test_fraction=0.1,
        eval_strategy="epoch",
        max_steps=max_steps,
        trainable_layers="head",
        run_name=PLACE_RELEVANCE_V2_RUN_NAME,
        tracking_project="place-relevance-v2",
        artifact_namespace="studies/place-relevance-v2/baseline",
        publish_to_hub=publish_to_hub,
        sync_trackio=sync_trackio,
    )


def _validate_boolean_settings(config: TrainingConfig) -> None:
    for name in ("publish_to_hub", "sync_trackio"):
        if not isinstance(getattr(config, name), bool):
            raise TrainingError(f"{name} must be a boolean")


def _validate_training_identity(config: TrainingConfig) -> None:
    _validate_model_name(config.model_name_or_path)
    _validate_run_name(config.run_name)
    _validate_model_revision(config.model_revision)
    _validate_training_modes(config)
    _validate_tracking_names(config)


def _validate_model_name(value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise TrainingError("model_name_or_path must be a non-empty string")


def _validate_run_name(value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise TrainingError("run_name must be a non-empty string")


def _validate_model_revision(value: object) -> None:
    if value is not None and (
        not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value) is None
    ):
        raise TrainingError(
            "model_revision must be exactly 40 lowercase hexadecimal characters"
        )


def _validate_training_modes(config: TrainingConfig) -> None:
    if config.trainable_layers not in {None, "head", "last2"}:
        raise TrainingError("trainable_layers must be head or last2")
    if config.class_weight_mode not in {None, "none", "balanced"}:
        raise TrainingError("class_weight_mode must be none or balanced")
    if config.eval_strategy not in {"steps", "epoch"}:
        raise TrainingError("eval_strategy must be steps or epoch")


def _validate_tracking_names(config: TrainingConfig) -> None:
    for name in ("tracking_project", "artifact_namespace"):
        value = getattr(config, name)
        if value is not None:
            _validate_tracking_name(name, value)


def _validate_tracking_name(name: str, value: object) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\n" in value
        or "\r" in value
    ):
        raise TrainingError(f"{name} must be a non-empty single-line string")


def _normalize_training_paths(config: TrainingConfig) -> None:
    output_subdirectory = Path(config.output_subdirectory)
    if not _is_clean_relative_path(output_subdirectory):
        raise TrainingError(
            "output_subdirectory must be a clean relative path beneath the "
            "managed data root"
        )
    object.__setattr__(config, "output_subdirectory", output_subdirectory)

    if config.artifact_namespace is not None:
        namespace = Path(config.artifact_namespace)
        if not _is_clean_relative_path(namespace):
            raise TrainingError("artifact_namespace must be a clean relative path")
        object.__setattr__(config, "artifact_namespace", namespace.as_posix())


def _is_clean_relative_path(path: Path) -> bool:
    return (
        bool(path.parts)
        and not path.is_absolute()
        and not any(part in (".", "..") for part in path.parts)
    )


def _validate_numeric_settings(config: TrainingConfig) -> None:
    _validate_fraction_settings(config)
    _validate_positive_integer_settings(config)
    _validate_learning_rate(config.learning_rate)


def _is_finite_fraction(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and 0 <= value <= 1
    )


def _validate_fraction_settings(config: TrainingConfig) -> None:
    if not _is_finite_fraction(config.validation_fraction):
        raise TrainingError(
            "validation_fraction must be a finite number between 0 and 1"
        )
    if not _is_finite_fraction(config.test_fraction) or (
        config.validation_fraction + config.test_fraction > 1
    ):
        raise TrainingError(
            "validation and test fractions must be finite, non-negative, "
            "and sum to at most 1"
        )


def _validate_positive_integer_settings(config: TrainingConfig) -> None:
    for name in (
        "max_length",
        "max_steps",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "logging_steps",
        "eval_steps",
        "save_steps",
        "save_total_limit",
        "hub_checkpoint_steps",
    ):
        value = getattr(config, name)
        _validate_positive_integer(name, value)
    _validate_checkpoint_interval(config)


def _validate_positive_integer(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TrainingError(f"{name} must be a positive integer")


def _validate_checkpoint_interval(config: TrainingConfig) -> None:
    if config.hub_checkpoint_steps % config.save_steps != 0:
        raise TrainingError("hub_checkpoint_steps must be a multiple of save_steps")


def _validate_learning_rate(value: object) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise TrainingError("learning_rate must be a positive finite number")


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Local result of a completed Trainer run."""

    output_directory: Path
    train_output: object
    model_publication: ModelPublicationResult | None = None
    tracking_space_id: str | None = None
    metrics: Mapping[str, object] | None = None


def _training_config_payload(config: TrainingConfig) -> dict[str, object]:
    payload: dict[str, object] = {}
    for item in fields(config):
        value = getattr(config, item.name)
        if (
            item.name
            in {
                "trainable_layers",
                "class_weight_mode",
                "tracking_project",
                "artifact_namespace",
            }
            and value is None
        ):
            continue
        payload[item.name] = str(value) if isinstance(value, Path) else value
    return payload


def _model_card_identity(
    identity: Mapping[str, object] | None,
    *,
    config: TrainingConfig,
    contract: DatasetContract,
) -> dict[str, object]:
    payload = dict(identity or {})
    payload.setdefault(
        "task_name",
        "place-relevance-v2"
        if contract is WORLDWIDE_PLACE_RELEVANCE_V2_CONTRACT
        else "landuse",
    )
    payload.setdefault("dataset_revision", contract.provenance.repository_revision)
    payload.setdefault("model_name_or_path", config.model_name_or_path)
    payload.setdefault("model_revision", config.model_revision)
    if not isinstance(payload.get("training_config"), Mapping):
        payload["training_config"] = _training_config_payload(config)
    return payload


def _sync_static_trackio(
    settings: TrackioSettings,
    *,
    failure_message: str,
    finalize: bool = False,
) -> None:
    try:
        sync_project_to_static_space(settings, finalize=finalize)
    except TrackingError as error:
        raise TrainingError(failure_message) from error


def _prepare_checkpoint_resume(
    output_directory: Path,
    *,
    resume_from_checkpoint: Path | None,
    checkpoint_identity: Mapping[str, object] | None,
) -> tuple[Path | None, dict[str, object] | None]:
    _validate_checkpoint_resume_arguments(
        resume_from_checkpoint=resume_from_checkpoint,
        checkpoint_identity=checkpoint_identity,
    )
    if checkpoint_identity is None:
        return resume_from_checkpoint, None
    identity = dict(checkpoint_identity)
    if resume_from_checkpoint is None:
        return None, identity
    _validate_requested_checkpoint(
        output_directory,
        resume_from_checkpoint=resume_from_checkpoint,
        identity=identity,
    )
    return resume_from_checkpoint, identity


def _validate_checkpoint_resume_arguments(
    *,
    resume_from_checkpoint: Path | None,
    checkpoint_identity: Mapping[str, object] | None,
) -> None:
    if resume_from_checkpoint is not None and checkpoint_identity is None:
        raise TrainingError("checkpoint identity is required for resume")


def _validate_requested_checkpoint(
    output_directory: Path,
    *,
    resume_from_checkpoint: Path,
    identity: Mapping[str, object],
) -> None:
    try:
        selected = find_latest_complete_checkpoint(
            output_directory,
            identity=dict(identity),
        )
    except CheckpointError as error:
        raise TrainingError("checkpoint evidence is invalid") from error
    if selected is None or selected.path != resume_from_checkpoint:
        raise TrainingError("requested checkpoint is not a complete identity match")


def iter_split_training_records(
    rows_factory: Callable[[], Iterable[Mapping[str, object]]],
    *,
    split: DatasetSplit,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.0,
    seed: int = 42,
    contract: DatasetContract = LANDUSE_DATASET_CONTRACT,
) -> Iterator[TrainingRecord]:
    """Lazily adapt clean examples to Trainer record dictionaries."""

    if split not in ("train", "validation", "test"):
        raise TrainingError(f"unsupported dataset split: {split!r}")
    for example in iter_clean_training_examples(
        rows_factory,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
        contract=contract,
    ):
        if example.split != split:
            continue
        yield {"text": example.text, "labels": LABEL_TO_ID[example.label]}


def _make_tokenized_dataset(
    dependencies: _training_runtime.TrainingDependencies,
    rows_factory: Callable[[], Iterable[Mapping[str, object]]],
    *,
    split: DatasetSplit,
    config: TrainingConfig,
    contract: DatasetContract,
    tokenizer: Any,
) -> Any:
    def generate() -> Iterator[TrainingRecord]:
        yield from iter_split_training_records(
            rows_factory,
            split=split,
            validation_fraction=config.validation_fraction,
            test_fraction=config.test_fraction,
            seed=config.seed,
            contract=contract,
        )

    dataset = dependencies.iterable_dataset.from_generator(generate)

    def tokenize(batch: Mapping[str, Any]) -> Mapping[str, Any]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=config.max_length,
        )

    return dataset.map(tokenize, batched=True, remove_columns=["text"])


def _training_argument_values(
    config: TrainingConfig,
    *,
    output_directory: Path,
    tracking_project: str,
    trackio_space_id: str | None,
    trackio_bucket_id: str | None,
) -> dict[str, object]:
    return {
        "output_dir": str(output_directory),
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "learning_rate": config.learning_rate,
        "max_steps": config.max_steps,
        "seed": config.seed,
        "logging_steps": config.logging_steps,
        "eval_strategy": config.eval_strategy,
        "eval_steps": config.eval_steps,
        "save_strategy": "steps",
        "save_steps": config.save_steps,
        "save_total_limit": config.save_total_limit,
        "report_to": ["trackio"],
        "project": tracking_project,
        "run_name": config.run_name,
        "trackio_space_id": trackio_space_id,
        "trackio_bucket_id": trackio_bucket_id,
        "trackio_static_space_id": False,
        "remove_unused_columns": False,
    }


@contextmanager
def _managed_training_environment(
    config: ProjectConfig,
    *,
    tracking_project: str | None = None,
) -> Iterator[None]:
    paths = ManagedPaths(config)
    values = _training_environment_values(
        config,
        paths=paths,
        tracking_project=tracking_project,
    )
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
        _restore_environment(previous)


def _load_hugging_face_token() -> str:
    token = os.environ.get("HF_TOKEN", "").strip()
    if token:
        return token
    token_path = (
        Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
        / "token"
    )
    try:
        return token_path.read_text(encoding="utf-8").strip()
    except OSError:
        return ""


def _training_environment_values(
    config: ProjectConfig,
    *,
    paths: ManagedPaths,
    tracking_project: str | None,
) -> dict[str, str]:
    values = {
        "HF_HOME": str(paths.child("cache/huggingface")),
        **settings_for(config, project=tracking_project).environment(),
    }
    token = _load_hugging_face_token()
    if token:
        values["HF_TOKEN"] = token
    return values


def _restore_environment(previous: Mapping[str, str | None]) -> None:
    for name, value in previous.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def _evaluate_test_dataset(trainer: Any, test_dataset: Any) -> dict[str, object]:
    """Evaluate the held-out dataset once, after training has finished."""

    if test_dataset is None:
        return {}
    evaluate = getattr(trainer, "evaluate", None)
    if not callable(evaluate):
        raise TrainingError("Trainer does not expose held-out test evaluation")
    try:
        raw_metrics = evaluate(test_dataset, metric_key_prefix="test")
    except Exception as error:
        raise TrainingError("held-out test evaluation failed") from error
    if not isinstance(raw_metrics, Mapping):
        raise TrainingError("held-out test metrics are invalid")
    return dict(raw_metrics)


def _evaluate_validation_dataset(
    trainer: Any, validation_dataset: Any
) -> dict[str, object]:
    """Evaluate the validation dataset once, after training has finished."""

    evaluate = getattr(trainer, "evaluate", None)
    if not callable(evaluate):
        raise TrainingError("Trainer does not expose validation evaluation")
    try:
        raw_metrics = evaluate(validation_dataset, metric_key_prefix="eval")
    except Exception as error:
        raise TrainingError("validation evaluation failed") from error
    if not isinstance(raw_metrics, Mapping):
        raise TrainingError("validation metrics are invalid")
    return dict(raw_metrics)


def _publish_completed_model(
    output_directory: Path,
    *,
    project_config: ProjectConfig,
    config: TrainingConfig,
    contract: DatasetContract,
    identity: Mapping[str, object],
    metrics: Mapping[str, object],
    tracking: TrackioSettings,
    checkpoint_hub_api: Any,
) -> ModelPublicationResult | None:
    """Publish one validated final model and any V2 study documents."""

    if not config.publish_to_hub:
        return None
    repository_readme = _repository_readme(
        config=config,
        contract=contract,
        identity=identity,
        tracking=tracking,
    )
    publication = publish_model_directory(
        output_directory,
        project_config.target_model_repository_id,
        identity=identity,
        repository_readme=repository_readme,
    )
    if _is_worldwide_v2_baseline(config, contract):
        _publish_worldwide_v2_documents(
            project_config,
            config=config,
            identity=identity,
            metrics=metrics,
            tracking=tracking,
            checkpoint_hub_api=checkpoint_hub_api,
        )
    return publication


def _repository_readme(
    *,
    config: TrainingConfig,
    contract: DatasetContract,
    identity: Mapping[str, object],
    tracking: TrackioSettings,
) -> str | None:
    if config.artifact_namespace is not None and (
        contract is not WORLDWIDE_PLACE_RELEVANCE_V2_CONTRACT
    ):
        return None
    return render_repository_readme(
        identity=identity,
        trackio_space_id=(tracking.static_space_id if config.sync_trackio else None),
    )


def _is_worldwide_v2_baseline(
    config: TrainingConfig, contract: DatasetContract
) -> bool:
    return contract is WORLDWIDE_PLACE_RELEVANCE_V2_CONTRACT and (
        config.artifact_namespace in {None, "studies/place-relevance-v2/baseline"}
    )


def _publish_worldwide_v2_documents(
    project_config: ProjectConfig,
    *,
    config: TrainingConfig,
    identity: Mapping[str, object],
    metrics: Mapping[str, object],
    tracking: TrackioSettings,
    checkpoint_hub_api: Any,
) -> None:
    try:
        publish_study_documents(
            project_config.target_model_repository_id,
            render_place_relevance_study_documents(
                identity=identity,
                metrics=metrics,
                trackio_space_id=(
                    tracking.static_space_id if config.sync_trackio else None
                ),
            ),
            hub_api=checkpoint_hub_api,
        )
    except ModelPublicationError as error:
        raise TrainingError(
            "worldwide V2 study documentation publication failed"
        ) from error


@dataclass(frozen=True, slots=True)
class _TrainingContext:
    config: TrainingConfig
    project_config: ProjectConfig
    contract: DatasetContract
    rows_factory: Callable[[], Iterable[Mapping[str, object]]]
    output_directory: Path
    model_cache_directory: Path
    tracking: TrackioSettings
    resume_from_checkpoint: Path | None
    checkpoint_identity: Mapping[str, object] | None


def _effective_training_config(config: TrainingConfig | None) -> TrainingConfig:
    return config or TrainingConfig()


def _effective_project_config(config: ProjectConfig | None) -> ProjectConfig:
    return config or ProjectConfig()


def _effective_rows_factory(
    rows_factory: Callable[[], Iterable[Mapping[str, object]]] | None,
    *,
    project_config: ProjectConfig,
    contract: DatasetContract,
) -> Callable[[], Iterable[Mapping[str, object]]]:
    if rows_factory is not None:
        return rows_factory

    def load_rows() -> Iterable[Mapping[str, object]]:
        return load_streaming_rows(config=project_config, contract=contract)

    return load_rows


def _training_context(
    rows_factory: Callable[[], Iterable[Mapping[str, object]]] | None,
    *,
    config: TrainingConfig | None,
    project_config: ProjectConfig | None,
    contract: DatasetContract,
    resume_from_checkpoint: Path | None,
    checkpoint_identity: Mapping[str, object] | None,
) -> _TrainingContext:
    training_config = _effective_training_config(config)
    effective_project_config = _effective_project_config(project_config)
    paths = ManagedPaths(effective_project_config)
    output_directory = paths.child(training_config.output_subdirectory)
    tracking = settings_for(
        effective_project_config,
        project=training_config.tracking_project,
    )
    resume_from_checkpoint, checkpoint_identity = _prepare_checkpoint_resume(
        output_directory,
        resume_from_checkpoint=resume_from_checkpoint,
        checkpoint_identity=checkpoint_identity,
    )
    return _TrainingContext(
        config=training_config,
        project_config=effective_project_config,
        contract=contract,
        rows_factory=_effective_rows_factory(
            rows_factory,
            project_config=effective_project_config,
            contract=contract,
        ),
        output_directory=output_directory,
        model_cache_directory=paths.child("cache/huggingface/models"),
        tracking=tracking,
        resume_from_checkpoint=resume_from_checkpoint,
        checkpoint_identity=checkpoint_identity,
    )


def _restore_tracking_snapshot_if_needed(context: _TrainingContext) -> None:
    if context.config.sync_trackio and context.config.tracking_project is not None:
        restore_static_project_snapshot(context.tracking)


def _tokenizer_kwargs(
    config: TrainingConfig, cache_directory: Path
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "cache_dir": str(cache_directory),
        "use_fast": True,
    }
    if config.model_revision is not None:
        kwargs["revision"] = config.model_revision
    return kwargs


def _load_tokenizer(
    dependencies: _training_runtime.TrainingDependencies,
    context: _TrainingContext,
) -> Any:
    return dependencies.auto_tokenizer.from_pretrained(
        context.config.model_name_or_path,
        **_tokenizer_kwargs(context.config, context.model_cache_directory),
    )


def _training_datasets(
    dependencies: _training_runtime.TrainingDependencies,
    context: _TrainingContext,
    tokenizer: Any,
) -> tuple[Any, Any, Any]:
    train_dataset = _make_tokenized_dataset(
        dependencies,
        context.rows_factory,
        split="train",
        config=context.config,
        contract=context.contract,
        tokenizer=tokenizer,
    )
    validation_dataset = _make_tokenized_dataset(
        dependencies,
        context.rows_factory,
        split="validation",
        config=context.config,
        contract=context.contract,
        tokenizer=tokenizer,
    )
    test_dataset = None
    if context.config.test_fraction > 0:
        test_dataset = _make_tokenized_dataset(
            dependencies,
            context.rows_factory,
            split="test",
            config=context.config,
            contract=context.contract,
            tokenizer=tokenizer,
        )
    return train_dataset, validation_dataset, test_dataset


def _model_kwargs(config: TrainingConfig, cache_directory: Path) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "cache_dir": str(cache_directory),
        "classifier_dropout": 0.0,
        "num_labels": len(LABEL_TO_ID),
        "id2label": ID_TO_LABEL,
        "label2id": LABEL_TO_ID,
    }
    if config.model_revision is not None:
        kwargs["revision"] = config.model_revision
    return kwargs


def _load_model(
    dependencies: _training_runtime.TrainingDependencies,
    context: _TrainingContext,
) -> Any:
    model = dependencies.auto_model_for_sequence_classification.from_pretrained(
        context.config.model_name_or_path,
        **_model_kwargs(context.config, context.model_cache_directory),
    )
    configure_trainable_layers(model, context.config.trainable_layers)
    return model


def _checkpoint_publication_api(context: _TrainingContext) -> Any:
    if context.checkpoint_identity is None or not context.config.publish_to_hub:
        return None
    return _training_runtime.load_checkpoint_publication_api()


def _build_training_trainer(
    dependencies: _training_runtime.TrainingDependencies,
    *,
    context: _TrainingContext,
    model: Any,
    tokenizer: Any,
    train_dataset: Any,
    validation_dataset: Any,
    checkpoint_hub_api: Any,
) -> Any:
    training_arguments = dependencies.training_arguments(
        **_training_argument_values(
            context.config,
            output_directory=context.output_directory,
            tracking_project=context.tracking.project,
            trackio_space_id=None,
            trackio_bucket_id=None,
        )
    )
    return _training_runtime.build_trainer(
        dependencies,
        model=model,
        training_arguments=training_arguments,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        data_collator=dependencies.data_collator_with_padding(tokenizer=tokenizer),
        checkpoint_identity=context.checkpoint_identity,
        model_repository_id=(
            context.project_config.target_model_repository_id
            if context.config.publish_to_hub
            else None
        ),
        trackio_space_id=(
            context.tracking.static_space_id if context.config.sync_trackio else None
        ),
        tracking_settings=(context.tracking if context.config.sync_trackio else None),
        hub_api=checkpoint_hub_api,
        hub_checkpoint_steps=context.config.hub_checkpoint_steps,
        class_weight_mode=context.config.class_weight_mode,
    )


def _finalize_training_outputs(
    trainer: Any,
    *,
    context: _TrainingContext,
    tokenizer: Any,
    validation_dataset: Any,
    test_dataset: Any,
) -> tuple[Any, dict[str, object], dict[str, object]]:
    train_output = _training_runtime.run_trainer(
        trainer, context.resume_from_checkpoint
    )
    final_metrics = _training_metrics.metrics_for_model_card(train_output, trainer)
    final_metrics.update(_evaluate_validation_dataset(trainer, validation_dataset))
    final_metrics.update(_evaluate_test_dataset(trainer, test_dataset))
    trainer.save_model(str(context.output_directory))
    tokenizer.save_pretrained(str(context.output_directory))
    model_card_identity = _model_card_identity(
        context.checkpoint_identity,
        config=context.config,
        contract=context.contract,
    )
    _write_model_card(
        context.output_directory,
        identity=model_card_identity,
        training_metrics=final_metrics,
        trackio_space_id=(
            context.tracking.static_space_id if context.config.sync_trackio else None
        ),
    )
    return train_output, final_metrics, model_card_identity


def _train_classifier(
    rows_factory: Callable[[], Iterable[Mapping[str, object]]] | None = None,
    *,
    config: TrainingConfig | None = None,
    project_config: ProjectConfig | None = None,
    contract: DatasetContract = LANDUSE_DATASET_CONTRACT,
    resume_from_checkpoint: Path | None = None,
    checkpoint_identity: Mapping[str, object] | None = None,
) -> TrainingResult:
    """Train and save a binary classifier using the clean stream boundary.

    The optional dependency group is imported only when this function is called.
    Datasets, model caches, checkpoints, and Trackio state are directed beneath
    the approved external data root. Publication and Trackio synchronization are
    disabled by default and require explicit configuration flags. When
    checkpoint identity is supplied, enabled publication and synchronization
    are also performed at each complete checkpoint.
    """

    context = _training_context(
        rows_factory,
        config=config,
        project_config=project_config,
        contract=contract,
        resume_from_checkpoint=resume_from_checkpoint,
        checkpoint_identity=checkpoint_identity,
    )
    checkpoint_hub_api: Any
    train_output: Any
    final_metrics: dict[str, object]
    model_publication: ModelPublicationResult | None
    with _managed_training_environment(
        context.project_config,
        tracking_project=context.config.tracking_project,
    ):
        _restore_tracking_snapshot_if_needed(context)
        dependencies = _training_runtime.load_training_dependencies()
        tokenizer = _load_tokenizer(dependencies, context)
        train_dataset, validation_dataset, test_dataset = _training_datasets(
            dependencies,
            context,
            tokenizer,
        )
        model = _load_model(dependencies, context)
        checkpoint_hub_api = _checkpoint_publication_api(context)
        trainer = _build_training_trainer(
            dependencies,
            context=context,
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            checkpoint_hub_api=checkpoint_hub_api,
        )
        train_output, final_metrics, model_card_identity = _finalize_training_outputs(
            trainer,
            context=context,
            tokenizer=tokenizer,
            validation_dataset=validation_dataset,
            test_dataset=test_dataset,
        )
        model_publication = _publish_completed_model(
            context.output_directory,
            project_config=context.project_config,
            config=context.config,
            contract=context.contract,
            identity=model_card_identity,
            metrics=final_metrics,
            tracking=context.tracking,
            checkpoint_hub_api=checkpoint_hub_api,
        )
        if context.config.sync_trackio:
            _sync_static_trackio(
                context.tracking,
                failure_message="Trackio static snapshot failed",
                finalize=True,
            )
        tracking_space_id = (
            context.tracking.static_space_id if context.config.sync_trackio else None
        )

    return TrainingResult(
        output_directory=context.output_directory,
        train_output=train_output,
        model_publication=model_publication,
        tracking_space_id=tracking_space_id,
        metrics=final_metrics,
    )


def train_landuse_classifier(
    rows_factory: Callable[[], Iterable[Mapping[str, object]]] | None = None,
    *,
    config: TrainingConfig | None = None,
    project_config: ProjectConfig | None = None,
    contract: DatasetContract = LANDUSE_DATASET_CONTRACT,
    resume_from_checkpoint: Path | None = None,
    checkpoint_identity: Mapping[str, object] | None = None,
) -> TrainingResult:
    """Train the existing Afghanistan landuse classifier."""

    return _train_classifier(
        rows_factory,
        config=config,
        project_config=project_config,
        contract=contract,
        resume_from_checkpoint=resume_from_checkpoint,
        checkpoint_identity=checkpoint_identity,
    )


def train_place_relevance_classifier(
    rows_factory: Callable[[], Iterable[Mapping[str, object]]] | None = None,
    *,
    config: TrainingConfig | None = None,
    project_config: ProjectConfig | None = None,
    resume_from_checkpoint: Path | None = None,
    checkpoint_identity: Mapping[str, object] | None = None,
) -> TrainingResult:
    """Train the worldwide V2 place-relevance classifier."""

    training_config = config or place_relevance_v2_training_config()
    return _train_classifier(
        rows_factory,
        config=training_config,
        project_config=project_config,
        contract=WORLDWIDE_PLACE_RELEVANCE_V2_CONTRACT,
        resume_from_checkpoint=resume_from_checkpoint,
        checkpoint_identity=checkpoint_identity,
    )


__all__ = [
    "DEFAULT_MODEL_NAME",
    "ID_TO_LABEL",
    "LABEL_TO_ID",
    "PLACE_RELEVANCE_V2_DEFAULT_MAX_STEPS",
    "PLACE_RELEVANCE_V2_OUTPUT",
    "PLACE_RELEVANCE_V2_RUN_NAME",
    "TrainingConfig",
    "TrainingError",
    "TrainingRecord",
    "TrainingResult",
    "iter_split_training_records",
    "train_landuse_classifier",
    "train_place_relevance_classifier",
    "place_relevance_v2_training_config",
]
