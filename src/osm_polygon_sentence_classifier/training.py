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
    if (
        not isinstance(config.model_name_or_path, str)
        or not config.model_name_or_path.strip()
    ):
        raise TrainingError("model_name_or_path must be a non-empty string")
    if not isinstance(config.run_name, str) or not config.run_name.strip():
        raise TrainingError("run_name must be a non-empty string")
    if config.model_revision is not None and (
        not isinstance(config.model_revision, str)
        or re.fullmatch(r"[0-9a-f]{40}", config.model_revision) is None
    ):
        raise TrainingError(
            "model_revision must be exactly 40 lowercase hexadecimal characters"
        )
    if config.trainable_layers not in {None, "head", "last2"}:
        raise TrainingError("trainable_layers must be head or last2")
    if config.class_weight_mode not in {None, "none", "balanced"}:
        raise TrainingError("class_weight_mode must be none or balanced")
    if config.eval_strategy not in {"steps", "epoch"}:
        raise TrainingError("eval_strategy must be steps or epoch")
    for name in ("tracking_project", "artifact_namespace"):
        value = getattr(config, name)
        if value is not None and (
            not isinstance(value, str)
            or not value.strip()
            or "\n" in value
            or "\r" in value
        ):
            raise TrainingError(f"{name} must be a non-empty single-line string")


def _normalize_training_paths(config: TrainingConfig) -> None:
    output_subdirectory = Path(config.output_subdirectory)
    if (
        not output_subdirectory.parts
        or output_subdirectory.is_absolute()
        or any(part in (".", "..") for part in output_subdirectory.parts)
    ):
        raise TrainingError(
            "output_subdirectory must be a clean relative path beneath the "
            "managed data root"
        )
    object.__setattr__(config, "output_subdirectory", output_subdirectory)

    if config.artifact_namespace is not None:
        namespace = Path(config.artifact_namespace)
        if (
            not namespace.parts
            or namespace.is_absolute()
            or any(part in (".", "..") for part in namespace.parts)
        ):
            raise TrainingError("artifact_namespace must be a clean relative path")
        object.__setattr__(config, "artifact_namespace", namespace.as_posix())


def _validate_numeric_settings(config: TrainingConfig) -> None:
    if (
        isinstance(config.validation_fraction, bool)
        or not isinstance(config.validation_fraction, (int, float))
        or not math.isfinite(config.validation_fraction)
        or not 0 <= config.validation_fraction <= 1
    ):
        raise TrainingError(
            "validation_fraction must be a finite number between 0 and 1"
        )
    if (
        isinstance(config.test_fraction, bool)
        or not isinstance(config.test_fraction, (int, float))
        or not math.isfinite(config.test_fraction)
        or not 0 <= config.test_fraction <= 1
        or config.validation_fraction + config.test_fraction > 1
    ):
        raise TrainingError(
            "validation and test fractions must be finite, non-negative, "
            "and sum to at most 1"
        )
    for name in (
        "max_length",
        "max_steps",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "logging_steps",
        "eval_steps",
        "save_steps",
        "save_total_limit",
    ):
        value = getattr(config, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise TrainingError(f"{name} must be a positive integer")
    if (
        isinstance(config.learning_rate, bool)
        or not isinstance(config.learning_rate, (int, float))
        or not math.isfinite(config.learning_rate)
        or config.learning_rate <= 0
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
    if resume_from_checkpoint is not None and checkpoint_identity is None:
        raise TrainingError("checkpoint identity is required for resume")
    if checkpoint_identity is None:
        return resume_from_checkpoint, None
    identity = dict(checkpoint_identity)
    if resume_from_checkpoint is None:
        return None, identity
    try:
        selected = find_latest_complete_checkpoint(
            output_directory,
            identity=identity,
        )
    except CheckpointError as error:
        raise TrainingError("checkpoint evidence is invalid") from error
    if selected is None or selected.path != resume_from_checkpoint:
        raise TrainingError("requested checkpoint is not a complete identity match")
    return resume_from_checkpoint, identity


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
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        token_path = (
            Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
            / "token"
        )
        try:
            token = token_path.read_text(encoding="utf-8").strip()
        except OSError:
            token = ""
    values = {
        "HF_HOME": str(paths.child("cache/huggingface")),
        **settings_for(config, project=tracking_project).environment(),
    }
    if token:
        values["HF_TOKEN"] = token
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield
    finally:
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
    repository_readme = (
        render_repository_readme(
            identity=identity,
            trackio_space_id=(
                tracking.static_space_id if config.sync_trackio else None
            ),
        )
        if config.artifact_namespace is None
        or contract is WORLDWIDE_PLACE_RELEVANCE_V2_CONTRACT
        else None
    )
    publication = publish_model_directory(
        output_directory,
        project_config.target_model_repository_id,
        identity=identity,
        repository_readme=repository_readme,
    )
    if contract is WORLDWIDE_PLACE_RELEVANCE_V2_CONTRACT:
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
    return publication


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

    training_config = config or TrainingConfig()
    effective_project_config = project_config or ProjectConfig()
    paths = ManagedPaths(effective_project_config)
    output_directory = paths.child(training_config.output_subdirectory)
    model_cache_directory = paths.child("cache/huggingface/models")
    tracking = settings_for(
        effective_project_config,
        project=training_config.tracking_project,
    )
    resume_from_checkpoint, checkpoint_identity = _prepare_checkpoint_resume(
        output_directory,
        resume_from_checkpoint=resume_from_checkpoint,
        checkpoint_identity=checkpoint_identity,
    )

    effective_rows_factory = rows_factory
    if effective_rows_factory is None:

        def load_rows() -> Iterable[Mapping[str, object]]:
            return load_streaming_rows(
                config=effective_project_config,
                contract=contract,
            )

        effective_rows_factory = load_rows

    with _managed_training_environment(
        effective_project_config,
        tracking_project=training_config.tracking_project,
    ):
        if (
            training_config.sync_trackio
            and training_config.tracking_project is not None
        ):
            restore_static_project_snapshot(tracking)
        dependencies = _training_runtime.load_training_dependencies()
        tokenizer_kwargs: dict[str, object] = {
            "cache_dir": str(model_cache_directory),
            "use_fast": True,
        }
        if training_config.model_revision is not None:
            tokenizer_kwargs["revision"] = training_config.model_revision
        tokenizer = dependencies.auto_tokenizer.from_pretrained(
            training_config.model_name_or_path,
            **tokenizer_kwargs,
        )
        train_dataset = _make_tokenized_dataset(
            dependencies,
            effective_rows_factory,
            split="train",
            config=training_config,
            contract=contract,
            tokenizer=tokenizer,
        )
        validation_dataset = _make_tokenized_dataset(
            dependencies,
            effective_rows_factory,
            split="validation",
            config=training_config,
            contract=contract,
            tokenizer=tokenizer,
        )
        test_dataset = None
        if training_config.test_fraction > 0:
            test_dataset = _make_tokenized_dataset(
                dependencies,
                effective_rows_factory,
                split="test",
                config=training_config,
                contract=contract,
                tokenizer=tokenizer,
            )
        model_kwargs: dict[str, object] = {
            "cache_dir": str(model_cache_directory),
            "classifier_dropout": 0.0,
            "num_labels": len(LABEL_TO_ID),
            "id2label": ID_TO_LABEL,
            "label2id": LABEL_TO_ID,
        }
        if training_config.model_revision is not None:
            model_kwargs["revision"] = training_config.model_revision
        model = dependencies.auto_model_for_sequence_classification.from_pretrained(
            training_config.model_name_or_path,
            **model_kwargs,
        )
        configure_trainable_layers(model, training_config.trainable_layers)
        training_arguments = dependencies.training_arguments(
            **_training_argument_values(
                training_config,
                output_directory=output_directory,
                tracking_project=tracking.project,
                trackio_space_id=None,
                trackio_bucket_id=None,
            )
        )
        data_collator = dependencies.data_collator_with_padding(tokenizer=tokenizer)
        checkpoint_hub_api = None
        if checkpoint_identity is not None and training_config.publish_to_hub:
            checkpoint_hub_api = _training_runtime.load_checkpoint_publication_api()
        trainer = _training_runtime.build_trainer(
            dependencies,
            model=model,
            training_arguments=training_arguments,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            data_collator=data_collator,
            checkpoint_identity=checkpoint_identity,
            model_repository_id=(
                effective_project_config.target_model_repository_id
                if training_config.publish_to_hub
                else None
            ),
            trackio_space_id=(
                tracking.static_space_id if training_config.sync_trackio else None
            ),
            tracking_settings=(tracking if training_config.sync_trackio else None),
            hub_api=checkpoint_hub_api,
            class_weight_mode=training_config.class_weight_mode,
        )
        train_output = _training_runtime.run_trainer(trainer, resume_from_checkpoint)
        test_metrics = _evaluate_test_dataset(trainer, test_dataset)
        trainer.save_model(str(output_directory))
        tokenizer.save_pretrained(str(output_directory))
        final_metrics = _training_metrics.metrics_for_model_card(train_output, trainer)
        final_metrics.update(test_metrics)
        model_card_identity = _model_card_identity(
            checkpoint_identity,
            config=training_config,
            contract=contract,
        )
        _write_model_card(
            output_directory,
            identity=model_card_identity,
            training_metrics=final_metrics,
            trackio_space_id=(
                tracking.static_space_id if training_config.sync_trackio else None
            ),
        )
        model_publication = _publish_completed_model(
            output_directory,
            project_config=effective_project_config,
            config=training_config,
            contract=contract,
            identity=model_card_identity,
            metrics=final_metrics,
            tracking=tracking,
            checkpoint_hub_api=checkpoint_hub_api,
        )
        if training_config.sync_trackio:
            _sync_static_trackio(
                tracking,
                failure_message="Trackio static snapshot failed",
                finalize=True,
            )
        tracking_space_id = (
            tracking.static_space_id if training_config.sync_trackio else None
        )

    return TrainingResult(
        output_directory=output_directory,
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
