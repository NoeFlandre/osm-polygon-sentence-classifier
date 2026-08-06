"""Managed, streaming training orchestration for the landuse classifier."""

import math
import os
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, fields
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

from .checkpointing import (
    CheckpointError,
    find_latest_complete_checkpoint,
    write_checkpoint_manifest,
)
from .config import ProjectConfig
from .dataset_contract import LANDUSE_DATASET_CONTRACT, DatasetContract
from .dataset_loader import (
    DatasetSplit,
    TrainingLabel,
    iter_clean_training_examples,
    load_streaming_rows,
)
from .paths import ManagedPaths
from .publication import (
    ModelPublicationError,
    ModelPublicationResult,
    publish_checkpoint_directory,
    publish_model_directory,
    render_model_card,
)
from .tracking import (
    TrackingError,
    TrackioSettings,
    settings_for,
    sync_project_to_static_space,
)

LabelId = Literal[0, 1]

LABEL_TO_ID: dict[TrainingLabel, LabelId] = {"no": 0, "yes": 1}
ID_TO_LABEL: dict[int, str] = {0: "no", 1: "yes"}
DEFAULT_MODEL_NAME = "jhu-clsp/mmBERT-small"


class TrainingError(RuntimeError):
    """Raised when training dependencies or configuration are unavailable."""


class TrainingRecord(TypedDict):
    """One clean, split-assigned record before tokenization."""

    text: str
    labels: LabelId


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Small, explicit configuration for one landuse training run."""

    model_name_or_path: str = DEFAULT_MODEL_NAME
    output_subdirectory: Path = Path("models/landuse")
    validation_fraction: float = 0.2
    seed: int = 42
    max_length: int = 256
    max_steps: int = 1_000
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    learning_rate: float = 3e-4
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    save_total_limit: int = 5
    run_name: str = "landuse-mmbert-small-frozen-head"
    model_revision: str | None = None
    publish_to_hub: bool = False
    sync_trackio: bool = False

    def __post_init__(self) -> None:
        if (
            not isinstance(self.model_name_or_path, str)
            or not self.model_name_or_path.strip()
        ):
            raise TrainingError("model_name_or_path must be a non-empty string")
        if not isinstance(self.run_name, str) or not self.run_name.strip():
            raise TrainingError("run_name must be a non-empty string")
        if self.model_revision is not None and (
            not isinstance(self.model_revision, str)
            or re.fullmatch(r"[0-9a-f]{40}", self.model_revision) is None
        ):
            raise TrainingError(
                "model_revision must be exactly 40 lowercase hexadecimal characters"
            )
        _validate_boolean_settings(self)

        output_subdirectory = Path(self.output_subdirectory)
        if (
            not output_subdirectory.parts
            or output_subdirectory.is_absolute()
            or any(part in (".", "..") for part in output_subdirectory.parts)
        ):
            raise TrainingError(
                "output_subdirectory must be a clean relative path beneath the "
                "managed data root"
            )
        object.__setattr__(self, "output_subdirectory", output_subdirectory)

        if (
            isinstance(self.validation_fraction, bool)
            or not isinstance(self.validation_fraction, (int, float))
            or not math.isfinite(self.validation_fraction)
            or not 0 <= self.validation_fraction <= 1
        ):
            raise TrainingError(
                "validation_fraction must be a finite number between 0 and 1"
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
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise TrainingError(f"{name} must be a positive integer")
        if (
            isinstance(self.learning_rate, bool)
            or not isinstance(self.learning_rate, (int, float))
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise TrainingError("learning_rate must be a positive finite number")


def _validate_boolean_settings(config: TrainingConfig) -> None:
    for name in ("publish_to_hub", "sync_trackio"):
        if not isinstance(getattr(config, name), bool):
            raise TrainingError(f"{name} must be a boolean")


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Local result of a completed Trainer run."""

    output_directory: Path
    train_output: object
    model_publication: ModelPublicationResult | None = None
    tracking_space_id: str | None = None


def _is_card_scalar(value: object) -> bool:
    if value is None or isinstance(value, (bool, int, str)):
        return True
    return isinstance(value, float) and math.isfinite(value)


def _latest_training_metrics(state: Any) -> dict[str, object]:
    history = getattr(state, "log_history", ())
    if not isinstance(history, Sequence):
        return {}
    for entry in reversed(history):
        if isinstance(entry, Mapping):
            return {
                key: value
                for key, value in entry.items()
                if isinstance(key, str) and _is_card_scalar(value)
            }
    return {}


def _latest_evaluation_metrics(state: Any) -> dict[str, object]:
    history = getattr(state, "log_history", ())
    if not isinstance(history, Sequence):
        return {}
    for entry in reversed(history):
        if isinstance(entry, Mapping) and any(
            isinstance(key, str) and key.startswith("eval_") for key in entry
        ):
            return {
                key: value
                for key, value in entry.items()
                if isinstance(key, str) and _is_card_scalar(value)
            }
    return {}


def _metrics_for_model_card(train_output: Any, trainer: Any) -> dict[str, object]:
    metrics: dict[str, object] = {}
    raw_training_metrics = getattr(train_output, "metrics", None)
    if isinstance(raw_training_metrics, Mapping):
        metrics.update(raw_training_metrics)
    metrics.update(_latest_evaluation_metrics(getattr(trainer, "state", None)))
    return metrics


def _as_python(value: Any) -> object:
    tolist = getattr(value, "tolist", None)
    return tolist() if callable(tolist) else value


def _classification_metrics(eval_prediction: Any) -> dict[str, float]:
    """Compute binary accuracy, precision, recall, and F1 for Trainer evals."""

    predictions = _as_python(getattr(eval_prediction, "predictions", None))
    labels = _as_python(getattr(eval_prediction, "label_ids", None))
    if isinstance(predictions, tuple) and predictions:
        predictions = _as_python(predictions[0])
    if (
        not isinstance(predictions, Sequence)
        or isinstance(predictions, (str, bytes))
        or not isinstance(labels, Sequence)
        or isinstance(labels, (str, bytes))
        or len(predictions) != len(labels)
        or not predictions
    ):
        raise TrainingError("evaluation predictions and labels are invalid")

    predicted_labels: list[int] = []
    actual_labels: list[int] = []
    for logits, label in zip(predictions, labels, strict=True):
        row = _as_python(logits)
        actual = _as_python(label)
        if (
            not isinstance(row, Sequence)
            or isinstance(row, (str, bytes))
            or not row
            or isinstance(actual, bool)
            or not isinstance(actual, int)
            or actual not in (0, 1)
        ):
            raise TrainingError("evaluation predictions and labels are invalid")
        predicted = max(range(len(row)), key=lambda index: row[index])
        if predicted not in (0, 1):
            raise TrainingError("evaluation predictions and labels are invalid")
        predicted_labels.append(predicted)
        actual_labels.append(actual)

    true_positive = sum(
        predicted == actual == 1
        for predicted, actual in zip(predicted_labels, actual_labels, strict=True)
    )
    true_negative = sum(
        predicted == actual == 0
        for predicted, actual in zip(predicted_labels, actual_labels, strict=True)
    )
    false_positive = sum(
        predicted == 1 and actual == 0
        for predicted, actual in zip(predicted_labels, actual_labels, strict=True)
    )
    false_negative = sum(
        predicted == 0 and actual == 1
        for predicted, actual in zip(predicted_labels, actual_labels, strict=True)
    )
    accuracy = (true_positive + true_negative) / len(actual_labels)
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _training_config_payload(config: TrainingConfig) -> dict[str, object]:
    payload: dict[str, object] = {}
    for item in fields(config):
        value = getattr(config, item.name)
        payload[item.name] = str(value) if isinstance(value, Path) else value
    return payload


def _model_card_identity(
    identity: Mapping[str, object] | None,
    *,
    config: TrainingConfig,
    contract: DatasetContract,
) -> dict[str, object]:
    payload = dict(identity or {})
    payload.setdefault("task_name", "landuse")
    payload.setdefault("dataset_revision", contract.provenance.repository_revision)
    payload.setdefault("model_name_or_path", config.model_name_or_path)
    payload.setdefault("model_revision", config.model_revision)
    if not isinstance(payload.get("training_config"), Mapping):
        payload["training_config"] = _training_config_payload(config)
    return payload


def _write_model_card(
    directory: Path,
    *,
    identity: Mapping[str, object],
    training_metrics: Mapping[str, object] | None = None,
    checkpoint_step: int | None = None,
    trackio_space_id: str | None = None,
) -> None:
    card = render_model_card(
        identity=identity,
        training_metrics=training_metrics,
        checkpoint_step=checkpoint_step,
        trackio_space_id=trackio_space_id,
    )
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
    (directory / "README.md").write_text(card, encoding="utf-8")


def _sync_static_trackio(
    settings: TrackioSettings,
    *,
    failure_message: str,
) -> None:
    try:
        sync_project_to_static_space(settings)
    except TrackingError as error:
        raise TrainingError(failure_message) from error


@dataclass(frozen=True, slots=True)
class _TrainingDependencies:
    """Lazily imported Hugging Face classes, kept injectable for unit tests."""

    iterable_dataset: Any
    auto_tokenizer: Any
    auto_model_for_sequence_classification: Any
    data_collator_with_padding: Any
    training_arguments: Any
    trainer: Any
    trainer_callback: Any | None = None


class _CheckpointManifestCallback:
    """Record, publish, and track one checkpoint after Trainer saves it."""

    def __init__(
        self,
        identity: Mapping[str, object],
        *,
        model_repository_id: str | None = None,
        trackio_space_id: str | None = None,
        tracking_settings: TrackioSettings | None = None,
        hub_api: Any | None = None,
    ) -> None:
        self.identity = dict(identity)
        self.model_repository_id = model_repository_id
        self.trackio_space_id = trackio_space_id
        self.tracking_settings = tracking_settings
        self.hub_api = hub_api
        self._pending_publications: list[Any] = []

    def on_init_end(self, args: Any, state: Any, control: Any, **kwargs: object) -> Any:
        del args, state, kwargs
        return control

    def _wait_for_next_publication(self) -> None:
        if not self._pending_publications:
            return
        future = self._pending_publications.pop(0)
        try:
            future.result()
        except Exception as error:
            raise TrainingError("checkpoint model publication failed") from error

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: object) -> Any:
        del kwargs
        output_directory = getattr(args, "output_dir", None)
        global_step = getattr(state, "global_step", None)
        if not isinstance(output_directory, str) or not isinstance(global_step, int):
            raise TrainingError("checkpoint save did not expose a valid step")
        try:
            write_checkpoint_manifest(
                Path(output_directory) / f"checkpoint-{global_step}",
                identity=self.identity,
                global_step=global_step,
            )
        except CheckpointError as error:
            raise TrainingError("checkpoint manifest could not be written") from error
        if self.model_repository_id is not None:
            checkpoint = Path(output_directory) / f"checkpoint-{global_step}"
            try:
                _write_model_card(
                    checkpoint,
                    identity=self.identity,
                    training_metrics={
                        **_latest_training_metrics(state),
                        **_latest_evaluation_metrics(state),
                    },
                    checkpoint_step=global_step,
                    trackio_space_id=self.trackio_space_id,
                )
            except OSError as error:
                raise TrainingError(
                    "checkpoint model card could not be written"
                ) from error
            if self.hub_api is None:
                try:
                    publish_checkpoint_directory(
                        checkpoint,
                        self.model_repository_id,
                        identity=self.identity,
                    )
                except ModelPublicationError as error:
                    raise TrainingError(
                        "checkpoint model publication failed"
                    ) from error
            else:
                run_as_future = getattr(self.hub_api, "run_as_future", None)
                if not callable(run_as_future):
                    raise TrainingError(
                        "checkpoint publication API cannot queue background work"
                    )
                self._pending_publications.append(
                    run_as_future(
                        publish_checkpoint_directory,
                        checkpoint,
                        self.model_repository_id,
                        identity=self.identity,
                        hub_api=self.hub_api,
                    )
                )
                save_total_limit = getattr(args, "save_total_limit", None)
                if (
                    isinstance(save_total_limit, int)
                    and not isinstance(save_total_limit, bool)
                    and save_total_limit > 0
                    and len(self._pending_publications) >= save_total_limit
                ):
                    self._wait_for_next_publication()
        if self.tracking_settings is not None:
            _sync_static_trackio(
                self.tracking_settings,
                failure_message="checkpoint Trackio static snapshot failed",
            )
        return control

    def on_train_end(
        self, args: Any, state: Any, control: Any, **kwargs: object
    ) -> Any:
        del args, state, kwargs
        while self._pending_publications:
            self._wait_for_next_publication()
        return control


def _make_checkpoint_manifest_callback(
    identity: Mapping[str, object],
    trainer_callback: Any | None,
    *,
    model_repository_id: str | None = None,
    trackio_space_id: str | None = None,
    tracking_settings: TrackioSettings | None = None,
    hub_api: Any | None = None,
) -> Any:
    if trainer_callback is None:
        return _CheckpointManifestCallback(
            identity,
            model_repository_id=model_repository_id,
            trackio_space_id=trackio_space_id,
            tracking_settings=tracking_settings,
            hub_api=hub_api,
        )
    callback_type = type(
        "_BoundCheckpointManifestCallback",
        (_CheckpointManifestCallback, trainer_callback),
        {},
    )
    return callback_type(
        identity,
        model_repository_id=model_repository_id,
        trackio_space_id=trackio_space_id,
        tracking_settings=tracking_settings,
        hub_api=hub_api,
    )


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


def _build_trainer(
    dependencies: _TrainingDependencies,
    *,
    model: Any,
    training_arguments: Any,
    train_dataset: Any,
    validation_dataset: Any,
    data_collator: Any,
    checkpoint_identity: Mapping[str, object] | None,
    model_repository_id: str | None = None,
    trackio_space_id: str | None = None,
    tracking_settings: TrackioSettings | None = None,
    hub_api: Any | None = None,
) -> Any:
    trainer_values: dict[str, object] = {
        "model": model,
        "args": training_arguments,
        "train_dataset": train_dataset,
        "eval_dataset": validation_dataset,
        "data_collator": data_collator,
        "compute_metrics": _classification_metrics,
    }
    if checkpoint_identity is not None:
        trainer_values["callbacks"] = [
            _make_checkpoint_manifest_callback(
                checkpoint_identity,
                dependencies.trainer_callback,
                model_repository_id=model_repository_id,
                trackio_space_id=trackio_space_id,
                tracking_settings=tracking_settings,
                hub_api=hub_api,
            )
        ]
    return dependencies.trainer(**trainer_values)


def _run_trainer(trainer: Any, resume_from_checkpoint: Path | None) -> object:
    if resume_from_checkpoint is None:
        return trainer.train()
    return trainer.train(resume_from_checkpoint=str(resume_from_checkpoint))


def _load_training_dependencies() -> _TrainingDependencies:
    try:
        datasets_module = cast(Any, import_module("datasets"))
        transformers_module = cast(Any, import_module("transformers"))
    except ModuleNotFoundError as error:
        if error.name not in {"datasets", "transformers", "torch", "accelerate"}:
            raise
        raise TrainingError(
            "optional 'training' dependencies are required; install the training extra"
        ) from error
    return _TrainingDependencies(
        iterable_dataset=datasets_module.IterableDataset,
        auto_tokenizer=transformers_module.AutoTokenizer,
        auto_model_for_sequence_classification=(
            transformers_module.AutoModelForSequenceClassification
        ),
        data_collator_with_padding=transformers_module.DataCollatorWithPadding,
        training_arguments=transformers_module.TrainingArguments,
        trainer=transformers_module.Trainer,
        trainer_callback=transformers_module.TrainerCallback,
    )


def _load_checkpoint_publication_api() -> Any:
    try:
        hub = cast(Any, import_module("huggingface_hub"))
        return hub.HfApi()
    except Exception as error:
        raise TrainingError(
            "Hugging Face checkpoint publication requires the training dependencies"
        ) from error


def iter_split_training_records(
    rows_factory: Callable[[], Iterable[Mapping[str, object]]],
    *,
    split: DatasetSplit,
    validation_fraction: float = 0.2,
    seed: int = 42,
    contract: DatasetContract = LANDUSE_DATASET_CONTRACT,
) -> Iterator[TrainingRecord]:
    """Lazily adapt clean landuse examples to Trainer record dictionaries."""

    if split not in ("train", "validation"):
        raise TrainingError(f"unsupported dataset split: {split!r}")
    for example in iter_clean_training_examples(
        rows_factory,
        validation_fraction=validation_fraction,
        seed=seed,
        contract=contract,
    ):
        if example.split != split:
            continue
        yield {"text": example.text, "labels": LABEL_TO_ID[example.label]}


def _make_tokenized_dataset(
    dependencies: _TrainingDependencies,
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
        "eval_strategy": "steps",
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


def _freeze_encoder_for_head_training(model: Any) -> None:
    """Freeze the encoder while leaving its classification head trainable."""

    parameters = getattr(model, "parameters", None)
    head = getattr(model, "head", None)
    classifier = getattr(model, "classifier", None)
    head_parameters = getattr(head, "parameters", None)
    classifier_parameters = getattr(classifier, "parameters", None)
    if (
        not callable(parameters)
        or classifier is None
        or not callable(classifier_parameters)
    ):
        raise TrainingError(
            "model must expose a parameters() method and classifier head"
        )

    for parameter in parameters():
        parameter.requires_grad = False
    if callable(head_parameters):
        for parameter in head_parameters():
            parameter.requires_grad = True
    for parameter in classifier_parameters():
        parameter.requires_grad = True


@contextmanager
def _managed_training_environment(config: ProjectConfig) -> Iterator[None]:
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
        **settings_for(config).environment(),
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


def train_landuse_classifier(
    rows_factory: Callable[[], Iterable[Mapping[str, object]]] | None = None,
    *,
    config: TrainingConfig | None = None,
    project_config: ProjectConfig | None = None,
    contract: DatasetContract = LANDUSE_DATASET_CONTRACT,
    resume_from_checkpoint: Path | None = None,
    checkpoint_identity: Mapping[str, object] | None = None,
) -> TrainingResult:
    """Train and save a landuse classifier using only the clean stream boundary.

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
    tracking = settings_for(effective_project_config)
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

    with _managed_training_environment(effective_project_config):
        dependencies = _load_training_dependencies()
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
        _freeze_encoder_for_head_training(model)
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
            checkpoint_hub_api = _load_checkpoint_publication_api()
        trainer = _build_trainer(
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
        )
        train_output = _run_trainer(trainer, resume_from_checkpoint)
        trainer.save_model(str(output_directory))
        tokenizer.save_pretrained(str(output_directory))
        _write_model_card(
            output_directory,
            identity=_model_card_identity(
                checkpoint_identity,
                config=training_config,
                contract=contract,
            ),
            training_metrics=_metrics_for_model_card(train_output, trainer),
            trackio_space_id=(
                tracking.static_space_id if training_config.sync_trackio else None
            ),
        )
        model_publication = None
        if training_config.publish_to_hub:
            model_publication = publish_model_directory(
                output_directory,
                effective_project_config.target_model_repository_id,
            )
        if training_config.sync_trackio:
            _sync_static_trackio(
                tracking,
                failure_message="Trackio static snapshot failed",
            )
        tracking_space_id = (
            tracking.static_space_id if training_config.sync_trackio else None
        )

    return TrainingResult(
        output_directory=output_directory,
        train_output=train_output,
        model_publication=model_publication,
        tracking_space_id=tracking_space_id,
    )


__all__ = [
    "DEFAULT_MODEL_NAME",
    "ID_TO_LABEL",
    "LABEL_TO_ID",
    "TrainingConfig",
    "TrainingError",
    "TrainingRecord",
    "TrainingResult",
    "iter_split_training_records",
    "train_landuse_classifier",
]
