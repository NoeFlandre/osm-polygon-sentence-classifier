"""Managed, streaming training orchestration for the landuse classifier."""

import math
import os
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, TypedDict, cast

from .config import ProjectConfig
from .dataset_contract import LANDUSE_DATASET_CONTRACT, DatasetContract
from .dataset_loader import (
    DatasetSplit,
    TrainingLabel,
    iter_clean_training_examples,
    load_streaming_rows,
)
from .paths import ManagedPaths
from .tracking import settings_for

LabelId = Literal[0, 1]

LABEL_TO_ID: dict[TrainingLabel, LabelId] = {"no": 0, "yes": 1}
ID_TO_LABEL: dict[int, str] = {0: "no", 1: "yes"}


class TrainingError(RuntimeError):
    """Raised when training dependencies or configuration are unavailable."""


class TrainingRecord(TypedDict):
    """One clean, split-assigned record before tokenization."""

    text: str
    labels: LabelId


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Small, explicit configuration for one landuse training run."""

    model_name_or_path: str = "distilbert-base-multilingual-cased"
    output_subdirectory: Path = Path("models/landuse")
    validation_fraction: float = 0.2
    seed: int = 42
    max_length: int = 256
    max_steps: int = 1_000
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 8
    learning_rate: float = 5e-5
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 100
    run_name: str = "landuse-baseline"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.model_name_or_path, str)
            or not self.model_name_or_path.strip()
        ):
            raise TrainingError("model_name_or_path must be a non-empty string")
        if not isinstance(self.run_name, str) or not self.run_name.strip():
            raise TrainingError("run_name must be a non-empty string")

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


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Local result of a completed Trainer run."""

    output_directory: Path
    train_output: object


@dataclass(frozen=True, slots=True)
class _TrainingDependencies:
    """Lazily imported Hugging Face classes, kept injectable for unit tests."""

    iterable_dataset: Any
    auto_tokenizer: Any
    auto_model_for_sequence_classification: Any
    data_collator_with_padding: Any
    training_arguments: Any
    trainer: Any


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
    )


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
        "report_to": ["trackio"],
        "project": tracking_project,
        "run_name": config.run_name,
        "remove_unused_columns": False,
    }


@contextmanager
def _managed_training_environment(config: ProjectConfig) -> Iterator[None]:
    paths = ManagedPaths(config)
    values = {
        "HF_HOME": str(paths.child("cache/huggingface")),
        **settings_for(config).environment(),
    }
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
) -> TrainingResult:
    """Train and save a landuse classifier using only the clean stream boundary.

    The optional dependency group is imported only when this function is called.
    Datasets, model caches, checkpoints, and Trackio state are directed beneath
    the approved external data root. This function does not authenticate,
    upload, publish, or submit a remote job.
    """

    training_config = config or TrainingConfig()
    effective_project_config = project_config or ProjectConfig()
    paths = ManagedPaths(effective_project_config)
    output_directory = paths.child(training_config.output_subdirectory)
    model_cache_directory = paths.child("cache/huggingface/models")
    tracking = settings_for(effective_project_config)

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
        tokenizer = dependencies.auto_tokenizer.from_pretrained(
            training_config.model_name_or_path,
            cache_dir=str(model_cache_directory),
            use_fast=True,
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
        model = dependencies.auto_model_for_sequence_classification.from_pretrained(
            training_config.model_name_or_path,
            cache_dir=str(model_cache_directory),
            num_labels=len(LABEL_TO_ID),
            id2label=ID_TO_LABEL,
            label2id=LABEL_TO_ID,
        )
        training_arguments = dependencies.training_arguments(
            **_training_argument_values(
                training_config,
                output_directory=output_directory,
                tracking_project=tracking.project,
            )
        )
        data_collator = dependencies.data_collator_with_padding(tokenizer=tokenizer)
        trainer = dependencies.trainer(
            model=model,
            args=training_arguments,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            data_collator=data_collator,
        )
        train_output = trainer.train()
        trainer.save_model(str(output_directory))
        tokenizer.save_pretrained(str(output_directory))

    return TrainingResult(
        output_directory=output_directory,
        train_output=train_output,
    )


__all__ = [
    "ID_TO_LABEL",
    "LABEL_TO_ID",
    "TrainingConfig",
    "TrainingError",
    "TrainingRecord",
    "TrainingResult",
    "iter_split_training_records",
    "train_landuse_classifier",
]
