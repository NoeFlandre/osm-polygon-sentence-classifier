"""Optional training dependencies and Hugging Face Trainer wiring."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Literal, cast

from . import training_metrics as _training_metrics
from .tracking import TrackioSettings
from .training_freezing import TrainingError
from .training_publication import make_checkpoint_manifest_callback

ClassWeightMode = Literal["none", "balanced"]


@dataclass(frozen=True, slots=True)
class TrainingDependencies:
    """Lazily imported Hugging Face classes, kept injectable for unit tests."""

    iterable_dataset: Any
    auto_tokenizer: Any
    auto_model_for_sequence_classification: Any
    data_collator_with_padding: Any
    training_arguments: Any
    trainer: Any
    trainer_callback: Any | None = None


def classification_metrics(eval_prediction: Any) -> dict[str, float]:
    """Adapt metric input errors to the training error boundary."""

    try:
        return _training_metrics.classification_metrics(eval_prediction)
    except _training_metrics.MetricsInputError as error:
        raise TrainingError(str(error)) from error


def balanced_class_weights() -> tuple[float, float]:
    """Return normalized inverse-frequency weights from the pinned train split."""

    negative_count = 35_560
    positive_count = 8_648
    total = negative_count + positive_count
    return (
        total / (2 * negative_count),
        total / (2 * positive_count),
    )


def weighted_trainer_type(trainer_type: Any) -> Any:
    """Bind the fixed training-split class weights to a Trainer subclass."""

    class WeightedTrainer(trainer_type):
        def compute_loss(
            self,
            model: Any,
            inputs: dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Any = None,
        ) -> Any:
            del num_items_in_batch
            try:
                torch = import_module("torch")
            except ModuleNotFoundError as error:
                raise TrainingError(
                    "balanced loss requires the torch training dependency"
                ) from error
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = getattr(outputs, "logits", None)
            if logits is None and isinstance(outputs, Mapping):
                logits = outputs.get("logits")
            if logits is None:
                raise TrainingError(
                    "model output does not expose classification logits"
                )
            weights = torch.tensor(
                balanced_class_weights(),
                dtype=logits.dtype,
                device=logits.device,
            )
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=weights)
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer


def build_trainer(
    dependencies: TrainingDependencies,
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
    class_weight_mode: ClassWeightMode | None = None,
) -> Any:
    """Construct the configured Trainer and optional checkpoint callback."""

    trainer_values: dict[str, object] = {
        "model": model,
        "args": training_arguments,
        "train_dataset": train_dataset,
        "eval_dataset": validation_dataset,
        "data_collator": data_collator,
        "compute_metrics": classification_metrics,
    }
    if checkpoint_identity is not None:
        trainer_values["callbacks"] = [
            make_checkpoint_manifest_callback(
                checkpoint_identity,
                dependencies.trainer_callback,
                model_repository_id=model_repository_id,
                trackio_space_id=trackio_space_id,
                tracking_settings=tracking_settings,
                hub_api=hub_api,
            )
        ]
    trainer_type = dependencies.trainer
    if class_weight_mode == "balanced":
        trainer_type = weighted_trainer_type(trainer_type)
    return trainer_type(**trainer_values)


def run_trainer(trainer: Any, resume_from_checkpoint: Path | None) -> object:
    """Run Trainer with the exact optional resume argument used by the pipeline."""

    if resume_from_checkpoint is None:
        return trainer.train()
    return trainer.train(resume_from_checkpoint=str(resume_from_checkpoint))


def load_training_dependencies() -> TrainingDependencies:
    """Load optional Hugging Face training classes at the training boundary."""

    try:
        datasets_module = cast(Any, import_module("datasets"))
        transformers_module = cast(Any, import_module("transformers"))
    except ModuleNotFoundError as error:
        if error.name not in {"datasets", "transformers", "torch", "accelerate"}:
            raise
        raise TrainingError(
            "optional 'training' dependencies are required; install the training extra"
        ) from error
    return TrainingDependencies(
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


def load_checkpoint_publication_api() -> Any:
    """Load the Hub API only when checkpoint publication is explicitly enabled."""

    try:
        hub = cast(Any, import_module("huggingface_hub"))
        return hub.HfApi()
    except Exception as error:
        raise TrainingError(
            "Hugging Face checkpoint publication requires the training dependencies"
        ) from error


__all__ = [
    "ClassWeightMode",
    "TrainingDependencies",
    "balanced_class_weights",
    "build_trainer",
    "classification_metrics",
    "load_checkpoint_publication_api",
    "load_training_dependencies",
    "run_trainer",
    "weighted_trainer_type",
]
