from pathlib import Path
from typing import Any

import pytest

from osm_polygon_sentence_classifier.training_freezing import TrainingError
from osm_polygon_sentence_classifier.training_runtime import (
    TrainingDependencies,
    build_trainer,
    load_training_dependencies,
    run_trainer,
)


def test_load_training_dependencies_maps_hugging_face_classes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class DatasetsModule:
        IterableDataset = object

    class TransformersModule:
        AutoTokenizer = object
        AutoModelForSequenceClassification = object
        DataCollatorWithPadding = object
        TrainingArguments = object
        Trainer = object
        TrainerCallback = object

    modules = {
        "datasets": DatasetsModule,
        "transformers": TransformersModule,
    }
    monkeypatch.setattr(
        "osm_polygon_sentence_classifier.training_runtime.import_module",
        modules.__getitem__,
    )

    dependencies = load_training_dependencies()

    assert dependencies.iterable_dataset is DatasetsModule.IterableDataset
    assert dependencies.auto_tokenizer is TransformersModule.AutoTokenizer
    assert (
        dependencies.auto_model_for_sequence_classification
        is TransformersModule.AutoModelForSequenceClassification
    )
    assert dependencies.trainer_callback is TransformersModule.TrainerCallback


def test_load_training_dependencies_reports_missing_training_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_dependency(name: str) -> Any:
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(
        "osm_polygon_sentence_classifier.training_runtime.import_module",
        missing_dependency,
    )

    with pytest.raises(TrainingError, match="optional 'training' dependencies"):
        load_training_dependencies()


def test_build_and_run_trainer_preserve_callback_and_resume_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeTrainer:
        init_calls: list[dict[str, object]] = []
        train_calls: list[str | None] = []

        def __init__(self, **kwargs: object) -> None:
            self.init_calls.append(kwargs)

        def train(self, resume_from_checkpoint: str | None = None) -> object:
            self.train_calls.append(resume_from_checkpoint)
            return "trained"

    callback = object()
    monkeypatch.setattr(
        "osm_polygon_sentence_classifier.training_runtime.make_checkpoint_manifest_callback",
        lambda *args, **kwargs: callback,
    )
    dependencies = TrainingDependencies(
        iterable_dataset=object(),
        auto_tokenizer=object(),
        auto_model_for_sequence_classification=object(),
        data_collator_with_padding=object(),
        training_arguments=object(),
        trainer=FakeTrainer,
        trainer_callback=None,
    )

    trainer = build_trainer(
        dependencies,
        model=object(),
        training_arguments=object(),
        train_dataset=object(),
        validation_dataset=object(),
        data_collator=object(),
        checkpoint_identity={"run_id": "a" * 20},
    )
    result = run_trainer(trainer, Path("models/landuse/checkpoint-7"))

    assert isinstance(trainer, FakeTrainer)
    assert FakeTrainer.init_calls[0]["callbacks"] == [callback]
    assert result == "trained"
    assert FakeTrainer.train_calls == ["models/landuse/checkpoint-7"]
