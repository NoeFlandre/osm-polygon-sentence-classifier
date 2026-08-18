from pathlib import Path
from types import SimpleNamespace
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


def test_run_trainer_loads_a_completed_checkpoint_without_replaying(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint-12"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        '{"global_step": 12, "log_history": [{"loss": 0.1}]}',
        encoding="utf-8",
    )

    class FakeState:
        def __init__(self) -> None:
            self.global_step = 0
            self.log_history: list[dict[str, float]] = []

        @classmethod
        def load_from_json(cls, path: str) -> "FakeState":
            import json

            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            state = cls()
            state.global_step = payload["global_step"]
            state.log_history = payload["log_history"]
            return state

    class FakeTrainer:
        def __init__(self) -> None:
            self.args = SimpleNamespace(max_steps=12)
            self.state = FakeState()
            self.load_calls: list[str] = []
            self.train_calls: list[str | None] = []

        def _load_from_checkpoint(self, path: str) -> None:
            self.load_calls.append(path)

        def train(self, resume_from_checkpoint: str | None = None) -> object:
            self.train_calls.append(resume_from_checkpoint)
            return "replayed"

    trainer = FakeTrainer()

    result = run_trainer(trainer, checkpoint)

    assert result is None
    assert trainer.load_calls == [str(checkpoint)]
    assert trainer.train_calls == []
    assert trainer.state.global_step == 12
    assert trainer.state.log_history == [{"loss": 0.1}]
