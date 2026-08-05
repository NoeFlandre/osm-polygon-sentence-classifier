import os
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any, cast

import pytest

from osm_polygon_sentence_classifier.config import ProjectConfig
from osm_polygon_sentence_classifier.dataset_contract import (
    LANDUSE_DATASET_CONTRACT,
)
from osm_polygon_sentence_classifier.dataset_loader import split_for_polygon
from osm_polygon_sentence_classifier.training import (
    TrainingConfig,
    TrainingError,
    TrainingResult,
    iter_split_training_records,
    train_landuse_classifier,
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


def _polygon_for_split(split: str) -> str:
    for index in range(100):
        polygon_id = f"polygon-{index}"
        if split_for_polygon(polygon_id, validation_fraction=0.5) == split:
            return polygon_id
    raise AssertionError(f"no polygon found for {split}")


def test_training_config_is_frozen_and_uses_a_managed_relative_output() -> None:
    config = TrainingConfig()

    assert config.output_subdirectory == Path("models/landuse")
    assert not config.output_subdirectory.is_absolute()
    assert config.max_steps > 0
    assert config.max_length > 0

    with pytest.raises(AttributeError):
        config.max_steps = 1  # type: ignore[misc]  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize("output_subdirectory", [Path("."), Path(".."), Path("/tmp")])
def test_training_config_rejects_an_unsafe_output_subdirectory(
    output_subdirectory: Path,
) -> None:
    with pytest.raises(TrainingError, match="output_subdirectory"):
        TrainingConfig(output_subdirectory=output_subdirectory)


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


class _FakeModel:
    from_pretrained_calls: list[dict[str, object]] = []

    @classmethod
    def from_pretrained(cls, *args: object, **kwargs: object) -> "_FakeModel":
        cls.from_pretrained_calls.append({"args": args, **kwargs})
        return cls()


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
    environment_during_train: dict[str, str | None] = {}
    train_output = object()

    def __init__(self, **kwargs: object) -> None:
        self.init_calls.append(kwargs)

    def train(self) -> object:
        _FakeTrainer.environment_during_train = {
            "HF_HOME": os.environ.get("HF_HOME"),
            "TRACKIO_DIR": os.environ.get("TRACKIO_DIR"),
        }
        return self.train_output

    def save_model(self, output_directory: str) -> None:
        self.save_model_calls.append(output_directory)


def test_training_wires_managed_streams_tokenizer_trainer_and_tracking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_classifier import training

    _FakeDataset.created.clear()
    _FakeTokenizer.from_pretrained_calls.clear()
    _FakeTokenizer.save_pretrained_calls.clear()
    _FakeModel.from_pretrained_calls.clear()
    _FakeTrainingArguments.calls.clear()
    _FakeDataCollator.calls.clear()
    _FakeTrainer.init_calls.clear()
    _FakeTrainer.save_model_calls.clear()
    _FakeTrainer.environment_during_train.clear()

    dependencies = training._TrainingDependencies(
        iterable_dataset=_FakeDataset,
        auto_tokenizer=_FakeTokenizer,
        auto_model_for_sequence_classification=_FakeModel,
        data_collator_with_padding=_FakeDataCollator,
        training_arguments=_FakeTrainingArguments,
        trainer=_FakeTrainer,
    )
    monkeypatch.setattr(training, "_load_training_dependencies", lambda: dependencies)

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
    arguments = _FakeTrainingArguments.calls[0]
    assert arguments["output_dir"] == str(ProjectConfig().data_root / "models/landuse")
    assert arguments["max_steps"] == 12
    assert arguments["eval_strategy"] == "steps"
    assert arguments["report_to"] == ["trackio"]
    assert arguments["project"] == "osm-polygon-sentence-classifier"
    assert arguments["run_name"] == "test-run"
    assert arguments["remove_unused_columns"] is False
    assert _FakeTrainer.init_calls[0]["train_dataset"] is _FakeDataset.created[0]
    assert _FakeTrainer.init_calls[0]["eval_dataset"] is _FakeDataset.created[1]
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


def test_training_reports_missing_optional_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_classifier import training

    def missing_dependency(module_name: str) -> Any:
        raise ModuleNotFoundError(name=module_name)

    monkeypatch.setattr(training, "import_module", missing_dependency)

    with pytest.raises(TrainingError, match="optional 'training' dependencies"):
        training._load_training_dependencies()
