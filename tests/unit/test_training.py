import os
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from osm_polygon_sentence_classifier.checkpointing import (
    find_latest_complete_checkpoint,
    write_checkpoint_manifest,
)
from osm_polygon_sentence_classifier.config import ProjectConfig
from osm_polygon_sentence_classifier.dataset_contract import (
    LANDUSE_DATASET_CONTRACT,
)
from osm_polygon_sentence_classifier.dataset_loader import split_for_polygon
from osm_polygon_sentence_classifier.tracking import (
    TRACKIO_STATIC_SPACE_ID,
)
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

    assert config.model_name_or_path == "jhu-clsp/mmBERT-small"
    assert config.learning_rate == 3e-4
    assert config.run_name == "landuse-mmbert-small-frozen-head"
    assert config.publish_to_hub is False
    assert config.sync_trackio is False
    assert config.output_subdirectory == Path("models/landuse")
    assert not config.output_subdirectory.is_absolute()
    assert config.max_steps > 0
    assert config.max_length > 0
    assert config.save_total_limit == 2

    with pytest.raises(AttributeError):
        config.max_steps = 1  # type: ignore[misc]  # ty: ignore[invalid-assignment]


@pytest.mark.parametrize("output_subdirectory", [Path("."), Path(".."), Path("/tmp")])
def test_training_config_rejects_an_unsafe_output_subdirectory(
    output_subdirectory: Path,
) -> None:
    with pytest.raises(TrainingError, match="output_subdirectory"):
        TrainingConfig(output_subdirectory=output_subdirectory)


@pytest.mark.parametrize("model_revision", ["unpinned", "A" * 40, 42])
def test_training_config_rejects_an_invalid_model_revision(
    model_revision: object,
) -> None:
    with pytest.raises(TrainingError, match="model_revision"):
        cast(Any, TrainingConfig)(model_revision=model_revision)


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


class _FakeParameter:
    def __init__(self) -> None:
        self.requires_grad = True


class _FakeClassifier:
    def __init__(self) -> None:
        self.parameters_list = [_FakeParameter()]

    def parameters(self) -> Iterable[_FakeParameter]:
        return self.parameters_list


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
    assert arguments["save_total_limit"] == 2
    assert arguments["remove_unused_columns"] is False
    assert arguments["trackio_static_space_id"] is False
    assert arguments["trackio_space_id"] is None
    assert arguments["trackio_bucket_id"] is None
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


def test_training_resumes_from_a_checkpoint_and_registers_identity_callback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_classifier import training

    dependencies = training._TrainingDependencies(
        iterable_dataset=_FakeDataset,
        auto_tokenizer=_FakeTokenizer,
        auto_model_for_sequence_classification=_FakeModel,
        data_collator_with_padding=_FakeDataCollator,
        training_arguments=_FakeTrainingArguments,
        trainer=_FakeTrainer,
    )
    monkeypatch.setattr(training, "_load_training_dependencies", lambda: dependencies)
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


def test_checkpoint_callback_writes_identity_after_a_save(tmp_path: Path) -> None:
    from osm_polygon_sentence_classifier import training

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

    training._CheckpointManifestCallback(identity).on_save(
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
    from osm_polygon_sentence_classifier import training

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
    sync_calls: list[object] = []

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

    monkeypatch.setattr(training, "publish_checkpoint_directory", publish)
    monkeypatch.setattr(
        training,
        "sync_project_to_static_space",
        lambda settings: sync_calls.append(settings) or TRACKIO_STATIC_SPACE_ID,
    )

    training._CheckpointManifestCallback(
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


def test_checkpoint_callback_queues_hub_publication_until_training_end(
    tmp_path: Path,
) -> None:
    from osm_polygon_sentence_classifier import training

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

    callback = training._CheckpointManifestCallback(
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
    from osm_polygon_sentence_classifier import training

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

    callback = training._CheckpointManifestCallback(
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
    from osm_polygon_sentence_classifier import training

    control = object()

    result = training._CheckpointManifestCallback({}).on_init_end(
        args=SimpleNamespace(),
        state=SimpleNamespace(),
        control=control,
    )

    assert result is control


def test_checkpoint_callback_uses_the_trainer_callback_base() -> None:
    from osm_polygon_sentence_classifier import training

    dependencies = training._TrainingDependencies(
        iterable_dataset=_FakeDataset,
        auto_tokenizer=_FakeTokenizer,
        auto_model_for_sequence_classification=_FakeModel,
        data_collator_with_padding=_FakeDataCollator,
        training_arguments=_FakeTrainingArguments,
        trainer=_FakeTrainer,
        trainer_callback=_FakeTrainerCallback,
    )
    _FakeTrainer.init_calls.clear()

    training._build_trainer(
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
    from osm_polygon_sentence_classifier import training

    def missing_dependency(module_name: str) -> Any:
        raise ModuleNotFoundError(name=module_name)

    monkeypatch.setattr(training, "import_module", missing_dependency)

    with pytest.raises(TrainingError, match="optional 'training' dependencies"):
        training._load_training_dependencies()


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

    dependencies = training._TrainingDependencies(
        iterable_dataset=_FakeDataset,
        auto_tokenizer=_FakeTokenizer,
        auto_model_for_sequence_classification=_FakeModel,
        data_collator_with_padding=_FakeDataCollator,
        training_arguments=_FakeTrainingArguments,
        trainer=_FakeTrainer,
    )
    monkeypatch.setattr(training, "_load_training_dependencies", lambda: dependencies)
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

    dependencies = training._TrainingDependencies(
        iterable_dataset=_FakeDataset,
        auto_tokenizer=_FakeTokenizer,
        auto_model_for_sequence_classification=_FakeModel,
        data_collator_with_padding=_FakeDataCollator,
        training_arguments=_FakeTrainingArguments,
        trainer=_FakeTrainer,
    )
    monkeypatch.setattr(training, "_load_training_dependencies", lambda: dependencies)
    checkpoint_hub_api = object()
    monkeypatch.setattr(
        training, "_load_checkpoint_publication_api", lambda: checkpoint_hub_api
    )
    publication_calls: list[Path] = []
    monkeypatch.setattr(
        training,
        "publish_model_directory",
        lambda directory, repository_id: publication_calls.append(directory)
        or training.ModelPublicationResult(
            repository_id=repository_id,
            commit_id="d" * 40,
            commit_url="https://huggingface.co/test/commit/" + "d" * 40,
            files=("config.json", "model.safetensors"),
        ),
    )
    sync_calls: list[object] = []
    monkeypatch.setattr(
        training,
        "sync_project_to_static_space",
        lambda settings: sync_calls.append(settings) or TRACKIO_STATIC_SPACE_ID,
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
    assert result.model_publication is not None
    assert result.model_publication.commit_id == "d" * 40
    assert result.tracking_space_id == TRACKIO_STATIC_SPACE_ID
    assert len(sync_calls) == 1
