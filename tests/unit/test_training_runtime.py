from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from osm_polygon_sentence_classifier import training_runtime
from osm_polygon_sentence_classifier.training_freezing import TrainingError
from osm_polygon_sentence_classifier.training_metrics import MetricsInputError
from osm_polygon_sentence_classifier.training_runtime import (
    TrainingDependencies,
    _checkpoint_global_step,
    _load_completed_checkpoint,
    _restore_state_attributes,
    _restore_trainer_state,
    _trainer_max_steps,
    build_trainer,
    load_training_dependencies,
    run_trainer,
    weighted_trainer_type,
)


def test_runtime_classification_metrics_forwards_the_evaluation_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = object()
    calls: list[object] = []
    result = {"accuracy": 0.8}

    def metrics(received: object) -> dict[str, float]:
        calls.append(received)
        return result

    monkeypatch.setattr(
        training_runtime._training_metrics, "classification_metrics", metrics
    )

    assert training_runtime.classification_metrics(payload) is result
    assert calls == [payload]


def test_runtime_classification_metrics_wraps_invalid_input_with_the_original_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cause = MetricsInputError("evaluation payload is invalid")

    def metrics(_payload: object) -> dict[str, float]:
        raise cause

    monkeypatch.setattr(
        training_runtime._training_metrics, "classification_metrics", metrics
    )

    with pytest.raises(TrainingError, match="^evaluation payload is invalid$") as error:
        training_runtime.classification_metrics(object())

    assert str(error.value) == "evaluation payload is invalid"
    assert error.value.__cause__ is cause


def test_load_checkpoint_publication_api_configures_http_and_constructs_hub_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    api = object()

    class HubModule:
        @staticmethod
        def HfApi() -> object:
            calls.append("HfApi")
            return api

    monkeypatch.setattr(
        training_runtime,
        "configure_huggingface_http",
        lambda: calls.append("configure"),
    )
    monkeypatch.setattr(
        training_runtime,
        "import_module",
        lambda name: calls.append(name) or HubModule,
    )

    assert training_runtime.load_checkpoint_publication_api() is api
    assert calls == ["configure", "huggingface_hub", "HfApi"]


def test_load_checkpoint_publication_api_wraps_dependency_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cause = RuntimeError("Hub import failed")
    monkeypatch.setattr(
        training_runtime,
        "configure_huggingface_http",
        lambda: (_ for _ in ()).throw(cause),
    )

    with pytest.raises(
        TrainingError,
        match="^Hugging Face checkpoint publication requires the training dependencies$",
    ) as error:
        training_runtime.load_checkpoint_publication_api()

    assert str(error.value) == (
        "Hugging Face checkpoint publication requires the training dependencies"
    )
    assert error.value.__cause__ is cause


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
    assert (
        dependencies.data_collator_with_padding
        is TransformersModule.DataCollatorWithPadding
    )
    assert dependencies.training_arguments is TransformersModule.TrainingArguments
    assert dependencies.trainer is TransformersModule.Trainer
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


@pytest.mark.parametrize(
    "missing_name", ["datasets", "transformers", "torch", "accelerate"]
)
def test_load_training_dependencies_handles_each_supported_missing_name(
    monkeypatch: pytest.MonkeyPatch,
    missing_name: str,
) -> None:
    missing = ModuleNotFoundError(name=missing_name)

    def missing_dependency(_: str) -> Any:
        raise missing

    monkeypatch.setattr(
        "osm_polygon_sentence_classifier.training_runtime.import_module",
        missing_dependency,
    )

    with pytest.raises(
        TrainingError,
        match="^optional 'training' dependencies are required; install the training extra$",
    ) as caught:
        load_training_dependencies()

    assert caught.value.__cause__ is missing


def test_load_training_dependencies_reraises_an_unrelated_missing_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = ModuleNotFoundError(name="unrelated")

    def missing_dependency(_: str) -> Any:
        raise missing

    monkeypatch.setattr(
        "osm_polygon_sentence_classifier.training_runtime.import_module",
        missing_dependency,
    )

    with pytest.raises(ModuleNotFoundError) as caught:
        load_training_dependencies()

    assert caught.value is missing


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
    callback_calls: list[dict[str, object]] = []

    def make_callback(*args: object, **kwargs: object) -> object:
        callback_calls.append({"args": args, **kwargs})
        return callback

    monkeypatch.setattr(
        "osm_polygon_sentence_classifier.training_runtime.make_checkpoint_manifest_callback",
        make_callback,
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

    model = object()
    training_arguments = object()
    train_dataset = object()
    validation_dataset = object()
    data_collator = object()
    trainer = build_trainer(
        dependencies,
        model=model,
        training_arguments=training_arguments,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        data_collator=data_collator,
        checkpoint_identity={"run_id": "a" * 20},
    )
    result = run_trainer(trainer, Path("models/landuse/checkpoint-7"))

    assert isinstance(trainer, FakeTrainer)
    assert FakeTrainer.init_calls[0]["callbacks"] == [callback]
    assert FakeTrainer.init_calls[0]["compute_metrics"] is not None
    assert FakeTrainer.init_calls[0]["model"] is model
    assert FakeTrainer.init_calls[0]["args"] is training_arguments
    assert FakeTrainer.init_calls[0]["train_dataset"] is train_dataset
    assert FakeTrainer.init_calls[0]["eval_dataset"] is validation_dataset
    assert FakeTrainer.init_calls[0]["data_collator"] is data_collator
    assert callback_calls[0]["hub_checkpoint_steps"] == 1
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
    original_state = trainer.state

    result = run_trainer(trainer, checkpoint)

    assert result is None
    assert trainer.load_calls == [str(checkpoint)]
    assert trainer.train_calls == []
    assert trainer.state.global_step == 12
    assert trainer.state.log_history == [{"loss": 0.1}]
    assert trainer.state is not original_state


def test_run_trainer_restores_state_attributes_without_hugging_face_loader(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint-12"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(
        '{"global_step": 12, "epoch": 3.5, "log_history": [{"loss": 0.1}]}',
        encoding="utf-8",
    )

    class FallbackTrainer:
        def __init__(self) -> None:
            self.args = SimpleNamespace(max_steps=12)
            self.state = SimpleNamespace(global_step=0, epoch=0, log_history=[])
            self.load_calls: list[str] = []
            self.callback_state_loaded = False

        def _load_from_checkpoint(self, path: str) -> None:
            self.load_calls.append(path)

        def _load_callback_state(self) -> None:
            self.callback_state_loaded = True

        def train(self, resume_from_checkpoint: str | None = None) -> object:
            raise AssertionError(f"unexpected replay: {resume_from_checkpoint}")

    trainer = FallbackTrainer()
    original_state = trainer.state

    assert run_trainer(trainer, checkpoint) is None
    assert trainer.load_calls == [str(checkpoint)]
    assert trainer.state.global_step == 12
    assert trainer.state.epoch == 3.5
    assert trainer.state.log_history == [{"loss": 0.1}]
    assert trainer.callback_state_loaded is True
    assert trainer.state is original_state


@pytest.mark.parametrize(
    "state_text",
    [
        "not-json",
        "[]",
        '{"global_step": true}',
        '{"global_step": "12"}',
    ],
)
def test_run_trainer_replays_when_checkpoint_step_is_invalid(
    tmp_path: Path,
    state_text: str,
) -> None:
    checkpoint = tmp_path / "checkpoint-12"
    checkpoint.mkdir()
    (checkpoint / "trainer_state.json").write_text(state_text, encoding="utf-8")

    class FakeTrainer:
        args = SimpleNamespace(max_steps=12)

        def __init__(self) -> None:
            self.train_calls: list[str | None] = []

        def train(self, resume_from_checkpoint: str | None = None) -> object:
            self.train_calls.append(resume_from_checkpoint)
            return "replayed"

    trainer = FakeTrainer()

    assert run_trainer(trainer, checkpoint) == "replayed"
    assert trainer.train_calls == [str(checkpoint)]


def test_checkpoint_global_step_reads_utf8_with_an_explicit_encoding() -> None:
    class FakePath:
        encoding: str | None = None

        def read_text(self, *, encoding: str | None) -> str:
            self.encoding = encoding
            return '{"global_step": 12}'

    path = FakePath()

    assert _checkpoint_global_step(path) == 12  # ty: ignore[invalid-argument-type]
    assert path.encoding == "utf-8"


def test_load_completed_checkpoint_uses_the_canonical_state_filename(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []

    class FakeCheckpoint:
        def __truediv__(self, name: str) -> object:
            seen.append(name)
            return object()

    class Trainer:
        args = SimpleNamespace(max_steps=12)

        def _load_from_checkpoint(self, _: str) -> None:
            return None

    monkeypatch.setattr(
        "osm_polygon_sentence_classifier.training_runtime._checkpoint_global_step",
        lambda _: 12,
    )
    monkeypatch.setattr(
        "osm_polygon_sentence_classifier.training_runtime._restore_trainer_state",
        lambda *_: None,
    )
    monkeypatch.setattr(
        "osm_polygon_sentence_classifier.training_runtime._restore_callback_state",
        lambda _: None,
    )

    assert _load_completed_checkpoint(Trainer(), cast(Any, FakeCheckpoint())) is True
    assert seen == ["trainer_state.json"]


def test_load_completed_checkpoint_rejects_a_trainer_without_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Trainer:
        args = SimpleNamespace(max_steps=12)

    monkeypatch.setattr(
        "osm_polygon_sentence_classifier.training_runtime._checkpoint_global_step",
        lambda _: 12,
    )

    assert _load_completed_checkpoint(Trainer(), Path("checkpoint")) is False


@pytest.mark.parametrize("max_steps", [0, -1, True, None, "12"])
def test_trainer_max_steps_requires_a_positive_integer(max_steps: object) -> None:
    trainer = SimpleNamespace(args=SimpleNamespace(max_steps=max_steps))
    assert _trainer_max_steps(trainer) is None


def test_trainer_max_steps_accepts_one() -> None:
    assert _trainer_max_steps(SimpleNamespace(args=SimpleNamespace(max_steps=1))) == 1


def test_restore_state_attributes_reads_utf8_with_an_explicit_encoding() -> None:
    class FakePath:
        encoding: str | None = None

        def read_text(self, *, encoding: str | None) -> str:
            self.encoding = encoding
            return '{"global_step": 12}'

    state = SimpleNamespace(global_step=0, epoch=None, log_history=[])
    path = FakePath()

    _restore_state_attributes(state, path)  # ty: ignore[invalid-argument-type]

    assert path.encoding == "utf-8"
    assert state.global_step == 12


def test_restore_trainer_state_tolerates_a_trainer_without_state() -> None:
    _restore_trainer_state(SimpleNamespace(), Path("unused"))


def test_weighted_trainer_computes_balanced_loss_for_attribute_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: dict[str, object] = {}

    class Functional:
        @staticmethod
        def cross_entropy(logits: object, labels: object, *, weight: object) -> str:
            calls["cross_entropy"] = (logits, labels, weight)
            return "loss"

    class Torch:
        class nn:
            functional = Functional

        @staticmethod
        def tensor(values: object, *, dtype: object, device: object) -> str:
            calls["tensor"] = (values, dtype, device)
            return "weights"

    monkeypatch.setattr(
        "osm_polygon_sentence_classifier.training_runtime.import_module",
        lambda name: Torch if name == "torch" else None,
    )

    logits = SimpleNamespace(dtype="dtype", device="device")

    class Output:
        def __init__(self) -> None:
            self.logits = logits

    received: dict[str, object] = {}
    outputs: list[Output] = []

    def model(**inputs: object) -> Output:
        received.update(inputs)
        output = Output()
        outputs.append(output)
        return output

    inputs = {"input_ids": "ids", "labels": "labels"}
    trainer = weighted_trainer_type(object)()

    result = trainer.compute_loss(
        model,
        inputs,
        return_outputs=True,
        num_items_in_batch=8,
    )

    assert result == ("loss", outputs[0])
    assert received == {"input_ids": "ids"}
    assert inputs == {"input_ids": "ids"}
    assert calls["tensor"] == (
        (44_208 / (2 * 35_560), 44_208 / (2 * 8_648)),
        "dtype",
        "device",
    )
    assert calls["cross_entropy"] == (
        logits,
        "labels",
        "weights",
    )


def test_weighted_trainer_accepts_mapping_outputs_and_default_return_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Functional:
        @staticmethod
        def cross_entropy(_: object, __: object, *, weight: object) -> str:
            assert weight == "weights"
            return "loss"

    class Torch:
        class nn:
            functional = Functional

        @staticmethod
        def tensor(*_: object, **__: object) -> str:
            return "weights"

    monkeypatch.setattr(
        "osm_polygon_sentence_classifier.training_runtime.import_module",
        lambda _: Torch,
    )

    trainer = weighted_trainer_type(object)()
    logits = SimpleNamespace(dtype="dtype", device="device")
    result = trainer.compute_loss(
        lambda **_: {"logits": logits},
        {"labels": "labels"},
    )

    assert result == "loss"


def test_weighted_trainer_prefers_attribute_logits_on_mapping_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Logits:
        dtype = "dtype"
        device = "device"

    logits = Logits()

    class MappingOutput(dict[str, object]):
        def __init__(self) -> None:
            super().__init__(logits="mapping-logits")
            self.logits = logits

    class Functional:
        @staticmethod
        def cross_entropy(actual: object, _: object, *, weight: object) -> str:
            assert actual is logits
            assert weight == "weights"
            return "loss"

    class Torch:
        class nn:
            functional = Functional

        @staticmethod
        def tensor(*_: object, **__: object) -> str:
            return "weights"

    monkeypatch.setattr(
        "osm_polygon_sentence_classifier.training_runtime.import_module",
        lambda _: Torch,
    )

    result = weighted_trainer_type(object)().compute_loss(
        lambda **_: MappingOutput(),
        {"labels": "labels"},
    )

    assert result == "loss"


def test_weighted_trainer_reports_missing_logits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Torch:
        class nn:
            class functional:
                cross_entropy = staticmethod(lambda *_, **__: "unreachable")

        tensor = staticmethod(lambda *_, **__: "unreachable")

    monkeypatch.setattr(
        "osm_polygon_sentence_classifier.training_runtime.import_module",
        lambda _: Torch,
    )

    with pytest.raises(
        TrainingError,
        match="^model output does not expose classification logits$",
    ):
        weighted_trainer_type(object)().compute_loss(
            lambda **_: {},
            {"labels": "labels"},
        )


def test_weighted_trainer_reports_missing_torch_with_the_original_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing = ModuleNotFoundError(name="torch")

    def import_missing(_: str) -> object:
        raise missing

    monkeypatch.setattr(
        "osm_polygon_sentence_classifier.training_runtime.import_module",
        import_missing,
    )

    with pytest.raises(
        TrainingError,
        match="^balanced loss requires the torch training dependency$",
    ) as caught:
        weighted_trainer_type(object)().compute_loss(
            lambda **_: {},
            {"labels": "labels"},
        )

    assert caught.value.__cause__ is missing
