import builtins
from collections.abc import Iterable, Mapping
from dataclasses import FrozenInstanceError, is_dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast, get_args

import pytest

import osm_polygon_sentence_classifier.dataset_loader as dataset_loader
from osm_polygon_sentence_classifier.config import ProjectConfig
from osm_polygon_sentence_classifier.dataset_contract import (
    LANDUSE_DATASET_CONTRACT,
    DatasetContractError,
)
from osm_polygon_sentence_classifier.dataset_loader import (
    DatasetLoaderError,
    DatasetSplit,
    TrainingExample,
    TrainingLabel,
    iter_clean_training_examples,
    iter_training_examples,
    load_streaming_rows,
    split_for_polygon,
)


def _row(
    *,
    sentence_id: object = "sentence-1",
    polygon_id: object = "polygon-1",
    text: str = "Normalized sentence.",
    label: str = "yes",
    content_hash: object = None,
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
    assert len(row) == 39
    return row


def test_training_types_are_limited_to_the_contract_values() -> None:
    assert get_args(DatasetSplit) == ("train", "validation", "test")
    assert get_args(TrainingLabel) == ("no", "yes")


def test_split_for_polygon_is_deterministic_and_returns_a_supported_split() -> None:
    result = split_for_polygon("polygon-1", validation_fraction=0.2, seed=42)

    assert result in ("train", "validation")
    assert result == split_for_polygon("polygon-1", validation_fraction=0.2, seed=42)


def test_split_for_polygon_supports_a_deterministic_held_out_test_split() -> None:
    splits = {
        split_for_polygon(
            f"polygon-{index}",
            validation_fraction=0.1,
            test_fraction=0.1,
            seed=42,
        )
        for index in range(100)
    }

    assert splits == {"train", "validation", "test"}
    assert split_for_polygon(
        "polygon-1", validation_fraction=0.1, test_fraction=0.1, seed=42
    ) == split_for_polygon(
        "polygon-1", validation_fraction=0.1, test_fraction=0.1, seed=42
    )


def test_split_for_polygon_respects_exact_fraction_boundaries(monkeypatch) -> None:
    class _Digest:
        def digest(self) -> bytes:
            return (1).to_bytes(8, byteorder="big") + b"unused"

    monkeypatch.setattr(dataset_loader.hashlib, "sha256", lambda _value: _Digest())
    boundary = 1 / 2**64

    assert (
        split_for_polygon(
            "polygon-1",
            validation_fraction=boundary,
            test_fraction=0.1,
        )
        == "test"
    )
    assert (
        split_for_polygon(
            "polygon-1",
            validation_fraction=0.0,
            test_fraction=boundary,
        )
        == "train"
    )


@pytest.mark.parametrize(
    ("validation_fraction", "test_fraction"),
    [(-0.1, 0.1), (0.1, -0.1), (0.8, 0.3)],
)
def test_split_for_polygon_rejects_invalid_three_way_fractions(
    validation_fraction: float, test_fraction: float
) -> None:
    with pytest.raises(
        DatasetLoaderError,
        match=(
            r"\Avalidation and test fractions must be finite, non-negative, "
            r"and sum to at most 1\Z"
        ),
    ):
        split_for_polygon(
            "polygon-1",
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.0, True), (1.0, True), (-0.1, False), (1.1, False)],
)
def test_split_fraction_validator_enforces_the_closed_unit_interval(
    value: float, expected: bool
) -> None:
    assert dataset_loader._is_valid_fraction(value) is expected


@pytest.mark.parametrize(
    "fraction",
    [-0.01, 1.01, float("nan"), float("inf"), True, False, "0.2", None],
)
def test_split_for_polygon_rejects_invalid_validation_fractions(
    fraction: float,
) -> None:
    with pytest.raises(
        DatasetLoaderError,
        match=r"\Avalidation_fraction must be a finite number between 0 and 1\Z",
    ):
        split_for_polygon("polygon-1", validation_fraction=fraction)


def test_split_for_polygon_accepts_zero_and_one_boundaries() -> None:
    assert split_for_polygon("polygon-1", validation_fraction=0.0) == "train"
    assert split_for_polygon("polygon-1", validation_fraction=1.0) == "validation"


def test_split_for_polygon_accepts_fractions_that_sum_to_one() -> None:
    assert split_for_polygon(
        "polygon-1",
        validation_fraction=0.5,
        test_fraction=0.5,
    ) in {"train", "validation", "test"}


@pytest.mark.parametrize("polygon_id", ["", "   ", None, 123])
def test_split_for_polygon_rejects_blank_or_non_string_ids(
    polygon_id: object,
) -> None:
    with pytest.raises(
        DatasetLoaderError,
        match=r"\Apolygon_id must be a non-empty string\Z",
    ):
        split_for_polygon(cast(str, polygon_id))


def test_iterator_skips_uncertain_and_keeps_no_and_yes_examples() -> None:
    rows = [
        _row(sentence_id="uncertain", polygon_id="polygon-u", label="uncertain"),
        _row(
            sentence_id="negative", polygon_id="polygon-n", text="No text.", label="no"
        ),
        _row(
            sentence_id="positive",
            polygon_id="polygon-p",
            text="Yes text.",
            label="yes",
        ),
    ]

    examples = list(iter_training_examples(rows))

    assert [
        (example.sentence_id, example.text, example.label) for example in examples
    ] == [
        ("negative", "No text.", "no"),
        ("positive", "Yes text.", "yes"),
    ]


def test_iterator_assigns_all_sentences_from_one_polygon_to_one_split() -> None:
    rows = [
        _row(sentence_id="sentence-1", polygon_id="shared", label="no"),
        _row(sentence_id="sentence-2", polygon_id="shared", label="yes"),
    ]

    examples = list(iter_training_examples(rows))

    assert {example.split for example in examples} in ({"train"}, {"validation"})


def test_iterator_forwards_the_seed_to_split_assignment() -> None:
    polygon_id = next(
        f"polygon-{index}"
        for index in range(100)
        if split_for_polygon(f"polygon-{index}", validation_fraction=0.5, seed=42)
        != split_for_polygon(f"polygon-{index}", validation_fraction=0.5, seed=43)
    )

    example = next(
        iter_training_examples([_row(polygon_id=polygon_id)], validation_fraction=0.5)
    )

    assert example.split == split_for_polygon(
        polygon_id,
        validation_fraction=0.5,
        seed=42,
    )


def test_iterator_preserves_polygon_id_and_forwards_a_non_default_seed() -> None:
    polygon_id = next(
        f"polygon-{index}"
        for index in range(100)
        if split_for_polygon(f"polygon-{index}", validation_fraction=0.5, seed=99)
        != split_for_polygon(
            f"polygon-{index}", validation_fraction=0.5, seed=cast(Any, None)
        )
    )

    example = next(
        iter_training_examples(
            [_row(polygon_id=polygon_id)],
            validation_fraction=0.5,
            seed=99,
        )
    )

    assert example.polygon_id == polygon_id
    assert example.split == split_for_polygon(
        polygon_id,
        validation_fraction=0.5,
        seed=99,
    )


@pytest.mark.parametrize("mutation", ["missing", "extra", "reordered"])
def test_iterator_rejects_a_first_row_schema_mismatch(mutation: str) -> None:
    row = _row()
    if mutation == "missing":
        del row[LANDUSE_DATASET_CONTRACT.required_columns[-1]]
    elif mutation == "extra":
        row["unexpected"] = None
    else:
        reordered_columns = (
            LANDUSE_DATASET_CONTRACT.required_columns[:2]
            + (
                LANDUSE_DATASET_CONTRACT.required_columns[3],
                LANDUSE_DATASET_CONTRACT.required_columns[2],
            )
            + LANDUSE_DATASET_CONTRACT.required_columns[4:]
        )
        row = {column: row[column] for column in reordered_columns}

    with pytest.raises(DatasetContractError):
        list(iter_training_examples([row]))


def test_iterator_rejects_an_extra_column_on_a_subsequent_row() -> None:
    valid = _row(sentence_id="valid")
    extra_column = _row(sentence_id="extra")
    extra_column["unexpected"] = None

    with pytest.raises(DatasetContractError, match="unexpected columns"):
        list(iter_training_examples([valid, extra_column]))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sentence_id", ""),
        ("sentence_id", "   "),
        ("sentence_id", None),
        ("sentence_id", 123),
        ("polygon_id", ""),
        ("polygon_id", "   "),
        ("polygon_id", None),
        ("polygon_id", 123),
    ],
)
def test_iterator_rejects_invalid_identifiers(field: str, value: object) -> None:
    row = _row()
    row[field] = value

    with pytest.raises(
        DatasetLoaderError,
        match=rf"\A{field} must be a non-empty string\Z",
    ):
        list(iter_training_examples([row]))


def test_training_label_error_preserves_the_exact_contract_message() -> None:
    contract = replace(
        LANDUSE_DATASET_CONTRACT,
        supported_label_values=("maybe",),
        training_label_values=("maybe",),
    )

    with pytest.raises(
        DatasetLoaderError,
        match=r"\Aunsupported training label: 'maybe'\Z",
    ):
        list(iter_training_examples([_row(label="maybe")], contract=contract))


def test_training_examples_are_frozen_slotted_dataclasses() -> None:
    example = next(iter(iter_training_examples([_row()])))

    assert isinstance(example, TrainingExample)
    assert is_dataclass(example)
    assert not hasattr(example, "__dict__")
    with pytest.raises(FrozenInstanceError):
        example.label = "no"  # type: ignore[misc]


def test_iter_training_examples_is_lazy() -> None:
    yielded = 0

    def rows():
        nonlocal yielded
        yielded += 1
        yield _row(sentence_id="sentence-1")
        yielded += 1
        yield _row(sentence_id="sentence-2")

    examples = iter_training_examples(rows())

    assert yielded == 0
    assert next(examples).sentence_id == "sentence-1"
    assert yielded == 1
    assert next(examples).sentence_id == "sentence-2"
    assert yielded == 2


def test_clean_iterator_is_lazy_until_consumed() -> None:
    factory_calls = 0
    rows_seen: list[tuple[int, int]] = []

    def rows_factory():
        nonlocal factory_calls
        stream_number = factory_calls
        factory_calls += 1

        def rows():
            for row_number in range(2):
                rows_seen.append((stream_number, row_number))
                yield _row(
                    sentence_id=f"sentence-{stream_number}-{row_number}",
                    content_hash=f"hash-{stream_number}-{row_number}",
                )

        return rows()

    examples = iter_clean_training_examples(rows_factory)

    assert factory_calls == 0
    assert rows_seen == []

    assert next(examples).sentence_id == "sentence-1-0"
    assert factory_calls == 2
    assert rows_seen == [(0, 0), (0, 1), (1, 0)]


def test_clean_iterator_calls_a_fresh_rows_factory_exactly_twice() -> None:
    calls = 0

    def rows_factory():
        nonlocal calls
        calls += 1
        return iter([_row(content_hash="hash-1")])

    examples = list(iter_clean_training_examples(rows_factory))

    assert calls == 2
    assert [example.sentence_id for example in examples] == ["sentence-1"]


def test_clean_iterator_excludes_every_row_from_a_conflicting_hash_group() -> None:
    rows = [
        _row(
            sentence_id="negative",
            polygon_id="polygon-negative",
            label="no",
            content_hash="conflict",
        ),
        _row(
            sentence_id="positive",
            polygon_id="polygon-positive",
            label="yes",
            content_hash="conflict",
        ),
    ]

    examples = list(iter_clean_training_examples(lambda: iter(rows)))

    assert examples == []


def test_clean_iterator_keeps_only_the_first_same_label_hash_representative() -> None:
    rows = [
        _row(
            sentence_id="first",
            polygon_id="polygon-first",
            text="First representative.",
            label="yes",
            content_hash="duplicate",
        ),
        _row(
            sentence_id="second",
            polygon_id="polygon-second",
            text="Second duplicate.",
            label="yes",
            content_hash="duplicate",
        ),
    ]

    examples = list(iter_clean_training_examples(lambda: iter(rows)))

    assert [(example.sentence_id, example.text) for example in examples] == [
        ("first", "First representative.")
    ]


def test_clean_iterator_places_a_repeated_hash_in_at_most_one_split() -> None:
    polygons_by_split: dict[DatasetSplit, str] = {}
    for index in range(100):
        polygon_id = f"polygon-{index}"
        split = split_for_polygon(
            polygon_id,
            validation_fraction=0.5,
            seed=42,
        )
        polygons_by_split.setdefault(split, polygon_id)
        if len(polygons_by_split) == 2:
            break

    assert set(polygons_by_split) == {"train", "validation"}
    rows = [
        _row(
            sentence_id=split,
            polygon_id=polygon_id,
            label="no",
            content_hash="cross-split",
        )
        for split, polygon_id in polygons_by_split.items()
    ]

    examples = list(
        iter_clean_training_examples(
            lambda: iter(rows),
            validation_fraction=0.5,
            seed=42,
        )
    )

    assert len(examples) == 1
    assert {example.split for example in examples} in (
        {"train"},
        {"validation"},
    )


def test_clean_iterator_can_assign_a_representative_to_the_held_out_test_split() -> (
    None
):
    test_polygon = next(
        f"polygon-{index}"
        for index in range(100)
        if split_for_polygon(
            f"polygon-{index}",
            validation_fraction=0.1,
            test_fraction=0.8,
            seed=42,
        )
        == "test"
    )
    example = next(
        example
        for example in iter_clean_training_examples(
            lambda: iter([_row(polygon_id=test_polygon, content_hash="test-hash")]),
            validation_fraction=0.1,
            test_fraction=0.8,
            seed=42,
        )
    )

    assert example.split == "test"


def test_clean_iterator_ignores_uncertain_rows_when_finding_conflicts() -> None:
    rows = [
        _row(
            sentence_id="uncertain",
            label="uncertain",
            content_hash="uncertain-shared",
        ),
        _row(
            sentence_id="positive",
            label="yes",
            content_hash="uncertain-shared",
        ),
    ]

    examples = list(iter_clean_training_examples(lambda: iter(rows)))

    assert [(example.sentence_id, example.label) for example in examples] == [
        ("positive", "yes")
    ]


def test_clean_iterator_preserves_each_trainable_row_without_a_usable_hash() -> None:
    rows = [
        _row(sentence_id="empty-hash", content_hash="", label="no"),
        _row(sentence_id="missing-hash", content_hash=None, label="no"),
    ]

    examples = list(iter_clean_training_examples(lambda: iter(rows)))

    assert [(example.sentence_id, example.label) for example in examples] == [
        ("empty-hash", "no"),
        ("missing-hash", "no"),
    ]


def test_clean_iterator_forwards_the_seed_for_hashless_rows() -> None:
    polygon_id = next(
        f"polygon-{index}"
        for index in range(100)
        if split_for_polygon(f"polygon-{index}", validation_fraction=0.5, seed=42)
        != split_for_polygon(
            f"polygon-{index}", validation_fraction=0.5, seed=cast(Any, None)
        )
    )

    example = next(
        iter_clean_training_examples(
            lambda: iter([_row(polygon_id=polygon_id)]),
            validation_fraction=0.5,
            seed=42,
        )
    )

    assert example.split == split_for_polygon(
        polygon_id,
        validation_fraction=0.5,
        seed=42,
    )


def test_clean_iterator_uses_the_default_seed_for_hashless_rows() -> None:
    polygon_id = next(
        f"polygon-{index}"
        for index in range(100)
        if split_for_polygon(f"polygon-{index}", validation_fraction=0.5, seed=42)
        != split_for_polygon(f"polygon-{index}", validation_fraction=0.5, seed=43)
    )

    example = next(
        iter_clean_training_examples(
            lambda: iter([_row(polygon_id=polygon_id)]),
            validation_fraction=0.5,
        )
    )

    assert example.split == split_for_polygon(
        polygon_id,
        validation_fraction=0.5,
        seed=42,
    )


def test_clean_iterator_rejects_a_malformed_row_in_the_first_pass() -> None:
    calls = 0

    def rows_factory():
        nonlocal calls
        calls += 1
        malformed = _row(sentence_id="malformed", content_hash="hash-2")
        malformed["unexpected"] = None
        return iter([_row(content_hash="hash-1"), malformed])

    with pytest.raises(DatasetContractError):
        list(iter_clean_training_examples(rows_factory))

    assert calls == 2


def test_clean_iterator_rejects_a_malformed_row_in_the_second_pass() -> None:
    calls = 0

    def rows_factory():
        nonlocal calls
        calls += 1
        if calls == 1:
            return iter([_row(content_hash="hash-1")])
        malformed = _row(sentence_id="malformed", content_hash="hash-2")
        malformed["unexpected"] = None
        return iter([_row(content_hash="hash-1"), malformed])

    with pytest.raises(DatasetContractError):
        list(iter_clean_training_examples(rows_factory))

    assert calls == 2


def test_clean_iterator_rejects_a_reused_stream_from_rows_factory() -> None:
    calls = 0
    shared = iter([_row(content_hash="hash-1")])

    def rows_factory():
        nonlocal calls
        calls += 1
        return shared

    with pytest.raises(
        DatasetLoaderError,
        match=r"\Arows_factory must return a fresh stream on each call\Z",
    ):
        list(iter_clean_training_examples(rows_factory))

    assert calls == 2


def test_load_streaming_rows_passes_pinned_streaming_arguments_without_materializing() -> (
    None
):
    calls: list[dict[str, object]] = []
    streaming_rows: Iterable[Mapping[str, object]] = iter(())

    def fake_load_dataset(**kwargs: object) -> Iterable[Mapping[str, object]]:
        calls.append(kwargs)
        return streaming_rows

    result = load_streaming_rows(load_dataset_fn=fake_load_dataset)

    assert result is streaming_rows
    assert calls == [
        {
            "path": LANDUSE_DATASET_CONTRACT.dataset_id,
            "name": LANDUSE_DATASET_CONTRACT.config,
            "split": LANDUSE_DATASET_CONTRACT.split,
            "revision": LANDUSE_DATASET_CONTRACT.provenance.repository_revision,
            "streaming": True,
            "cache_dir": str(
                ProjectConfig().data_root / Path("cache/huggingface/datasets")
            ),
        }
    ]


def test_load_streaming_rows_preserves_errors_from_the_injected_loader() -> None:
    expected = RuntimeError("remote failure")

    def failing_load_dataset(**kwargs: object) -> Iterable[Mapping[str, object]]:
        raise expected

    with pytest.raises(RuntimeError, match="remote failure") as caught:
        load_streaming_rows(load_dataset_fn=failing_load_dataset)

    assert caught.value is expected


def test_load_streaming_rows_reports_a_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def missing_datasets(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "datasets":
            raise ModuleNotFoundError("datasets", name="datasets")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", missing_datasets)

    with pytest.raises(
        DatasetLoaderError,
        match=r"\Athe optional 'datasets' dependency is required\Z",
    ):
        load_streaming_rows()


def test_load_streaming_rows_uses_the_imported_optional_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []
    streaming_rows: Iterable[Mapping[str, object]] = iter(())

    def fake_load_dataset(**kwargs: object) -> Iterable[Mapping[str, object]]:
        calls.append(kwargs)
        return streaming_rows

    original_import = builtins.__import__

    def imported_datasets(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "datasets":
            return SimpleNamespace(load_dataset=fake_load_dataset)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", imported_datasets)

    assert load_streaming_rows() is streaming_rows
    assert calls
    assert calls[0]["streaming"] is True


def test_load_streaming_rows_does_not_hide_an_unrelated_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = builtins.__import__

    def broken_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "datasets":
            raise ModuleNotFoundError("broken dependency", name="other-package")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", broken_import)

    with pytest.raises(ModuleNotFoundError, match="broken dependency"):
        load_streaming_rows()
