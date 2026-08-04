from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path
from typing import cast, get_args

import pytest

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
) -> dict[str, object]:
    row = dict.fromkeys(LANDUSE_DATASET_CONTRACT.required_columns)
    row.update(
        {
            "sentence_id": sentence_id,
            "polygon_id": polygon_id,
            "region": "afghanistan",
            "sentence_text_normalized": text,
            "landuse_relevance": label,
        }
    )
    assert len(row) == 39
    return row


def test_training_types_are_limited_to_the_contract_values() -> None:
    assert get_args(DatasetSplit) == ("train", "validation")
    assert get_args(TrainingLabel) == ("no", "yes")


def test_split_for_polygon_is_deterministic_and_returns_a_supported_split() -> None:
    result = split_for_polygon("polygon-1", validation_fraction=0.2, seed=42)

    assert result in ("train", "validation")
    assert result == split_for_polygon("polygon-1", validation_fraction=0.2, seed=42)


@pytest.mark.parametrize("fraction", [-0.01, 1.01, float("nan"), float("inf")])
def test_split_for_polygon_rejects_invalid_validation_fractions(
    fraction: float,
) -> None:
    with pytest.raises(DatasetLoaderError, match="validation_fraction"):
        split_for_polygon("polygon-1", validation_fraction=fraction)


@pytest.mark.parametrize("polygon_id", ["", "   ", None, 123])
def test_split_for_polygon_rejects_blank_or_non_string_ids(
    polygon_id: object,
) -> None:
    with pytest.raises(DatasetLoaderError, match="polygon_id"):
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

    with pytest.raises(DatasetLoaderError, match=field):
        list(iter_training_examples([row]))


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


def test_load_streaming_rows_passes_pinned_streaming_arguments_without_materializing() -> (
    None
):
    calls: list[dict[str, object]] = []
    streaming_rows = object()

    def fake_load_dataset(**kwargs: object) -> object:
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

    def failing_load_dataset(**kwargs: object) -> object:
        raise expected

    with pytest.raises(RuntimeError, match="remote failure") as caught:
        load_streaming_rows(load_dataset_fn=failing_load_dataset)

    assert caught.value is expected
