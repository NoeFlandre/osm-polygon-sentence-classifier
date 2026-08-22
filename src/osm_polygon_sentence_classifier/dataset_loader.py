"""Streaming transformation and loading for the landuse training dataset."""

import hashlib
import math
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Literal, cast

from .config import ProjectConfig
from .dataset_contract import LANDUSE_DATASET_CONTRACT, DatasetContract
from .paths import ManagedPaths

DatasetSplit = Literal["train", "validation", "test"]
TrainingLabel = Literal["no", "yes"]

__all__ = [
    "DatasetLoaderError",
    "DatasetSplit",
    "TrainingExample",
    "TrainingLabel",
    "iter_clean_training_examples",
    "iter_training_examples",
    "load_streaming_rows",
    "split_for_polygon",
]


class DatasetLoaderError(ValueError):
    """Raised when a row or loader configuration violates the training boundary."""


@dataclass(frozen=True, slots=True)
class TrainingExample:
    """One immutable, validated example suitable for training."""

    sentence_id: str
    polygon_id: str
    text: str
    label: TrainingLabel
    split: DatasetSplit


def _validate_validation_fraction(validation_fraction: float) -> None:
    if (
        isinstance(validation_fraction, bool)
        or not isinstance(validation_fraction, (int, float))
        or not math.isfinite(validation_fraction)
        or not 0 <= validation_fraction <= 1
    ):
        raise DatasetLoaderError(
            "validation_fraction must be a finite number between 0 and 1"
        )


def _validate_split_fractions(
    validation_fraction: float,
    test_fraction: float,
) -> None:
    if test_fraction == 0:
        _validate_validation_fraction(validation_fraction)
        return
    if not _is_valid_fraction(validation_fraction):
        _raise_invalid_split_fractions()
    if not _is_valid_fraction(test_fraction):
        _raise_invalid_split_fractions()
    if validation_fraction + test_fraction > 1:
        _raise_invalid_split_fractions()


def _is_valid_fraction(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and 0 <= value <= 1
    )


def _raise_invalid_split_fractions() -> None:
    raise DatasetLoaderError(
        "validation and test fractions must be finite, non-negative, "
        "and sum to at most 1"
    )


def _require_identifier(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetLoaderError(f"{name} must be a non-empty string")
    return value


def split_for_polygon(
    polygon_id: str,
    *,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.0,
    seed: int = 42,
) -> DatasetSplit:
    """Assign a polygon to deterministic training, validation, or test."""

    _validate_split_fractions(validation_fraction, test_fraction)
    polygon_id = _require_identifier("polygon_id", polygon_id)
    digest = hashlib.sha256(f"{seed}:{polygon_id}".encode()).digest()
    digest_value = int.from_bytes(digest[:8], "big")  # pragma: no mutate
    value = digest_value / 2**64
    if value < validation_fraction:
        return "validation"
    if value < validation_fraction + test_fraction:
        return "test"
    return "train"


def _training_label_for_row(
    row: Mapping[str, object],
    *,
    contract: DatasetContract,
) -> TrainingLabel | None:
    label = row[contract.label_column]
    if label not in contract.training_label_values:
        return None
    if label not in ("no", "yes"):
        raise DatasetLoaderError(f"unsupported training label: {label!r}")
    return cast(TrainingLabel, label)


def _training_example_from_row(
    row: Mapping[str, object],
    *,
    label: TrainingLabel,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> TrainingExample:
    sentence_id = _require_identifier("sentence_id", row["sentence_id"])
    polygon_id = _require_identifier("polygon_id", row["polygon_id"])
    return TrainingExample(
        sentence_id=sentence_id,
        polygon_id=polygon_id,
        text=cast(str, row["sentence_text_normalized"]),
        label=label,
        split=split_for_polygon(
            polygon_id,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            seed=seed,
        ),
    )


def _validated_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    contract: DatasetContract,
) -> Iterator[Mapping[str, object]]:
    for row in rows:
        contract.validate_columns(row.keys())
        contract.validate_row(row)
        yield row


def _usable_sentence_content_hash(row: Mapping[str, object]) -> str | None:
    content_hash = row.get("sentence_content_hash")
    if not isinstance(content_hash, str) or not content_hash.strip():
        return None
    return content_hash


def iter_training_examples(
    rows: Iterable[Mapping[str, object]],
    *,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.0,
    seed: int = 42,
    contract: DatasetContract = LANDUSE_DATASET_CONTRACT,
) -> Iterator[TrainingExample]:
    """Lazily validate every row, filter, and transform streaming dataset rows."""

    _validate_split_fractions(validation_fraction, test_fraction)
    for row in _validated_rows(rows, contract=contract):
        label = row[contract.label_column]
        if label not in contract.training_label_values:
            continue
        training_label = cast(
            TrainingLabel, _training_label_for_row(row, contract=contract)
        )
        yield _training_example_from_row(
            row,
            label=training_label,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )


def _discover_contradictory_hashes(
    iterator: Iterator[Mapping[str, object]],
    *,
    contract: DatasetContract,
) -> set[str]:
    """First pass: validate rows and find hashes with conflicting labels.

    Records only the trainable labels seen for each usable sentence content
    hash and returns the hashes that carried both training labels across the
    stream. Every row is validated even when it does not contribute a hash.
    """

    labels_by_hash: dict[str, set[TrainingLabel]] = {}
    for row in _validated_rows(iterator, contract=contract):
        _record_training_label(labels_by_hash, row, contract=contract)

    return {
        content_hash
        for content_hash, labels in labels_by_hash.items()
        if len(labels) > 1
    }


def _record_training_label(
    labels_by_hash: dict[str, set[TrainingLabel]],
    row: Mapping[str, object],
    *,
    contract: DatasetContract,
) -> None:
    label = _training_label_for_row(row, contract=contract)
    if label is None:
        return
    content_hash = _usable_sentence_content_hash(row)
    if content_hash is not None:
        labels_by_hash.setdefault(content_hash, set()).add(label)


def _emit_clean_examples(
    iterator: Iterator[Mapping[str, object]],
    *,
    contradictory_hashes: set[str],
    validation_fraction: float,
    test_fraction: float,
    seed: int,
    contract: DatasetContract,
) -> Iterator[TrainingExample]:
    """Second pass: validate, filter, select representatives, and emit.

    Revalidates every row, skips ``uncertain`` rows and contradictory hashes,
    and emits only the first trainable occurrence of each remaining usable
    hash. Rows without a usable hash retain the ordinary iterator behavior.
    """

    emitted_hashes: set[str] = set()
    for row in _validated_rows(iterator, contract=contract):
        example = _clean_example_for_row(
            row,
            contradictory_hashes=contradictory_hashes,
            emitted_hashes=emitted_hashes,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            seed=seed,
            contract=contract,
        )
        if example is not None:
            yield example


def _clean_example_for_row(
    row: Mapping[str, object],
    *,
    contradictory_hashes: set[str],
    emitted_hashes: set[str],
    validation_fraction: float,
    test_fraction: float,
    seed: int,
    contract: DatasetContract,
) -> TrainingExample | None:
    label = _training_label_for_row(row, contract=contract)
    if label is None:
        return None
    content_hash = _usable_sentence_content_hash(row)
    if content_hash is None:
        return _training_example_from_row(
            row,
            label=label,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            seed=seed,
        )
    if content_hash in contradictory_hashes or content_hash in emitted_hashes:
        return None
    emitted_hashes.add(content_hash)
    return _training_example_from_row(
        row,
        label=label,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
    )


def iter_clean_training_examples(
    rows_factory: Callable[[], Iterable[Mapping[str, object]]],
    *,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.0,
    seed: int = 42,
    contract: DatasetContract = LANDUSE_DATASET_CONTRACT,
) -> Iterator[TrainingExample]:
    """Yield deduplicated training examples from two fresh lazy streams.

    ``rows_factory`` is called exactly twice and must return a fresh,
    independently iterable stream on each call. The public iterator remains
    lazy until consumed. Once iteration starts, the first fresh stream is
    fully consumed to discover contradictory sentence-content-hash groups,
    then the second fresh stream is consumed to emit clean representatives.
    Rows are processed incrementally as they arrive from each stream rather
    than materialized into an intermediate list, and no cleaned dataset is
    written. Rows without a usable hash retain the ordinary iterator
    behavior.
    """

    _validate_split_fractions(validation_fraction, test_fraction)
    first_stream = rows_factory()
    second_stream = rows_factory()
    first_iterator = iter(first_stream)
    second_iterator = iter(second_stream)
    if first_iterator is second_iterator:
        raise DatasetLoaderError("rows_factory must return a fresh stream on each call")

    contradictory_hashes = _discover_contradictory_hashes(
        first_iterator,
        contract=contract,
    )
    yield from _emit_clean_examples(
        second_iterator,
        contradictory_hashes=contradictory_hashes,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
        contract=contract,
    )


def load_streaming_rows(
    *,
    config: ProjectConfig = ProjectConfig(),  # noqa: B008
    contract: DatasetContract = LANDUSE_DATASET_CONTRACT,
    load_dataset_fn: Callable[..., Iterable[Mapping[str, object]]] | None = None,
) -> Iterable[Mapping[str, object]]:
    """Return the pinned dataset's lazy Hugging Face streaming object."""

    if load_dataset_fn is None:
        try:
            from datasets import load_dataset
        except ModuleNotFoundError as error:
            if error.name != "datasets":
                raise
            raise DatasetLoaderError(
                "the optional 'datasets' dependency is required"
            ) from error
        load_dataset_fn = cast(
            Callable[..., Iterable[Mapping[str, object]]], load_dataset
        )

    return load_dataset_fn(
        path=contract.dataset_id,
        name=contract.config,
        split=contract.split,
        revision=contract.provenance.repository_revision,
        streaming=True,
        cache_dir=str(ManagedPaths(config).child("cache/huggingface/datasets")),
    )
