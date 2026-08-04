"""Streaming transformation and loading for the landuse training dataset."""

import hashlib
import math
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from itertools import chain
from typing import Literal, cast

from .config import ProjectConfig
from .dataset_contract import LANDUSE_DATASET_CONTRACT, DatasetContract
from .paths import ManagedPaths

DatasetSplit = Literal["train", "validation"]
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


def _require_identifier(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DatasetLoaderError(f"{name} must be a non-empty string")
    return value


def split_for_polygon(
    polygon_id: str,
    *,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> DatasetSplit:
    """Assign a polygon to a deterministic training or validation split."""

    _validate_validation_fraction(validation_fraction)
    polygon_id = _require_identifier("polygon_id", polygon_id)
    digest = hashlib.sha256(f"{seed}:{polygon_id}".encode()).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False) / 2**64
    return "validation" if value < validation_fraction else "train"


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
    seed: int = 42,
    contract: DatasetContract = LANDUSE_DATASET_CONTRACT,
) -> Iterator[TrainingExample]:
    """Lazily validate, filter, and transform streaming dataset rows."""

    _validate_validation_fraction(validation_fraction)
    iterator = iter(rows)
    try:
        first_row = next(iterator)
    except StopIteration:
        return

    contract.validate_columns(first_row.keys())
    for row in chain((first_row,), iterator):
        contract.validate_row(row)
        label = row[contract.label_column]
        if label not in contract.training_label_values:
            continue
        training_label = _training_label_for_row(row, contract=contract)
        if training_label is None:
            continue
        yield _training_example_from_row(
            row,
            label=training_label,
            validation_fraction=validation_fraction,
            seed=seed,
        )


def iter_clean_training_examples(
    rows_factory: Callable[[], Iterable[Mapping[str, object]]],
    *,
    validation_fraction: float = 0.2,
    seed: int = 42,
    contract: DatasetContract = LANDUSE_DATASET_CONTRACT,
) -> Iterator[TrainingExample]:
    """Yield deduplicated training examples from two fresh lazy streams.

    ``rows_factory`` is called exactly twice and must return a fresh,
    independently iterable stream on each call. The first pass validates every
    row and records only the trainable labels seen for each usable sentence
    content hash. Hashes with both training labels are excluded completely.
    The second pass validates every row again, skips ``uncertain`` rows and
    contradictory hashes, and emits only the first trainable occurrence of
    each remaining usable hash. Rows without a usable hash retain the ordinary
    iterator behavior.
    """

    _validate_validation_fraction(validation_fraction)
    first_stream = rows_factory()
    second_stream = rows_factory()
    first_iterator = iter(first_stream)
    second_iterator = iter(second_stream)
    if first_iterator is second_iterator:
        raise DatasetLoaderError("rows_factory must return a fresh stream on each call")

    labels_by_hash: dict[str, set[TrainingLabel]] = {}
    for row in _validated_rows(first_iterator, contract=contract):
        label = _training_label_for_row(row, contract=contract)
        if label is None:
            continue
        content_hash = _usable_sentence_content_hash(row)
        if content_hash is not None:
            labels_by_hash.setdefault(content_hash, set()).add(label)

    contradictory_hashes = {
        content_hash
        for content_hash, labels in labels_by_hash.items()
        if len(labels) > 1
    }
    emitted_hashes: set[str] = set()
    for row in _validated_rows(second_iterator, contract=contract):
        label = _training_label_for_row(row, contract=contract)
        if label is None:
            continue
        content_hash = _usable_sentence_content_hash(row)
        if content_hash is None:
            yield _training_example_from_row(
                row,
                label=label,
                validation_fraction=validation_fraction,
                seed=seed,
            )
            continue
        if content_hash in contradictory_hashes or content_hash in emitted_hashes:
            continue
        emitted_hashes.add(content_hash)
        yield _training_example_from_row(
            row,
            label=label,
            validation_fraction=validation_fraction,
            seed=seed,
        )


def load_streaming_rows(
    *,
    config: ProjectConfig = ProjectConfig(),  # noqa: B008
    contract: DatasetContract = LANDUSE_DATASET_CONTRACT,
    load_dataset_fn: Callable[..., object] | None = None,
) -> object:
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
        load_dataset_fn = load_dataset

    return load_dataset_fn(
        path=contract.dataset_id,
        name=contract.config,
        split=contract.split,
        revision=contract.provenance.repository_revision,
        streaming=True,
        cache_dir=str(ManagedPaths(config).child("cache/huggingface/datasets")),
    )
