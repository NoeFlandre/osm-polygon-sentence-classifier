"""Streaming audit and derived-artifact boundary for the landuse dataset."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import cast

from .config import ProjectConfig
from .dataset_contract import (
    LANDUSE_DATASET_CONTRACT,
    DatasetContract,
    DatasetContractError,
)
from .dataset_loader import DatasetSplit, split_for_polygon
from .paths import ManagedPaths

CounterItems = tuple[tuple[str, int], ...]
TrainableLabelCounts = tuple[tuple[str, CounterItems], ...]
SplitManifest = tuple[tuple[str, DatasetSplit], ...]

__all__ = [
    "AuditReport",
    "AuditResult",
    "DatasetAuditError",
    "audit_rows",
    "write_audit_artifacts",
]


class DatasetAuditError(ValueError):
    """Raised when an input row or audit configuration is invalid."""


@dataclass(frozen=True, slots=True)
class AuditReport:
    """Immutable summary of one validated landuse dataset stream."""

    dataset_id: str
    config: str
    split: str
    region: str
    repository_revision: str
    parquet_sha256: str
    validation_fraction: float
    seed: int
    total_rows: int
    trainable_rows: int
    total_polygons: int
    trainable_polygons: int
    label_counts: CounterItems
    split_row_counts: CounterItems
    split_polygon_counts: CounterItems
    trainable_label_counts: TrainableLabelCounts
    language_counts: CounterItems
    source_counts: CounterItems
    text_length_min: int | None
    text_length_max: int | None
    text_length_mean: float | None
    duplicate_hash_groups: int
    duplicate_rows_beyond_first: int
    cross_polygon_duplicate_groups: int
    conflicting_polygons: int
    review_required_reasons: tuple[str, ...]
    ready: bool

    @property
    def dataset_split(self) -> str:
        """Return the source dataset split using the descriptive alias."""

        return self.split

    @property
    def readiness(self) -> bool:
        """Return whether the audited data passed all review gates."""

        return self.ready

    @property
    def parquet_sha(self) -> str:
        """Return the pinned Parquet digest using a short alias."""

        return self.parquet_sha256

    def to_dict(self) -> dict[str, object]:
        """Return JSON-safe report fields without input rows or sentence text."""

        return {
            "config": self.config,
            "conflicting_polygons": self.conflicting_polygons,
            "cross_polygon_duplicate_groups": self.cross_polygon_duplicate_groups,
            "dataset_id": self.dataset_id,
            "duplicate_hash_groups": self.duplicate_hash_groups,
            "duplicate_rows_beyond_first": self.duplicate_rows_beyond_first,
            "label_counts": dict(self.label_counts),
            "language_counts": dict(self.language_counts),
            "parquet_sha256": self.parquet_sha256,
            "ready": self.ready,
            "region": self.region,
            "repository_revision": self.repository_revision,
            "review_required_reasons": list(self.review_required_reasons),
            "seed": self.seed,
            "source_counts": dict(self.source_counts),
            "split": self.split,
            "split_polygon_counts": dict(self.split_polygon_counts),
            "split_row_counts": dict(self.split_row_counts),
            "text_length_max": self.text_length_max,
            "text_length_mean": self.text_length_mean,
            "text_length_min": self.text_length_min,
            "total_polygons": self.total_polygons,
            "total_rows": self.total_rows,
            "trainable_label_counts": {
                split: dict(counts) for split, counts in self.trainable_label_counts
            },
            "trainable_polygons": self.trainable_polygons,
            "trainable_rows": self.trainable_rows,
            "validation_fraction": self.validation_fraction,
        }


@dataclass(frozen=True, slots=True)
class AuditResult:
    """Immutable audit report plus its deterministic polygon split manifest."""

    report: AuditReport
    split_manifest: SplitManifest

    @property
    def ready(self) -> bool:
        """Return whether the report requires no manual review."""

        return self.report.ready

    @property
    def readiness(self) -> bool:
        """Return the report readiness using a descriptive alias."""

        return self.report.ready


def _validate_validation_fraction(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value <= 1
    ):
        raise DatasetAuditError(
            "validation_fraction must be a finite number between 0 and 1"
        )
    return float(value)


def _require_identifier(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _counter_key(value: object) -> str:
    return value if isinstance(value, str) else str(value)


def _sorted_counter(counter: Counter[str]) -> CounterItems:
    return tuple(sorted(counter.items()))


def _wrap_contract_error(
    row_number: int, error: DatasetContractError
) -> DatasetAuditError:
    return DatasetAuditError(f"row {row_number}: {error}")


def audit_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    validation_fraction: float = 0.2,
    seed: int = 42,
    contract: DatasetContract = LANDUSE_DATASET_CONTRACT,
) -> AuditResult:
    """Validate and summarize rows in one lazy pass without materializing them."""

    normalized_fraction = _validate_validation_fraction(validation_fraction)
    iterator = iter(rows)
    try:
        first_row = next(iterator)
    except StopIteration:
        first_row = None

    if first_row is not None:
        try:
            contract.validate_columns(first_row.keys())
        except DatasetContractError as error:
            raise _wrap_contract_error(1, error) from error

    label_counts = Counter(dict.fromkeys(contract.supported_label_values, 0))
    split_row_counts: Counter[str] = Counter(dict.fromkeys(("train", "validation"), 0))
    split_polygon_counts: Counter[str] = Counter(
        dict.fromkeys(("train", "validation"), 0)
    )
    trainable_label_counts: dict[str, Counter[str]] = {
        split: Counter(dict.fromkeys(("no", "yes"), 0))
        for split in ("train", "validation")
    }
    language_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    polygon_splits: dict[str, DatasetSplit] = {}
    trainable_polygons: set[str] = set()
    polygon_labels: dict[str, set[str]] = {}
    hash_counts: Counter[str] = Counter()
    hash_polygons: dict[str, set[str]] = {}
    text_lengths: list[int] = []
    total_rows = 0
    trainable_rows = 0

    stream: Iterable[Mapping[str, object]]
    stream = () if first_row is None else chain((first_row,), iterator)

    for row_number, row in enumerate(stream, start=1):
        try:
            contract.validate_row(row)
        except DatasetContractError as error:
            raise _wrap_contract_error(row_number, error) from error

        try:
            _require_identifier("sentence_id", row["sentence_id"])
            polygon_id = _require_identifier("polygon_id", row["polygon_id"])
        except ValueError as error:
            raise DatasetAuditError(f"row {row_number}: {error}") from error

        if polygon_id not in polygon_splits:
            try:
                polygon_splits[polygon_id] = split_for_polygon(
                    polygon_id,
                    validation_fraction=normalized_fraction,
                    seed=seed,
                )
            except ValueError as error:
                raise DatasetAuditError(f"row {row_number}: {error}") from error

        row_split = polygon_splits[polygon_id]
        label = _counter_key(row[contract.label_column])
        label_counts[label] += 1
        split_row_counts[row_split] += 1
        total_rows += 1

        if label not in ("no", "yes"):
            continue

        trainable_rows += 1
        trainable_polygons.add(polygon_id)
        trainable_label_counts[row_split][label] += 1
        language_counts[_counter_key(row["language"])] += 1
        source_counts[_counter_key(row["source"])] += 1
        text = cast(str, row["sentence_text_normalized"])
        text_lengths.append(len(text))
        polygon_labels.setdefault(polygon_id, set()).add(label)

        sentence_content_hash = row.get("sentence_content_hash")
        if isinstance(sentence_content_hash, str) and sentence_content_hash:
            hash_counts[sentence_content_hash] += 1
            hash_polygons.setdefault(sentence_content_hash, set()).add(polygon_id)

    for row_split in polygon_splits.values():
        split_polygon_counts[row_split] += 1

    duplicate_hash_groups = sum(count > 1 for count in hash_counts.values())
    duplicate_rows_beyond_first = sum(
        count - 1 for count in hash_counts.values() if count > 1
    )
    cross_polygon_duplicate_groups = sum(
        count > 1 and len(hash_polygons[content_hash]) > 1
        for content_hash, count in hash_counts.items()
    )
    conflicting_polygons = sum(
        labels.issuperset({"no", "yes"}) for labels in polygon_labels.values()
    )

    review_reasons: set[str] = set()
    if conflicting_polygons:
        review_reasons.add("polygon_label_conflicts")
    if cross_polygon_duplicate_groups:
        review_reasons.add("cross_polygon_duplicate_groups")
    for row_split in ("train", "validation"):
        if any(
            trainable_label_counts[row_split][label] == 0 for label in ("no", "yes")
        ):
            review_reasons.add(f"{row_split}_split_missing_label")

    reasons = tuple(sorted(review_reasons))
    report = AuditReport(
        dataset_id=contract.dataset_id,
        config=contract.config,
        split=contract.split,
        region=contract.region,
        repository_revision=contract.provenance.repository_revision,
        parquet_sha256=contract.provenance.parquet_sha256,
        validation_fraction=normalized_fraction,
        seed=seed,
        total_rows=total_rows,
        trainable_rows=trainable_rows,
        total_polygons=len(polygon_splits),
        trainable_polygons=len(trainable_polygons),
        label_counts=_sorted_counter(label_counts),
        split_row_counts=_sorted_counter(split_row_counts),
        split_polygon_counts=_sorted_counter(split_polygon_counts),
        trainable_label_counts=tuple(
            (
                row_split,
                _sorted_counter(trainable_label_counts[row_split]),
            )
            for row_split in sorted(trainable_label_counts)
        ),
        language_counts=_sorted_counter(language_counts),
        source_counts=_sorted_counter(source_counts),
        text_length_min=min(text_lengths) if text_lengths else None,
        text_length_max=max(text_lengths) if text_lengths else None,
        text_length_mean=(sum(text_lengths) / len(text_lengths))
        if text_lengths
        else None,
        duplicate_hash_groups=duplicate_hash_groups,
        duplicate_rows_beyond_first=duplicate_rows_beyond_first,
        cross_polygon_duplicate_groups=cross_polygon_duplicate_groups,
        conflicting_polygons=conflicting_polygons,
        review_required_reasons=reasons,
        ready=not reasons,
    )
    return AuditResult(
        report=report, split_manifest=tuple(sorted(polygon_splits.items()))
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_audit_artifacts(
    result: AuditResult,
    *,
    config: ProjectConfig = ProjectConfig(),  # noqa: B008
) -> tuple[Path, Path]:
    """Write the explicit derived report and split manifest under audit/landuse."""

    audit_directory = ManagedPaths(config).child("audit/landuse")
    audit_directory.mkdir(parents=True, exist_ok=True)
    report_path = audit_directory / "audit_report.json"
    manifest_path = audit_directory / "split_manifest.json"
    _write_json(report_path, result.report.to_dict())
    _write_json(manifest_path, dict(result.split_manifest))
    return report_path, manifest_path
