"""Streaming audit and derived-artifact boundary for the landuse dataset."""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from collections.abc import Iterable, Mapping
from contextlib import suppress
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
from .paths import ManagedPathError, ManagedPaths

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
    text_length_count = 0
    text_length_total = 0
    text_length_min: int | None = None
    text_length_max: int | None = None
    total_rows = 0
    trainable_rows = 0

    stream: Iterable[Mapping[str, object]]
    stream = () if first_row is None else chain((first_row,), iterator)

    for row_number, row in enumerate(stream, start=1):
        try:
            contract.validate_columns(row.keys())
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
        text_length = len(text)
        text_length_count += 1
        text_length_total += text_length
        text_length_min = (
            text_length
            if text_length_min is None
            else min(text_length_min, text_length)
        )
        text_length_max = (
            text_length
            if text_length_max is None
            else max(text_length_max, text_length)
        )
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
        text_length_min=text_length_min,
        text_length_max=text_length_max,
        text_length_mean=(text_length_total / text_length_count)
        if text_length_count
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


def _directory_fd_supported() -> bool:
    return all(hasattr(os, flag) for flag in ("O_DIRECTORY", "O_NOFOLLOW")) and all(
        function in getattr(os, "supports_dir_fd", ())
        for function in (os.open, os.mkdir)
    )


def _open_directory_no_follow(path: Path, *, create: bool) -> int:
    """Open every directory component without following symlinks."""

    if not _directory_fd_supported():
        raise OSError("directory no-follow support is unavailable")

    absolute_path = Path(os.path.abspath(path))
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_fd = os.open(absolute_path.anchor, directory_flags)
    try:
        for component in absolute_path.parts[1:]:
            try:
                child_fd = os.open(
                    component,
                    directory_flags,
                    dir_fd=directory_fd,
                )
            except FileNotFoundError:
                if not create:
                    raise
                with suppress(FileExistsError):
                    os.mkdir(component, 0o777, dir_fd=directory_fd)
                child_fd = os.open(
                    component,
                    directory_flags,
                    dir_fd=directory_fd,
                )
            os.close(directory_fd)
            directory_fd = child_fd
        return directory_fd
    except OSError:
        os.close(directory_fd)
        raise


def _prepare_audit_directory(audit_directory: Path) -> None:
    """Create the fixed audit path without following directory symlinks."""

    try:
        if _directory_fd_supported():
            directory_fd = _open_directory_no_follow(audit_directory, create=True)
            os.close(directory_fd)
        else:
            _prepare_audit_directory_without_directory_fds(audit_directory)
    except OSError as error:
        raise DatasetAuditError(
            f"unable to prepare audit artifact directory: {audit_directory}"
        ) from error


def _prepare_audit_directory_without_directory_fds(audit_directory: Path) -> None:
    """Fallback directory preparation for platforms without directory FDs."""

    absolute_path = Path(os.path.abspath(audit_directory))
    current = Path(absolute_path.anchor)
    for component in absolute_path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise DatasetAuditError(
                f"audit artifact directory must not contain a symlink: {current}"
            )
        if not current.exists():
            with suppress(FileExistsError):
                current.mkdir()
        if not current.is_dir():
            raise DatasetAuditError(
                f"audit artifact path is not a directory: {current}"
            )


def _write_json(path: Path, payload: object) -> None:
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | no_follow
    directory_fd: int | None = None
    file_fd: int | None = None
    try:
        if _directory_fd_supported():
            directory_fd = _open_directory_no_follow(path.parent, create=False)
            file_fd = os.open(
                path.name,
                flags,
                0o666,
                dir_fd=directory_fd,
            )
            output = os.fdopen(file_fd, "w", encoding="utf-8")
            file_fd = None
            with output:
                output.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        else:

            def opener(file_path: str, open_flags: int) -> int:
                return os.open(file_path, open_flags | no_follow, 0o666)

            with open(path, "w", encoding="utf-8", opener=opener) as output:
                output.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    except OSError as error:
        raise DatasetAuditError(f"unable to write audit artifact: {path}") from error
    finally:
        if file_fd is not None:
            os.close(file_fd)
        if directory_fd is not None:
            os.close(directory_fd)


def _safe_artifact_path(audit_directory: Path, filename: str) -> Path:
    """Reject final symlinks and resolved paths outside the audit directory."""

    artifact_path = audit_directory / filename
    if artifact_path.is_symlink():
        raise DatasetAuditError(f"artifact path must not be a symlink: {artifact_path}")

    resolved_directory = audit_directory.resolve()
    resolved_artifact = artifact_path.resolve()
    if not resolved_artifact.is_relative_to(resolved_directory):
        raise DatasetAuditError(
            f"artifact path must remain beneath the audit directory: {artifact_path}"
        )
    return artifact_path


def write_audit_artifacts(
    result: AuditResult,
    *,
    config: ProjectConfig = ProjectConfig(),  # noqa: B008
) -> tuple[Path, Path]:
    """Write the explicit derived report and split manifest under audit/landuse."""

    try:
        audit_directory = ManagedPaths(config).child("audit/landuse")
    except ManagedPathError as error:
        raise DatasetAuditError(
            "unable to prepare audit artifact directory under the managed root"
        ) from error
    _prepare_audit_directory(audit_directory)
    report_path = _safe_artifact_path(audit_directory, "audit_report.json")
    manifest_path = _safe_artifact_path(audit_directory, "split_manifest.json")
    _write_json(report_path, result.report.to_dict())
    _write_json(manifest_path, dict(result.split_manifest))
    return report_path, manifest_path
