import io
import json
import math
import os
from collections.abc import Callable, Iterator
from dataclasses import FrozenInstanceError, is_dataclass, replace
from itertools import count
from pathlib import Path
from typing import Any, cast

import pytest

import osm_polygon_sentence_classifier.dataset_audit as dataset_audit
from osm_polygon_sentence_classifier.config import ProjectConfig
from osm_polygon_sentence_classifier.dataset_audit import (
    AuditReport,
    AuditResult,
    DatasetAuditError,
    _write_json,
    audit_rows,
    write_audit_artifacts,
)
from osm_polygon_sentence_classifier.dataset_contract import (
    LANDUSE_DATASET_CONTRACT,
)
from osm_polygon_sentence_classifier.dataset_loader import split_for_polygon
from osm_polygon_sentence_classifier.paths import ManagedPathError


def _row(
    *,
    sentence_id: object,
    polygon_id: object,
    label: str,
    text: str,
    language: str = "en",
    source: str = "wikipedia",
    sentence_content_hash: object = None,
) -> dict[str, object]:
    row = dict.fromkeys(LANDUSE_DATASET_CONTRACT.required_columns)
    row.update(
        {
            "sentence_id": sentence_id,
            "polygon_id": polygon_id,
            "region": "afghanistan",
            "sentence_text_normalized": text,
            "language": language,
            "source": source,
            "sentence_content_hash": sentence_content_hash,
            "landuse_relevance": label,
        }
    )
    return row


def _polygon_for_split(split: str, *, seed: int = 7) -> str:
    for suffix in count():
        polygon_id = f"polygon-{split}-{suffix}"
        if (
            split_for_polygon(
                polygon_id,
                validation_fraction=0.5,
                seed=seed,
            )
            == split
        ):
            return polygon_id
    raise AssertionError("unreachable")


@pytest.mark.parametrize("value", [0, 0.5, 1])
def test_validation_fraction_accepts_the_inclusive_finite_range(value: float) -> None:
    assert dataset_audit._validate_validation_fraction(value) == float(value)


@pytest.mark.parametrize(
    "value",
    [True, "0.5", math.nan, math.inf, -0.1, 1.5],
)
def test_validation_fraction_rejects_invalid_values_with_the_exact_error(
    value: object,
) -> None:
    with pytest.raises(DatasetAuditError) as error:
        dataset_audit._validate_validation_fraction(value)

    assert str(error.value) == (
        "validation_fraction must be a finite number between 0 and 1"
    )


@pytest.mark.parametrize("value", [0, 1.5, None])
def test_counter_key_stringifies_non_string_values(value: object) -> None:
    assert dataset_audit._counter_key(value) == str(value)


def _balanced_rows() -> list[dict[str, object]]:
    train_polygon = _polygon_for_split("train")
    validation_polygon = _polygon_for_split("validation")
    return [
        _row(
            sentence_id="sentence-1",
            polygon_id=train_polygon,
            label="no",
            text="No",
            language="en",
            source="wikipedia",
        ),
        _row(
            sentence_id="sentence-2",
            polygon_id=train_polygon,
            label="yes",
            text="Yes!",
            language="en",
            source="wikivoyage",
        ),
        _row(
            sentence_id="sentence-3",
            polygon_id=train_polygon,
            label="uncertain",
            text="Maybe",
            language="fr",
            source="wikipedia",
        ),
        _row(
            sentence_id="sentence-4",
            polygon_id=validation_polygon,
            label="no",
            text="Nope",
            language="fr",
            source="wikipedia",
        ),
        _row(
            sentence_id="sentence-5",
            polygon_id=validation_polygon,
            label="yes",
            text="Oui",
            language="en",
            source="wikipedia",
        ),
        _row(
            sentence_id="sentence-6",
            polygon_id=validation_polygon,
            label="uncertain",
            text="Maybe",
            language="fr",
            source="wikivoyage",
        ),
    ]


def test_directory_fd_support_requires_both_flags_and_functions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(os, "O_DIRECTORY", 0o40000, raising=False)
    monkeypatch.setattr(os, "O_NOFOLLOW", 0o100000, raising=False)
    monkeypatch.setattr(os, "supports_dir_fd", {os.open, os.mkdir}, raising=False)

    assert dataset_audit._directory_fd_supported() is True


@pytest.mark.parametrize("missing_flag", ["O_DIRECTORY", "O_NOFOLLOW"])
def test_directory_fd_support_requires_each_os_flag(
    monkeypatch: pytest.MonkeyPatch,
    missing_flag: str,
) -> None:
    monkeypatch.setattr(os, "O_DIRECTORY", 0o40000, raising=False)
    monkeypatch.setattr(os, "O_NOFOLLOW", 0o100000, raising=False)
    monkeypatch.setattr(os, "supports_dir_fd", {os.open, os.mkdir}, raising=False)
    monkeypatch.delattr(os, missing_flag, raising=False)

    assert dataset_audit._directory_fd_supported() is False


@pytest.mark.parametrize("supported_functions", [set(), {os.open}])
def test_directory_fd_support_requires_open_and_mkdir(
    monkeypatch: pytest.MonkeyPatch,
    supported_functions: set[object],
) -> None:
    monkeypatch.setattr(os, "O_DIRECTORY", 0o40000, raising=False)
    monkeypatch.setattr(os, "O_NOFOLLOW", 0o100000, raising=False)
    monkeypatch.setattr(os, "supports_dir_fd", supported_functions, raising=False)

    assert dataset_audit._directory_fd_supported() is False


def test_directory_fd_support_treats_missing_support_metadata_as_unsupported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(os, "O_DIRECTORY", 0o40000, raising=False)
    monkeypatch.setattr(os, "O_NOFOLLOW", 0o100000, raising=False)
    monkeypatch.delattr(os, "supports_dir_fd", raising=False)

    assert dataset_audit._directory_fd_supported() is False


def test_open_child_directory_handles_a_create_race_and_preserves_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    open_calls: list[tuple[object, ...]] = []
    mkdir_calls: list[tuple[object, ...]] = []

    def fake_open(*args: object, **_kwargs: object) -> int:
        open_calls.append(args)
        if len(open_calls) == 1:
            raise FileNotFoundError("child disappeared")
        return 23

    def fake_mkdir(*args: object, **kwargs: object) -> None:
        mkdir_calls.append((*args, kwargs))
        raise FileExistsError("another writer created it")

    monkeypatch.setattr(os, "open", fake_open)
    monkeypatch.setattr(os, "mkdir", fake_mkdir)

    assert (
        dataset_audit._open_child_directory(
            "child",
            parent_fd=17,
            flags=os.O_RDONLY,
            create=True,
        )
        == 23
    )
    assert mkdir_calls == [("child", 0o777, {"dir_fd": 17})]
    assert open_calls == [
        ("child", os.O_RDONLY),
        ("child", os.O_RDONLY),
    ]


@pytest.mark.parametrize("missing", ["O_DIRECTORY", "O_NOFOLLOW", "supports_dir_fd"])
def test_open_directory_no_follow_reports_missing_platform_support(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    missing: str,
) -> None:
    monkeypatch.setattr(os, "O_DIRECTORY", 0o40000, raising=False)
    monkeypatch.setattr(os, "O_NOFOLLOW", 0o100000, raising=False)
    monkeypatch.setattr(os, "supports_dir_fd", {os.open, os.mkdir}, raising=False)
    monkeypatch.delattr(os, missing, raising=False)

    with pytest.raises(
        OSError, match="^directory no-follow support is unavailable$"
    ) as error:
        dataset_audit._open_directory_no_follow(tmp_path, create=False)

    assert str(error.value) == "directory no-follow support is unavailable"


def test_open_directory_no_follow_combines_all_required_open_flags(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(dataset_audit, "_directory_fd_supported", lambda: True)
    monkeypatch.setattr(os, "O_DIRECTORY", 0o40000, raising=False)
    monkeypatch.setattr(os, "O_NOFOLLOW", 0o100000, raising=False)
    flags_seen: list[int] = []
    next_fd = count(10)

    def fake_open(_path: object, flags: int, **_kwargs: object) -> int:
        flags_seen.append(flags)
        return next(next_fd)

    monkeypatch.setattr(os, "open", fake_open)
    monkeypatch.setattr(os, "close", lambda _fd: None)

    dataset_audit._open_directory_no_follow(tmp_path, create=False)

    expected_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    assert flags_seen
    assert all(flags == expected_flags for flags in flags_seen)


def test_audit_counts_labels_polygons_splits_and_training_text_metrics() -> None:
    result = audit_rows(
        _balanced_rows(),
        validation_fraction=0.5,
        seed=7,
    )

    assert isinstance(result, AuditResult)
    assert isinstance(result.report, AuditReport)
    assert is_dataclass(result.report)
    assert is_dataclass(result)
    assert not hasattr(result.report, "__dict__")
    assert not hasattr(result, "__dict__")

    report = result.report
    assert report.dataset_id == LANDUSE_DATASET_CONTRACT.dataset_id
    assert report.config == LANDUSE_DATASET_CONTRACT.config
    assert report.dataset_split == LANDUSE_DATASET_CONTRACT.split
    assert report.region == LANDUSE_DATASET_CONTRACT.region
    assert (
        report.repository_revision
        == LANDUSE_DATASET_CONTRACT.provenance.repository_revision
    )
    assert report.parquet_sha256 == LANDUSE_DATASET_CONTRACT.provenance.parquet_sha256
    assert report.validation_fraction == 0.5
    assert report.seed == 7

    assert report.total_rows == 6
    assert report.trainable_rows == 4
    assert report.total_polygons == 2
    assert report.trainable_polygons == 2
    assert report.label_counts == (
        ("no", 2),
        ("uncertain", 2),
        ("yes", 2),
    )
    assert report.split_row_counts == (("train", 3), ("validation", 3))
    assert report.split_polygon_counts == (("train", 1), ("validation", 1))
    assert report.trainable_label_counts == (
        ("train", (("no", 1), ("yes", 1))),
        ("validation", (("no", 1), ("yes", 1))),
    )
    assert report.language_counts == (("en", 3), ("fr", 1))
    assert report.source_counts == (("wikipedia", 3), ("wikivoyage", 1))
    assert report.text_length_min == 2
    assert report.text_length_max == 4
    assert report.text_length_mean == pytest.approx(3.25)
    assert report.mixed_label_polygons == 2
    assert report.cross_split_duplicate_groups == 0
    assert report.conflicting_content_hash_groups == 0
    assert report.review_required_reasons == ()
    assert report.ready is True

    expected_manifest = tuple(
        sorted(
            {
                (
                    row["polygon_id"],
                    split_for_polygon(
                        row["polygon_id"], validation_fraction=0.5, seed=7
                    ),
                )
                for row in _balanced_rows()
            }
        )
    )
    assert result.split_manifest == expected_manifest

    with pytest.raises(FrozenInstanceError):
        report.total_rows = 0  # type: ignore[misc]


def test_audit_uses_custom_training_labels_for_counts_and_conflicts() -> None:
    custom_contract = replace(
        LANDUSE_DATASET_CONTRACT,
        supported_label_values=("negative", "positive", "uncertain"),
        training_label_values=("negative", "positive"),
    )
    train_polygon = _polygon_for_split("train", seed=23)
    validation_polygon = _polygon_for_split("validation", seed=23)

    def custom_row(
        *, sentence_id: str, polygon_id: str, label: str, text: str, content_hash: str
    ) -> dict[str, object]:
        row = _row(
            sentence_id=sentence_id,
            polygon_id=polygon_id,
            label=label,
            text=text,
            sentence_content_hash=content_hash,
        )
        return row

    rows = [
        custom_row(
            sentence_id="train-negative",
            polygon_id=train_polygon,
            label="negative",
            text="Train negative",
            content_hash="conflicting-hash",
        ),
        custom_row(
            sentence_id="train-positive",
            polygon_id=train_polygon,
            label="positive",
            text="Train positive",
            content_hash="conflicting-hash",
        ),
        custom_row(
            sentence_id="train-uncertain",
            polygon_id=train_polygon,
            label="uncertain",
            text="Train uncertain",
            content_hash="train-uncertain-hash",
        ),
        custom_row(
            sentence_id="validation-negative",
            polygon_id=validation_polygon,
            label="negative",
            text="Validation negative",
            content_hash="validation-negative-hash",
        ),
        custom_row(
            sentence_id="validation-positive",
            polygon_id=validation_polygon,
            label="positive",
            text="Validation positive",
            content_hash="validation-positive-hash",
        ),
        custom_row(
            sentence_id="validation-uncertain",
            polygon_id=validation_polygon,
            label="uncertain",
            text="Validation uncertain",
            content_hash="validation-uncertain-hash",
        ),
    ]

    report = audit_rows(
        rows,
        validation_fraction=0.5,
        seed=23,
        contract=custom_contract,
    ).report

    assert report.trainable_rows == 4
    assert report.trainable_label_counts == (
        ("train", (("negative", 1), ("positive", 1))),
        ("validation", (("negative", 1), ("positive", 1))),
    )
    assert report.mixed_label_polygons == 2
    assert report.conflicting_content_hash_groups == 1
    assert report.review_required_reasons == ("content_hash_label_conflicts",)
    assert report.ready is False


def test_audit_counts_duplicate_hash_risks_and_split_readiness_reasons() -> None:
    train_polygon = _polygon_for_split("train", seed=11)
    validation_polygon = _polygon_for_split("validation", seed=11)
    rows = [
        _row(
            sentence_id="duplicate-1",
            polygon_id=train_polygon,
            label="no",
            text="First",
            sentence_content_hash="same-hash",
        ),
        _row(
            sentence_id="duplicate-2",
            polygon_id=train_polygon,
            label="yes",
            text="Second",
            sentence_content_hash="same-hash",
        ),
        _row(
            sentence_id="duplicate-3",
            polygon_id=validation_polygon,
            label="no",
            text="Third",
            sentence_content_hash="same-hash",
        ),
    ]

    report = audit_rows(rows, validation_fraction=0.5, seed=11).report

    assert report.duplicate_hash_groups == 1
    assert report.duplicate_rows_beyond_first == 2
    assert report.cross_polygon_duplicate_groups == 1
    assert report.mixed_label_polygons == 1
    assert report.cross_split_duplicate_groups == 1
    assert report.conflicting_content_hash_groups == 1
    assert report.review_required_reasons == (
        "content_hash_label_conflicts",
        "cross_split_duplicate_groups",
        "validation_split_missing_label",
    )
    assert report.ready is False


def test_same_hash_with_both_trainable_labels_is_a_content_hash_blocker() -> None:
    train_polygon = _polygon_for_split("train", seed=13)
    validation_polygon = _polygon_for_split("validation", seed=13)
    rows = [
        _row(
            sentence_id="conflict-no",
            polygon_id=train_polygon,
            label="no",
            text="Same sentence",
            sentence_content_hash="conflicting-hash",
        ),
        _row(
            sentence_id="conflict-yes",
            polygon_id=train_polygon,
            label="yes",
            text="Same sentence",
            sentence_content_hash="conflicting-hash",
        ),
        _row(
            sentence_id="validation-no",
            polygon_id=validation_polygon,
            label="no",
            text="Validation no",
        ),
        _row(
            sentence_id="validation-yes",
            polygon_id=validation_polygon,
            label="yes",
            text="Validation yes",
        ),
    ]

    report = audit_rows(rows, validation_fraction=0.5, seed=13).report

    assert report.duplicate_hash_groups == 1
    assert report.duplicate_rows_beyond_first == 1
    assert report.cross_polygon_duplicate_groups == 0
    assert report.cross_split_duplicate_groups == 0
    assert report.conflicting_content_hash_groups == 1
    assert report.mixed_label_polygons == 2
    assert report.review_required_reasons == ("content_hash_label_conflicts",)
    assert report.ready is False


def test_duplicate_hash_spanning_splits_is_a_cross_split_blocker() -> None:
    train_polygon = _polygon_for_split("train", seed=17)
    validation_polygon = _polygon_for_split("validation", seed=17)
    rows = [
        _row(
            sentence_id="split-train-no",
            polygon_id=train_polygon,
            label="no",
            text="Repeated sentence",
            sentence_content_hash="cross-split-hash",
        ),
        _row(
            sentence_id="split-train-yes",
            polygon_id=train_polygon,
            label="yes",
            text="Train yes",
        ),
        _row(
            sentence_id="split-validation-no",
            polygon_id=validation_polygon,
            label="no",
            text="Repeated sentence",
            sentence_content_hash="cross-split-hash",
        ),
        _row(
            sentence_id="split-validation-yes",
            polygon_id=validation_polygon,
            label="yes",
            text="Validation yes",
        ),
    ]

    report = audit_rows(rows, validation_fraction=0.5, seed=17).report

    assert report.duplicate_hash_groups == 1
    assert report.duplicate_rows_beyond_first == 1
    assert report.cross_polygon_duplicate_groups == 1
    assert report.cross_split_duplicate_groups == 1
    assert report.conflicting_content_hash_groups == 0
    assert report.mixed_label_polygons == 2
    assert report.review_required_reasons == ("cross_split_duplicate_groups",)
    assert report.ready is False


def test_audit_ignores_non_string_sentence_content_hashes() -> None:
    polygon_id = _polygon_for_split("train", seed=19)
    rows = [
        _row(
            sentence_id="non-string-hash-no",
            polygon_id=polygon_id,
            label="no",
            text="No",
            sentence_content_hash=123,
        ),
        _row(
            sentence_id="non-string-hash-yes",
            polygon_id=polygon_id,
            label="yes",
            text="Yes",
            sentence_content_hash=123,
        ),
    ]

    report = audit_rows(rows, validation_fraction=0.5, seed=19).report

    assert report.duplicate_hash_groups == 0
    assert report.conflicting_content_hash_groups == 0


def test_audit_keeps_different_content_hashes_in_separate_groups() -> None:
    polygon_id = _polygon_for_split("train", seed=29)
    rows = [
        _row(
            sentence_id="separate-hash-no",
            polygon_id=polygon_id,
            label="no",
            text="No",
            sentence_content_hash="hash-no",
        ),
        _row(
            sentence_id="separate-hash-yes",
            polygon_id=polygon_id,
            label="yes",
            text="Yes",
            sentence_content_hash="hash-yes",
        ),
    ]

    report = audit_rows(rows, validation_fraction=0.5, seed=29).report

    assert report.conflicting_content_hash_groups == 0


def test_audit_accumulates_repeated_trainable_labels() -> None:
    rows = _balanced_rows()
    rows.append(
        _row(
            sentence_id="sentence-duplicate-no",
            polygon_id=cast(str, rows[0]["polygon_id"]),
            label="no",
            text="Another no",
        )
    )

    report = audit_rows(rows, validation_fraction=0.5, seed=7).report

    assert report.trainable_label_counts == (
        ("train", (("no", 2), ("yes", 1))),
        ("validation", (("no", 1), ("yes", 1))),
    )


def test_audit_counts_all_polygons_in_their_assigned_split() -> None:
    rows = [
        _row(
            sentence_id="train-one",
            polygon_id="train-one",
            label="no",
            text="One",
        ),
        _row(
            sentence_id="train-two",
            polygon_id="train-two",
            label="yes",
            text="Two",
        ),
    ]

    report = audit_rows(rows, validation_fraction=0.0).report

    assert report.split_polygon_counts == (("train", 2), ("validation", 0))


def test_audit_does_not_call_a_unique_hash_a_duplicate() -> None:
    report = audit_rows(
        [
            _row(
                sentence_id="unique-hash",
                polygon_id="unique-polygon",
                label="yes",
                text="Unique",
                sentence_content_hash="unique-hash",
            )
        ],
        validation_fraction=0.0,
    ).report

    assert report.duplicate_hash_groups == 0
    assert report.duplicate_rows_beyond_first == 0


def test_audit_wraps_split_assignment_errors_with_the_row_number(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_split(*_args: object, **_kwargs: object) -> str:
        raise ValueError("split failed")

    monkeypatch.setattr(dataset_audit, "split_for_polygon", fail_split)

    with pytest.raises(DatasetAuditError) as error:
        audit_rows(
            [
                _row(
                    sentence_id="split-error",
                    polygon_id="polygon",
                    label="yes",
                    text="Text",
                )
            ]
        )

    assert str(error.value) == "row 1: split failed"
    assert isinstance(error.value.__cause__, ValueError)


def test_audit_uses_the_documented_default_split_seed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[float, int]] = []

    def record_split(
        _polygon_id: str,
        *,
        validation_fraction: float,
        seed: int,
    ) -> str:
        calls.append((validation_fraction, seed))
        return "train"

    monkeypatch.setattr(dataset_audit, "split_for_polygon", record_split)

    audit_rows(
        [
            _row(
                sentence_id="default-seed",
                polygon_id="polygon",
                label="yes",
                text="Text",
            )
        ]
    )

    assert calls == [(0.2, 42)]


def test_audit_wraps_schema_contract_and_identifier_failures_with_row_numbers() -> None:
    valid = _row(
        sentence_id="sentence-1",
        polygon_id="polygon-1",
        label="yes",
        text="Valid",
    )
    reordered = {
        column: valid[column]
        for column in (
            LANDUSE_DATASET_CONTRACT.required_columns[:2]
            + (
                LANDUSE_DATASET_CONTRACT.required_columns[3],
                LANDUSE_DATASET_CONTRACT.required_columns[2],
            )
            + LANDUSE_DATASET_CONTRACT.required_columns[4:]
        )
    }
    with pytest.raises(DatasetAuditError, match=r"row 1.*required column order"):
        audit_rows([reordered])

    invalid_identifier = valid.copy()
    invalid_identifier["sentence_id"] = "   "
    with pytest.raises(DatasetAuditError) as error:
        audit_rows([valid, invalid_identifier])
    assert str(error.value) == "row 2: sentence_id must be a non-empty string"

    invalid_polygon = valid.copy()
    invalid_polygon["polygon_id"] = ""
    with pytest.raises(DatasetAuditError) as error:
        audit_rows([valid, invalid_polygon])
    assert str(error.value) == "row 2: polygon_id must be a non-empty string"

    invalid_contract = valid.copy()
    invalid_contract["region"] = "iran"
    with pytest.raises(DatasetAuditError, match=r"row 2.*region"):
        audit_rows([valid, invalid_contract])


def test_audit_rejects_extra_columns_on_a_subsequent_row() -> None:
    valid = _row(
        sentence_id="sentence-1",
        polygon_id="polygon-1",
        label="yes",
        text="Valid",
    )
    extra_column = valid.copy()
    extra_column["unexpected_column"] = "not in the contract"

    with pytest.raises(DatasetAuditError, match=r"row 2.*unexpected columns"):
        audit_rows([valid, extra_column])


def test_audit_rejects_reordered_columns_on_a_subsequent_row() -> None:
    valid = _row(
        sentence_id="sentence-1",
        polygon_id="polygon-1",
        label="yes",
        text="Valid",
    )
    reordered = {
        column: valid[column]
        for column in (
            LANDUSE_DATASET_CONTRACT.required_columns[:2]
            + (
                LANDUSE_DATASET_CONTRACT.required_columns[3],
                LANDUSE_DATASET_CONTRACT.required_columns[2],
            )
            + LANDUSE_DATASET_CONTRACT.required_columns[4:]
        )
    }

    with pytest.raises(DatasetAuditError, match=r"row 2.*required column order"):
        audit_rows([valid, reordered])


def test_audit_consumes_a_one_pass_iterator_without_materializing_rows() -> None:
    rows = _balanced_rows()

    class OnePassRows:
        def __init__(self, values: list[dict[str, object]]) -> None:
            self.values = values
            self.iteration_count = 0

        def __iter__(self) -> Iterator[dict[str, object]]:
            self.iteration_count += 1
            if self.iteration_count > 1:
                raise AssertionError("rows were iterated more than once")
            yield from self.values

    one_pass_rows = OnePassRows(rows)

    result = audit_rows(one_pass_rows, validation_fraction=0.5, seed=7)

    assert result.report.total_rows == len(rows)
    assert one_pass_rows.iteration_count == 1


def test_empty_input_returns_a_zero_report() -> None:
    report = audit_rows([]).report

    assert report.total_rows == 0
    assert report.trainable_rows == 0
    assert report.total_polygons == 0
    assert report.trainable_polygons == 0
    assert report.label_counts == (("no", 0), ("uncertain", 0), ("yes", 0))
    assert report.split_row_counts == (("train", 0), ("validation", 0))
    assert report.split_polygon_counts == (("train", 0), ("validation", 0))
    assert report.trainable_label_counts == (
        ("train", (("no", 0), ("yes", 0))),
        ("validation", (("no", 0), ("yes", 0))),
    )
    assert report.text_length_min is None
    assert report.text_length_max is None
    assert report.text_length_mean is None


def test_write_audit_artifacts_uses_only_the_approved_audit_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = audit_rows(_balanced_rows(), validation_fraction=0.5, seed=7)
    approved_root = ProjectConfig().data_root
    prepared_paths: list[Path] = []
    writes: dict[Path, str] = {}

    class RecordingManagedPaths:
        def __init__(self, config: ProjectConfig) -> None:
            assert config == ProjectConfig()

        def child(self, relative_path: str) -> Path:
            assert relative_path == "audit/landuse"
            return approved_root / relative_path

    def record_prepare(path: Path) -> None:
        prepared_paths.append(path)

    def record_write_json(path: Path, payload: object) -> None:
        writes[path] = json.dumps(payload, indent=2, sort_keys=True) + "\n"

    monkeypatch.setattr(
        "osm_polygon_sentence_classifier.dataset_audit.ManagedPaths",
        RecordingManagedPaths,
    )
    monkeypatch.setattr(
        "osm_polygon_sentence_classifier.dataset_audit._prepare_audit_directory",
        record_prepare,
    )
    monkeypatch.setattr(
        "osm_polygon_sentence_classifier.dataset_audit._write_json",
        record_write_json,
    )

    report_path, manifest_path = write_audit_artifacts(result)

    audit_directory = approved_root / "audit/landuse"
    assert prepared_paths == [audit_directory]
    assert (report_path, manifest_path) == (
        audit_directory / "audit_report.json",
        audit_directory / "split_manifest.json",
    )
    assert set(writes) == {report_path, manifest_path}
    assert writes[report_path].endswith("\n")
    assert writes[manifest_path].endswith("\n")
    assert "Train text" not in writes[report_path]
    assert "sentence_text_normalized" not in writes[report_path]

    report_payload = json.loads(writes[report_path])
    assert report_payload["dataset_id"] == LANDUSE_DATASET_CONTRACT.dataset_id
    assert report_payload["ready"] is True
    assert report_payload["label_counts"] == {
        "no": 2,
        "uncertain": 2,
        "yes": 2,
    }
    assert report_payload["mixed_label_polygons"] == 2
    assert report_payload["cross_split_duplicate_groups"] == 0
    assert report_payload["conflicting_content_hash_groups"] == 0
    assert "conflicting_polygons" not in report_payload
    assert "polygon_label_conflicts" not in writes[report_path]
    assert json.loads(writes[manifest_path]) == dict(result.split_manifest)


def test_write_audit_artifacts_wraps_managed_path_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = audit_rows(_balanced_rows(), validation_fraction=0.5, seed=7)

    class FailingManagedPaths:
        def __init__(self, _config: ProjectConfig) -> None:
            raise ManagedPathError("managed root unavailable")

    monkeypatch.setattr(dataset_audit, "ManagedPaths", FailingManagedPaths)

    with pytest.raises(DatasetAuditError) as error:
        write_audit_artifacts(result)

    assert str(error.value) == (
        "unable to prepare audit artifact directory under the managed root"
    )
    assert isinstance(error.value.__cause__, ManagedPathError)


def test_safe_artifact_path_reports_an_escape_from_the_audit_directory(
    tmp_path: Path,
) -> None:
    audit_directory = tmp_path / "audit"
    audit_directory.mkdir()

    with pytest.raises(DatasetAuditError) as error:
        dataset_audit._safe_artifact_path(audit_directory, "../outside.json")

    assert str(error.value) == (
        "artifact path must remain beneath the audit directory: "
        f"{audit_directory / '../outside.json'}"
    )


def test_write_audit_artifacts_writes_documented_json_payloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    rows = _balanced_rows()
    for index, row in enumerate(rows):
        row["sentence_text_normalized"] = f"unique-sentence-text-{index}"

    result = audit_rows(rows, validation_fraction=0.5, seed=7)
    audit_directory = tmp_path / "audit" / "landuse"

    class TemporaryManagedPaths:
        def __init__(self, config: ProjectConfig) -> None:
            assert config == ProjectConfig()

        def child(self, relative_path: str) -> Path:
            assert relative_path == "audit/landuse"
            return audit_directory

    monkeypatch.setattr(
        "osm_polygon_sentence_classifier.dataset_audit.ManagedPaths",
        TemporaryManagedPaths,
    )

    report_path, manifest_path = write_audit_artifacts(result)

    assert report_path == audit_directory / "audit_report.json"
    assert manifest_path == audit_directory / "split_manifest.json"
    assert sorted(path.name for path in audit_directory.iterdir()) == [
        "audit_report.json",
        "split_manifest.json",
    ]

    source_texts: list[str] = []
    for row in rows:
        text = row["sentence_text_normalized"]
        if isinstance(text, str):
            source_texts.append(text)

    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert report_payload == result.report.to_dict()
    assert manifest_payload == dict(result.split_manifest)
    assert "sentence_text_normalized" not in report_payload
    report_text = report_path.read_text(encoding="utf-8")
    assert all(text not in report_text for text in source_texts)


def test_write_json_rejects_a_final_symlink_without_following_it(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.json"
    outside.write_text("sentinel\n", encoding="utf-8")
    artifact_path = tmp_path / "audit_report.json"
    artifact_path.symlink_to(outside)

    with pytest.raises(DatasetAuditError, match="audit artifact"):
        _write_json(artifact_path, {"ready": True})

    assert outside.read_text(encoding="utf-8") == "sentinel\n"


def test_write_json_uses_the_stable_pretty_sorted_json_format(tmp_path: Path) -> None:
    artifact_path = tmp_path / "audit_report.json"

    _write_json(artifact_path, {"z": 1, "a": 2})

    assert artifact_path.read_text(encoding="utf-8") == ('{\n  "a": 2,\n  "z": 1\n}\n')


def test_write_json_fallback_uses_the_stable_pretty_sorted_json_format(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(dataset_audit, "_directory_fd_supported", lambda: False)
    artifact_path = tmp_path / "audit_report.json"

    _write_json(artifact_path, {"z": 1, "a": 2})

    assert artifact_path.read_text(encoding="utf-8") == ('{\n  "a": 2,\n  "z": 1\n}\n')


def test_write_json_passes_the_stable_serialization_options_to_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}
    original_dumps = cast(Any, dataset_audit.json.dumps)

    def recording_dumps(value: object, *args: object, **kwargs: object) -> str:
        observed["value"] = value
        observed["args"] = args
        observed["kwargs"] = kwargs
        return original_dumps(value, *args, **kwargs)

    monkeypatch.setattr(dataset_audit.json, "dumps", recording_dumps)
    payload = {"z": 1, "a": 2}

    _write_json(tmp_path / "audit_report.json", payload)

    assert observed == {
        "value": payload,
        "args": (),
        "kwargs": {"indent": 2, "sort_keys": True},
    }


def test_write_json_uses_secure_directory_fd_open_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {"close": []}
    fdopen_arguments: list[tuple[object, ...]] = []
    opened_directory = 17
    opened_file = 23

    monkeypatch.setattr(dataset_audit, "_directory_fd_supported", lambda: True)
    monkeypatch.setattr(
        dataset_audit,
        "_open_directory_no_follow",
        lambda path, *, create: calls.update({"directory": path, "create": create})
        or opened_directory,
    )

    def fake_open(*arguments: object, **kwargs: object) -> int:
        calls["open"] = arguments
        calls["open_kwargs"] = kwargs
        return opened_file

    def fake_fdopen(*arguments: object, **kwargs: object) -> io.StringIO:
        fdopen_arguments.append((*arguments, kwargs))
        return io.StringIO()

    def fake_close(file_descriptor: object) -> None:
        cast(list[object], calls["close"]).append(file_descriptor)

    monkeypatch.setattr(os, "open", fake_open)
    monkeypatch.setattr(os, "fdopen", fake_fdopen)
    monkeypatch.setattr(os, "close", fake_close)

    artifact_path = tmp_path / "audit_report.json"
    _write_json(artifact_path, {"ready": True})

    assert calls["directory"] == tmp_path
    assert calls["create"] is False
    assert calls["open"] == (
        artifact_path.name,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW,
        0o666,
    )
    assert calls["open_kwargs"] == {"dir_fd": opened_directory}
    assert fdopen_arguments == [(opened_file, "w", {"encoding": "utf-8"})]
    assert calls["close"] == [opened_directory]


def test_write_json_closes_open_file_and_directory_after_fdopen_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    closed: list[object] = []
    monkeypatch.setattr(dataset_audit, "_directory_fd_supported", lambda: True)
    monkeypatch.setattr(
        dataset_audit,
        "_open_directory_no_follow",
        lambda *_args, **_kwargs: 17,
    )
    monkeypatch.setattr(os, "open", lambda *_args, **_kwargs: 23)
    monkeypatch.setattr(
        os,
        "fdopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("cannot wrap")),
    )
    monkeypatch.setattr(os, "close", closed.append)

    with pytest.raises(DatasetAuditError, match="unable to write audit artifact"):
        _write_json(tmp_path / "audit_report.json", {"ready": True})

    assert closed == [23, 17]


def test_write_json_fallback_uses_the_secure_opener_and_utf8(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}
    monkeypatch.setattr(dataset_audit, "_directory_fd_supported", lambda: False)

    def fake_open(*arguments: object, **kwargs: object) -> io.StringIO:
        calls["open"] = arguments
        calls["open_kwargs"] = kwargs
        return io.StringIO()

    def fake_os_open(*arguments: object, **kwargs: object) -> int:
        calls["opener"] = arguments
        calls["opener_kwargs"] = kwargs
        return 1

    monkeypatch.setattr("builtins.open", fake_open)
    monkeypatch.setattr(os, "open", fake_os_open)

    artifact_path = tmp_path / "audit_report.json"
    _write_json(artifact_path, {"ready": True})

    open_arguments = cast(tuple[object, ...], calls["open"])
    open_kwargs = cast(dict[str, object], calls["open_kwargs"])
    assert open_arguments[:2] == (artifact_path, "w")
    assert open_kwargs["encoding"] == "utf-8"
    opener = open_kwargs["opener"]
    assert callable(opener)

    cast(Callable[[str, int], int], opener)("artifact", os.O_CREAT)
    assert calls["opener"] == (
        "artifact",
        os.O_CREAT | getattr(os, "O_NOFOLLOW", 0),
        0o666,
    )
    assert calls["opener_kwargs"] == {}


def test_write_json_fallback_treats_missing_no_follow_flag_as_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: dict[str, object] = {}
    monkeypatch.setattr(dataset_audit, "_directory_fd_supported", lambda: False)
    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)

    def fake_open(*arguments: object, **kwargs: object) -> io.StringIO:
        calls["open_kwargs"] = kwargs
        return io.StringIO()

    def fake_os_open(*arguments: object, **kwargs: object) -> int:
        calls["opener"] = arguments
        return 1

    monkeypatch.setattr("builtins.open", fake_open)
    monkeypatch.setattr(os, "open", fake_os_open)

    artifact_path = tmp_path / "audit_report.json"
    _write_json(artifact_path, {"ready": True})
    opener = cast(dict[str, object], calls["open_kwargs"])["opener"]
    cast(Callable[[str, int], int], opener)("artifact", os.O_CREAT)

    assert calls["opener"] == ("artifact", os.O_CREAT, 0o666)


def test_write_json_does_not_close_an_unopened_directory_fd(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    closed: list[object] = []
    monkeypatch.setattr(dataset_audit, "_directory_fd_supported", lambda: True)
    monkeypatch.setattr(
        dataset_audit,
        "_open_directory_no_follow",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("cannot open")),
    )
    monkeypatch.setattr(os, "close", closed.append)

    with pytest.raises(DatasetAuditError, match="unable to write audit artifact"):
        _write_json(tmp_path / "audit_report.json", {"ready": True})

    assert closed == []


def test_write_audit_artifacts_rejects_symlinked_final_paths_without_writing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = audit_rows(_balanced_rows(), validation_fraction=0.5, seed=7)
    audit_directory = tmp_path / "audit" / "landuse"
    audit_directory.mkdir(parents=True)
    outside_report = tmp_path / "outside-report.json"
    outside_manifest = tmp_path / "outside-manifest.json"
    outside_report.write_text("report sentinel\n", encoding="utf-8")
    outside_manifest.write_text("manifest sentinel\n", encoding="utf-8")
    (audit_directory / "audit_report.json").symlink_to(outside_report)
    (audit_directory / "split_manifest.json").symlink_to(outside_manifest)

    class TemporaryManagedPaths:
        def __init__(self, config: ProjectConfig) -> None:
            assert config == ProjectConfig()

        def child(self, relative_path: str) -> Path:
            assert relative_path == "audit/landuse"
            return audit_directory

    monkeypatch.setattr(
        "osm_polygon_sentence_classifier.dataset_audit.ManagedPaths",
        TemporaryManagedPaths,
    )

    with pytest.raises(DatasetAuditError, match="symlink"):
        write_audit_artifacts(result)

    assert outside_report.read_text(encoding="utf-8") == "report sentinel\n"
    assert outside_manifest.read_text(encoding="utf-8") == "manifest sentinel\n"


@pytest.mark.skipif(
    not all(hasattr(os, flag) for flag in ("O_DIRECTORY", "O_NOFOLLOW"))
    or os.open not in os.supports_dir_fd,
    reason="directory-FD no-following is not available on this platform",
)
def test_write_audit_artifacts_rejects_a_symlinked_audit_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = audit_rows(_balanced_rows(), validation_fraction=0.5, seed=7)
    outside_directory = tmp_path / "outside"
    outside_directory.mkdir()
    outside_report = outside_directory / "audit_report.json"
    outside_manifest = outside_directory / "split_manifest.json"
    outside_report.write_text("report sentinel\n", encoding="utf-8")
    outside_manifest.write_text("manifest sentinel\n", encoding="utf-8")

    audit_directory = tmp_path / "audit" / "landuse"
    audit_directory.parent.mkdir()
    audit_directory.symlink_to(outside_directory, target_is_directory=True)

    class TemporaryManagedPaths:
        def __init__(self, config: ProjectConfig) -> None:
            assert config == ProjectConfig()

        def child(self, relative_path: str) -> Path:
            assert relative_path == "audit/landuse"
            return audit_directory

    monkeypatch.setattr(
        "osm_polygon_sentence_classifier.dataset_audit.ManagedPaths",
        TemporaryManagedPaths,
    )

    with pytest.raises(DatasetAuditError, match="audit artifact"):
        write_audit_artifacts(result)

    assert outside_report.read_text(encoding="utf-8") == "report sentinel\n"
    assert outside_manifest.read_text(encoding="utf-8") == "manifest sentinel\n"


@pytest.mark.skipif(
    not all(hasattr(os, flag) for flag in ("O_DIRECTORY", "O_NOFOLLOW")),
    reason="directory no-following is not available on this platform",
)
def test_write_audit_artifacts_rejects_an_intermediate_directory_symlink(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result = audit_rows(_balanced_rows(), validation_fraction=0.5, seed=7)
    outside_directory = tmp_path / "outside"
    outside_directory.mkdir()
    audit_directory = tmp_path / "audit" / "landuse"
    audit_directory.parent.symlink_to(outside_directory, target_is_directory=True)

    class TemporaryManagedPaths:
        def __init__(self, config: ProjectConfig) -> None:
            assert config == ProjectConfig()

        def child(self, relative_path: str) -> Path:
            assert relative_path == "audit/landuse"
            return audit_directory

    monkeypatch.setattr(
        "osm_polygon_sentence_classifier.dataset_audit.ManagedPaths",
        TemporaryManagedPaths,
    )

    with pytest.raises(DatasetAuditError, match="audit artifact"):
        write_audit_artifacts(result)

    assert not (outside_directory / "landuse").exists()


def test_audit_directory_fallback_creates_nested_directories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audit_directory = tmp_path / "audit" / "landuse"
    monkeypatch.setattr(dataset_audit, "_directory_fd_supported", lambda: False)

    dataset_audit._prepare_audit_directory_without_directory_fds(audit_directory)

    assert audit_directory.is_dir()


def test_prepare_audit_directory_forwards_the_exact_fallback_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audit_directory = tmp_path / "audit" / "landuse"
    prepared: list[Path] = []
    monkeypatch.setattr(dataset_audit, "_directory_fd_supported", lambda: False)
    monkeypatch.setattr(
        dataset_audit,
        "_prepare_audit_directory_without_directory_fds",
        prepared.append,
    )

    dataset_audit._prepare_audit_directory(audit_directory)

    assert prepared == [audit_directory]


def test_audit_directory_fallback_tolerates_a_concurrent_component_creation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(Path, "is_symlink", lambda _path: False)
    monkeypatch.setattr(Path, "exists", lambda _path: False)
    monkeypatch.setattr(
        Path,
        "mkdir",
        lambda _path: (_ for _ in ()).throw(FileExistsError("created concurrently")),
    )
    monkeypatch.setattr(Path, "is_dir", lambda _path: True)

    dataset_audit._prepare_audit_directory_without_directory_fds(
        tmp_path / "audit" / "landuse"
    )


def test_audit_directory_fallback_rejects_an_intermediate_symlink(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    audit_parent = tmp_path / "audit"
    audit_parent.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(dataset_audit, "_directory_fd_supported", lambda: False)

    with pytest.raises(DatasetAuditError, match="symlink"):
        dataset_audit._prepare_audit_directory_without_directory_fds(
            audit_parent / "landuse"
        )

    assert not (outside / "landuse").exists()


def test_audit_directory_fallback_rejects_a_file_component(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audit_parent = tmp_path / "audit"
    audit_parent.write_text("not a directory", encoding="utf-8")
    monkeypatch.setattr(dataset_audit, "_directory_fd_supported", lambda: False)

    with pytest.raises(DatasetAuditError, match="not a directory"):
        dataset_audit._prepare_audit_directory_without_directory_fds(
            audit_parent / "landuse"
        )
