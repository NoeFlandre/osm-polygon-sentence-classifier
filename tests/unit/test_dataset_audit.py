import json
from collections.abc import Iterator
from dataclasses import FrozenInstanceError, is_dataclass
from itertools import count
from pathlib import Path

import pytest

from osm_polygon_sentence_classifier.config import ProjectConfig
from osm_polygon_sentence_classifier.dataset_audit import (
    AuditReport,
    AuditResult,
    DatasetAuditError,
    audit_rows,
    write_audit_artifacts,
)
from osm_polygon_sentence_classifier.dataset_contract import (
    LANDUSE_DATASET_CONTRACT,
)
from osm_polygon_sentence_classifier.dataset_loader import split_for_polygon


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
    assert report.review_required_reasons == ("polygon_label_conflicts",)
    assert report.ready is False

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
    assert report.conflicting_polygons == 1
    assert report.review_required_reasons == (
        "cross_polygon_duplicate_groups",
        "polygon_label_conflicts",
        "validation_split_missing_label",
    )
    assert report.ready is False


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
    with pytest.raises(DatasetAuditError, match=r"row 2.*sentence_id"):
        audit_rows([valid, invalid_identifier])

    invalid_contract = valid.copy()
    invalid_contract["region"] = "iran"
    with pytest.raises(DatasetAuditError, match=r"row 2.*region"):
        audit_rows([valid, invalid_contract])


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
    assert report.text_length_min is None
    assert report.text_length_max is None
    assert report.text_length_mean is None


def test_write_audit_artifacts_uses_only_the_approved_audit_directory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = audit_rows(_balanced_rows(), validation_fraction=0.5, seed=7)
    approved_root = ProjectConfig().data_root
    mkdir_paths: list[Path] = []
    writes: dict[Path, str] = {}

    class RecordingManagedPaths:
        def __init__(self, config: ProjectConfig) -> None:
            assert config == ProjectConfig()

        def child(self, relative_path: str) -> Path:
            assert relative_path == "audit/landuse"
            return approved_root / relative_path

    def record_mkdir(
        path: Path,
        *,
        parents: bool = False,
        exist_ok: bool = False,
    ) -> None:
        assert parents is True
        assert exist_ok is True
        mkdir_paths.append(path)

    def record_write_text(
        path: Path,
        data: str,
        *,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> int:
        assert encoding == "utf-8"
        assert errors is None
        assert newline is None
        writes[path] = data
        return len(data)

    monkeypatch.setattr(
        "osm_polygon_sentence_classifier.dataset_audit.ManagedPaths",
        RecordingManagedPaths,
    )
    monkeypatch.setattr(Path, "mkdir", record_mkdir)
    monkeypatch.setattr(Path, "write_text", record_write_text)

    report_path, manifest_path = write_audit_artifacts(result)

    audit_directory = approved_root / "audit/landuse"
    assert mkdir_paths == [audit_directory]
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
    assert report_payload["ready"] is False
    assert report_payload["label_counts"] == {
        "no": 2,
        "uncertain": 2,
        "yes": 2,
    }
    assert json.loads(writes[manifest_path]) == dict(result.split_manifest)


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
