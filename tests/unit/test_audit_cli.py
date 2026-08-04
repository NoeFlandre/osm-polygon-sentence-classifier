from itertools import count
from pathlib import Path

import pytest

from osm_polygon_sentence_classifier import audit_cli
from osm_polygon_sentence_classifier.dataset_audit import audit_rows
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
    return row


def _polygon_for_split(split: str, *, seed: int) -> str:
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


def _ready_result():
    train_polygon = _polygon_for_split("train", seed=19)
    validation_polygon = _polygon_for_split("validation", seed=19)
    train_yes_polygon = next(
        candidate
        for suffix in range(1, 100)
        for candidate in (f"polygon-train-other-{suffix}",)
        if split_for_polygon(candidate, validation_fraction=0.5, seed=19) == "train"
    )
    validation_yes_polygon = next(
        candidate
        for suffix in range(1, 100)
        for candidate in (f"polygon-validation-other-{suffix}",)
        if split_for_polygon(candidate, validation_fraction=0.5, seed=19)
        == "validation"
    )
    rows = [
        _row(
            sentence_id="ready-train-no",
            polygon_id=train_polygon,
            label="no",
            text="No",
        ),
        _row(
            sentence_id="ready-train-yes",
            polygon_id=train_yes_polygon,
            label="yes",
            text="Yes",
        ),
        _row(
            sentence_id="ready-validation-no",
            polygon_id=validation_polygon,
            label="no",
            text="No",
        ),
        _row(
            sentence_id="ready-validation-yes",
            polygon_id=validation_yes_polygon,
            label="yes",
            text="Yes",
        ),
    ]
    assert split_for_polygon(train_polygon, validation_fraction=0.5, seed=19) == "train"
    assert split_for_polygon(validation_polygon, validation_fraction=0.5, seed=19) == (
        "validation"
    )
    return audit_rows(rows, validation_fraction=0.5, seed=19)


def test_main_loads_audits_writes_and_prints_ready_artifact_paths(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = _ready_result()
    calls: list[tuple[str, object]] = []
    expected_rows = object()
    expected_paths = (
        Path("/tmp/audit_report.json"),
        Path("/tmp/split_manifest.json"),
    )

    def fake_load_streaming_rows() -> object:
        calls.append(("load", None))
        return expected_rows

    def fake_audit_rows(rows: object):
        calls.append(("audit", rows))
        assert rows is expected_rows
        return result

    def fake_write_audit_artifacts(received_result: object) -> tuple[Path, Path]:
        calls.append(("write", received_result))
        assert received_result is result
        return expected_paths

    monkeypatch.setattr(audit_cli, "load_streaming_rows", fake_load_streaming_rows)
    monkeypatch.setattr(audit_cli, "audit_rows", fake_audit_rows)
    monkeypatch.setattr(audit_cli, "write_audit_artifacts", fake_write_audit_artifacts)

    assert audit_cli.main() is None

    assert calls == [("load", None), ("audit", expected_rows), ("write", result)]
    output = capsys.readouterr().out
    assert str(expected_paths[0]) in output
    assert str(expected_paths[1]) in output
    assert "readiness: True" in output


def test_main_writes_review_artifacts_then_exits_with_status_two(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    train_polygon = _polygon_for_split("train", seed=23)
    result = audit_rows(
        [
            _row(
                sentence_id="review-train-no",
                polygon_id=train_polygon,
                label="no",
                text="No",
            )
        ],
        validation_fraction=0.5,
        seed=23,
    )
    calls: list[object] = []

    monkeypatch.setattr(audit_cli, "load_streaming_rows", lambda: object())
    monkeypatch.setattr(audit_cli, "audit_rows", lambda rows: result)

    def fake_write_audit_artifacts(received_result: object) -> tuple[Path, Path]:
        calls.append(received_result)
        return Path("/tmp/audit_report.json"), Path("/tmp/split_manifest.json")

    monkeypatch.setattr(audit_cli, "write_audit_artifacts", fake_write_audit_artifacts)

    with pytest.raises(SystemExit) as raised:
        audit_cli.main()

    assert raised.value.code == 2
    assert calls == [result]
    output = capsys.readouterr().out
    assert "readiness: False" in output
