import hashlib
import json

import pytest

from osm_polygon_sentence_classifier.ablation_reporting import (
    _definition_rows,
    _format_count,
    _public_specification,
    _result_rows,
    _source_commit_history,
    _source_history_note,
    _status_label,
    render_public_documents,
)
from osm_polygon_sentence_classifier.ablation_study import (
    ABLATION_STUDY_ID,
    render_study_documents,
    study_specification,
    study_specification_fingerprint,
)
from osm_polygon_sentence_classifier.tracking import TRACKIO_STATIC_SPACE_ID


def test_public_report_renderer_preserves_existing_document_bytes() -> None:
    specification = study_specification(
        source_commit="b" * 40,
        model_revision="a" * 40,
    )
    state = {
        "study_id": ABLATION_STUDY_ID,
        "fingerprint": study_specification_fingerprint(specification),
        "specification": specification,
        "phase": "running",
        "runs": {
            "a01-head-128|seed-42": {
                "ablation_id": "a01-head-128",
                "seed": 42,
                "run_id": "c" * 20,
                "phase": "completed",
                "metrics": {"eval_f1": 0.8, "eval_macro_f1": 0.7},
            }
        },
    }
    rows = [
        {
            "ablation_id": "a00-baseline-head-256-lr3e-4",
            "seed": 42,
            "status": "pending",
            "run_id": None,
            "source_commit": None,
            "metrics": {},
            "model_path": None,
        },
        {
            "ablation_id": "a01-head-128",
            "seed": 42,
            "status": "completed",
            "run_id": "c" * 20,
            "source_commit": None,
            "metrics": {"eval_f1": 0.8, "eval_macro_f1": 0.7},
            "model_path": "studies/landuse-v1/a01-head-128/run-cccccccccccccccccccc/final/",
        },
        *(
            {
                "ablation_id": ablation_id,
                "seed": 42,
                "status": "pending",
                "run_id": None,
                "source_commit": None,
                "metrics": {},
                "model_path": None,
            }
            for ablation_id in (
                "a02-head-512",
                "a03-head-lr1e-4",
                "a04-head-lr1e-3",
                "a05-balanced-head",
                "a06-last2-256",
            )
        ),
    ]

    documents = render_public_documents(
        state,
        rows=rows,
        study_id=ABLATION_STUDY_ID,
        tracking_space_id=TRACKIO_STATIC_SPACE_ID,
    )
    assert render_study_documents(state) == documents

    assert {
        path: hashlib.sha256(content.encode("utf-8")).hexdigest()
        for path, content in documents.items()
    } == {
        "README.md": "6e1cbc4d998c2e801bfcc820b94a05cab0acd6b010080f16d2e17449c39c4055",
        "studies/landuse-v1/README.md": "13d604bcb3b0c6880e410e72bc9548b85c9d10a5c5516be44ae4589d426cbeba",
        "studies/landuse-v1/results.json": "58dceaec2cbebc32b30d0fb18544b95b8128f820936bba451526f1611d216e07",
        "studies/landuse-v1/study.json": "c7eb9dfa6dab3d0295334e78739a801fe8201166f401fa634985718bb191fbd2",
    }


@pytest.mark.parametrize(
    ("metrics", "name", "expected"),
    [
        ({"support": 12}, "support", "12"),
        ({"support": 12.5}, "support", "12.5"),
        ({"support": True}, "support", "—"),
        ({"support": "12"}, "support", "—"),
        ({}, "support", "—"),
        ({"other": 12}, "support", "—"),
    ],
)
def test_format_count_preserves_count_and_invalid_value_formatting(
    metrics: dict[str, object], name: str, expected: str
) -> None:
    assert _format_count(metrics, name) == expected


def test_public_specification_requires_both_state_values_and_preserves_mapping() -> (
    None
):
    specification = {"definitions": []}
    assert _public_specification(
        {"specification": specification, "fingerprint": "f" * 64}
    ) == (specification, "f" * 64)

    for state in (
        {"specification": [], "fingerprint": "f" * 64},
        {"specification": specification, "fingerprint": 42},
    ):
        with pytest.raises(
            ValueError,
            match=r"\Aablation study state lacks its public specification\Z",
        ):
            _public_specification(state)


def test_definition_rows_skip_invalid_entries_and_render_missing_fields() -> None:
    header = [
        "| Ablation | Change | Maximum length | Learning rate | "
        "Trainable layers | Class weighting |",
        "|---|---|---:|---:|---|---|",
    ]
    assert _definition_rows({}) == header
    assert _definition_rows(
        {"definitions": [None, {}, {"label": "Only documented change"}]}
    ) == [
        *header,
        "| `—` | — | `—` | `—` | `—` | `—` |",
        "| `—` | Only documented change | `—` | `—` | `—` | `—` |",
    ]


def test_result_rows_render_every_metric_and_artifact_column() -> None:
    rows = _result_rows(
        [
            {
                "ablation_id": "a-test",
                "seed": 7,
                "status": "completed",
                "metrics": {
                    "eval_negative_support": 11,
                    "eval_positive_support": 2.5,
                    "eval_accuracy": 0.1,
                    "eval_precision": 0.2,
                    "eval_recall": 0.3,
                    "eval_f1": 0.4,
                    "eval_macro_f1": 0.5,
                    "eval_balanced_accuracy": 0.6,
                },
                "model_path": "studies/test/final/",
            }
        ],
        "study",
    )
    assert rows[-1] == (
        "| `study|a-test|seed-7` | `a-test` | 7 | completed | "
        "0.1000 | 0.2000 | 0.3000 | 0.4000 | 0.5000 | 0.6000 | "
        "11 / 2.5 | `studies/test/final/` |"
    )


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ({}, []),
        ({"source_commit_history": None}, []),
        ({"source_commit_history": "not-a-list"}, []),
        ({"source_commit_history": ["a", "b"]}, ["a", "b"]),
        ({"source_commit_history": ["a", 2]}, []),
    ],
)
def test_source_commit_history_accepts_only_string_lists(
    state: dict[str, object], expected: list[str]
) -> None:
    assert _source_commit_history(state) == expected


def test_source_history_note_is_empty_or_lists_each_prior_commit() -> None:
    assert _source_history_note([]) == ""
    assert (
        _source_history_note(["a", "b"])
        == "\n- Earlier source commits retained for historical run provenance: `a`, `b`"
    )


def test_status_label_distinguishes_completed_from_other_phases() -> None:
    assert _status_label("completed") == "Completed"
    assert _status_label("running") == "In-progress"


def test_public_renderer_preserves_optional_text_defaults_and_json_unicode() -> None:
    specification = {
        "definitions": [{"label": "é"}],
    }
    state = {
        "specification": specification,
        "fingerprint": "f" * 64,
        "phase": "completed",
        "source_commit_history": ["old"],
    }
    rows = [
        {
            "ablation_id": "é",
            "seed": 7,
            "status": "completed",
            "metrics": {},
            "model_path": None,
        }
    ]

    documents = render_public_documents(
        state,
        rows=rows,
        study_id="study",
        tracking_space_id="space",
        study_title="Custom title",
        study_introduction="Custom introduction",
        evaluation_note="Custom evaluation note",
        root_scope="Custom scope.",
        include_root_readme=False,
    )
    assert set(documents) == {
        "studies/study/README.md",
        "studies/study/study.json",
        "studies/study/results.json",
    }
    study_readme = documents["studies/study/README.md"]
    assert "Custom introduction" in study_readme
    assert "Custom evaluation note" in study_readme
    assert "Dataset revision: `not recorded`" in study_readme
    assert "Model revision: `not recorded`" in study_readme
    assert "Source commit: `not recorded`" in study_readme
    assert "Earlier source commits retained" in study_readme
    assert "`old`" in study_readme
    assert "é" in documents["studies/study/study.json"]
    assert "\\u00e9" not in documents["studies/study/study.json"]
    assert "é" in documents["studies/study/results.json"]
    assert "\\u00e9" not in documents["studies/study/results.json"]

    running_state = {"specification": specification, "fingerprint": "f" * 64}
    running_documents = render_public_documents(
        running_state,
        rows=[],
        study_id="study",
        tracking_space_id="space",
    )
    assert json.loads(running_documents["studies/study/results.json"])["phase"] == (
        "running"
    )

    completed_documents = render_public_documents(
        state,
        rows=[],
        study_id="study",
        tracking_space_id="space",
        study_title="Custom title",
        study_introduction="Custom introduction",
        evaluation_note="Custom evaluation note",
        root_scope="Custom scope.",
    )
    assert "Custom scope." in completed_documents["README.md"]
    assert "Completed ablation study" in completed_documents["README.md"]
