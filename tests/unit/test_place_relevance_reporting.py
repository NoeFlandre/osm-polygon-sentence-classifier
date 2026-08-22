import json

import pytest

from osm_polygon_sentence_classifier import place_relevance_reporting as reporting
from osm_polygon_sentence_classifier.config import (
    SOURCE_DATASET_ID,
    TARGET_MODEL_REPOSITORY_ID,
)
from osm_polygon_sentence_classifier.dataset_contract import (
    WORLDWIDE_PLACE_RELEVANCE_V2_CONTRACT,
)
from osm_polygon_sentence_classifier.place_relevance_reporting import (
    render_place_relevance_study_documents,
)


@pytest.mark.parametrize(
    "value",
    [None, True, False, 0, 3, 1.5, "text"],
)
def test_safe_scalar_accepts_public_json_scalars(value: object) -> None:
    assert reporting._safe_scalar(value) is True


@pytest.mark.parametrize(
    "value",
    [float("nan"), float("inf"), [], {}, object()],
)
def test_safe_scalar_rejects_non_finite_and_non_scalar_values(value: object) -> None:
    assert reporting._safe_scalar(value) is False


def test_safe_mapping_item_drops_secrets_and_unsupported_values() -> None:
    assert reporting._safe_mapping_item("token", "hidden") is None
    assert reporting._safe_mapping_item("nested", {"value": 1}) is None
    assert reporting._safe_mapping_item(42, "value") is None
    assert reporting._safe_mapping_item("nullable", None) == ("nullable", None)


@pytest.mark.parametrize(
    ("value", "fallback", "expected"),
    [
        ("  text  ", "fallback", "text"),
        (None, "fallback", "fallback"),
        ("", "fallback", "fallback"),
        ("line\nbreak", "fallback", "fallback"),
        ("line\rbreak", "fallback", "fallback"),
        (42, "fallback", "fallback"),
    ],
)
def test_safe_text_only_returns_single_line_non_empty_strings(
    value: object,
    fallback: str,
    expected: str,
) -> None:
    assert reporting._safe_text(value, fallback) == expected


def test_place_relevance_report_is_public_and_reproducible() -> None:
    documents = render_place_relevance_study_documents(
        identity={
            "run_id": "a" * 20,
            "task_name": "place-relevance-v2",
            "dataset_revision": "b" * 40,
            "model_name_or_path": "jhu-clsp/mmBERT-small",
            "model_revision": "c" * 40,
            "training_config": {
                "validation_fraction": 0.1,
                "test_fraction": 0.1,
                "seed": 42,
                "trainable_layers": "head",
                "HF_TOKEN": "must-not-appear",
            },
        },
        metrics={"eval_accuracy": 0.8, "test_macro_f1": 0.7},
        trackio_space_id="NoeFlandre/osm-polygon-sentence-classifier-trackio",
    )

    assert set(documents) == {
        "studies/place-relevance-v2/README.md",
        "studies/place-relevance-v2/data-audit.json",
        "studies/place-relevance-v2/results.json",
        "studies/place-relevance-v2/study.json",
    }
    assert "Worldwide V2" in documents["studies/place-relevance-v2/README.md"]
    assert '"test_fraction": 0.1' in documents["studies/place-relevance-v2/study.json"]
    assert (
        '"clean_rows": 175458'
        in documents["studies/place-relevance-v2/data-audit.json"]
    )
    assert "data-audit.json" in documents["studies/place-relevance-v2/README.md"]
    assert (
        '"test_macro_f1": 0.7' in documents["studies/place-relevance-v2/results.json"]
    )
    assert "HF_TOKEN" not in "".join(documents.values())


def test_place_relevance_report_preserves_the_complete_public_schema() -> None:
    documents = render_place_relevance_study_documents(
        identity={
            "run_id": "run-custom",
            "task_name": "custom-task",
            "dataset_revision": "dataset-custom",
            "model_name_or_path": "model-custom",
            "model_revision": "model-revision-custom",
            "training_config": {
                "validation_fraction": 0.2,
                "test_fraction": 0.3,
                "seed": 7,
                "trainable_layers": "encoder",
                "tracking_project": "custom-project",
            },
        },
        metrics={
            "accuracy": 0.8,
            "step": 10,
            "nullable": None,
            "token": "must-not-appear",
            "nested": {"value": 1},
            "nan": float("nan"),
        },
        trackio_space_id="custom-trackio-space",
    )

    study = json.loads(documents["studies/place-relevance-v2/study.json"])
    assert study == {
        "schema_version": 1,
        "study_id": "place-relevance-v2",
        "task_name": "custom-task",
        "dataset": {
            "id": SOURCE_DATASET_ID,
            "config": WORLDWIDE_PLACE_RELEVANCE_V2_CONTRACT.config,
            "revision": "dataset-custom",
            "parquet_sha256": (
                WORLDWIDE_PLACE_RELEVANCE_V2_CONTRACT.provenance.parquet_sha256
            ),
            "label_column": WORLDWIDE_PLACE_RELEVANCE_V2_CONTRACT.label_column,
            "labels": {"no": 0, "yes": 1},
        },
        "input": {
            "columns": list(WORLDWIDE_PLACE_RELEVANCE_V2_CONTRACT.model_input_columns),
            "text_column": "sentence_text_normalized",
            "metadata_columns_are_not_features": True,
        },
        "split": {
            "unit": "polygon_id",
            "method": "sha256(seed:polygon_id)",
            "validation_fraction": 0.2,
            "test_fraction": 0.3,
            "seed": 7,
        },
        "data_audit": reporting._DATA_AUDIT,
        "evaluation": {
            "validation": "at the end of each logical streamed epoch",
            "test": "once after training; never used for model selection",
            "metrics": [
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
            ],
        },
        "model": {
            "name": "model-custom",
            "revision": "model-revision-custom",
            "trainable_layers": "encoder",
        },
        "tracking": {
            "project": "custom-project",
            "space": "custom-trackio-space",
        },
        "publication": {"repository": TARGET_MODEL_REPOSITORY_ID},
    }

    results = json.loads(documents["studies/place-relevance-v2/results.json"])
    assert results == {
        "schema_version": 1,
        "study_id": "place-relevance-v2",
        "run_id": "run-custom",
        "dataset_revision": "dataset-custom",
        "model_revision": "model-revision-custom",
        "metrics": {"accuracy": 0.8, "nullable": None, "step": 10},
    }
    assert json.loads(documents["studies/place-relevance-v2/data-audit.json"]) == (
        reporting._DATA_AUDIT
    )


def test_place_relevance_report_uses_documented_defaults_and_fallbacks() -> None:
    documents = render_place_relevance_study_documents(
        identity={
            "task_name": "   ",
            "run_id": "\n",
            "model_name_or_path": "",
            "model_revision": "\r",
            "dataset_revision": "   ",
            "training_config": {},
        },
        metrics=None,
    )

    study = json.loads(documents["studies/place-relevance-v2/study.json"])
    assert study["task_name"] == "place-relevance-v2"
    assert study["split"] == {
        "unit": "polygon_id",
        "method": "sha256(seed:polygon_id)",
        "validation_fraction": 0.1,
        "test_fraction": 0.1,
        "seed": 42,
    }
    assert study["model"] == {
        "name": "not recorded",
        "revision": "not pinned",
        "trainable_layers": "head",
    }
    assert study["tracking"] == {
        "project": "place-relevance-v2",
        "space": "not enabled",
    }
    assert json.loads(documents["studies/place-relevance-v2/results.json"]) == {
        "schema_version": 1,
        "study_id": "place-relevance-v2",
        "run_id": "not recorded",
        "dataset_revision": (
            WORLDWIDE_PLACE_RELEVANCE_V2_CONTRACT.provenance.repository_revision
        ),
        "model_revision": "not pinned",
        "metrics": {},
    }


def test_place_relevance_readme_is_a_stable_public_artifact() -> None:
    documents = render_place_relevance_study_documents(
        identity={
            "run_id": "run-custom",
            "dataset_revision": "dataset-custom",
            "model_revision": "model-custom",
            "training_config": {},
        },
        metrics=None,
        trackio_space_id="trackio-custom",
    )

    assert documents["studies/place-relevance-v2/README.md"] == (
        "# Worldwide V2 place-relevance baseline\n\n"
        "This study trains a binary classifier for whether a sentence is relevant "
        "to a place. It is separate from the earlier Afghanistan landuse study.\n\n"
        "## Protocol\n\n"
        "- Input: `sentence_text_normalized` only. IDs, labels, probabilities, "
        "geometry, and other metadata are excluded.\n"
        "- Split: deterministic polygon-level train/validation/test partitions "
        "with seed 42.\n"
        "- Model: pinned `jhu-clsp/mmBERT-small`; the encoder is frozen and the "
        "classification head is trained.\n"
        "- Evaluation: validation metrics are recorded after each logical streamed "
        "epoch; the held-out test set is evaluated once at the end.\n\n"
        "## Data audit\n\n"
        "The pinned artifact has 200,000 raw rows. The clean boundary removes "
        "contradictory sentence-content-hash groups and duplicate representatives, "
        "leaving 175,458 rows: 141,283 train, 17,619 validation, and 16,556 test. "
        "The exact counts are in [`data-audit.json`](data-audit.json).\n\n"
        "## Reproduction\n\n"
        "The exact dataset revision is `dataset-custom` and the model revision "
        "is `model-custom`. The run identity is `run-custom`.\n\n"
        "- Protocol: [`study.json`](study.json)\n"
        "- Data audit: [`data-audit.json`](data-audit.json)\n"
        "- Results: [`results.json`](results.json)\n"
        f"- Model repository: [{TARGET_MODEL_REPOSITORY_ID}]"
        f"(https://huggingface.co/{TARGET_MODEL_REPOSITORY_ID})\n"
        "- Trackio: `trackio-custom`\n"
    )
