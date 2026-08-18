"""Public, credential-free reporting for the worldwide V2 baseline."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping

from .config import SOURCE_DATASET_ID, TARGET_MODEL_REPOSITORY_ID
from .dataset_contract import WORLDWIDE_PLACE_RELEVANCE_V2_CONTRACT

STUDY_ID = "place-relevance-v2"
_SENSITIVE_KEY_PARTS = ("credential", "password", "secret", "token")

# Read-only audit of the pinned V2 Parquet artifact. The audit was performed
# with the same clean iterator rules used by training and is recorded here so
# the public study documents state the exact protocol budget.
_DATA_AUDIT = {
    "raw_rows": 200_000,
    "raw_polygons": 10_268,
    "regions": 364,
    "languages": 317,
    "sources": 2,
    "raw_label_counts": {"no": 79_280, "yes": 120_720},
    "sentence_content_hash_groups": 176_193,
    "contradictory_hash_groups": 735,
    "clean_rows": 175_458,
    "clean_rows_by_split": {
        "train": 141_283,
        "validation": 17_619,
        "test": 16_556,
    },
    "clean_label_counts_by_split": {
        "train": {"no": 61_202, "yes": 80_081},
        "validation": {"no": 7_255, "yes": 10_364},
        "test": {"no": 6_946, "yes": 9_610},
    },
    "clean_polygons_by_split": {"train": 7_658, "validation": 967, "test": 913},
    "steps_per_epoch_at_batch_size_8": 17_661,
}


def _safe_scalar(value: object) -> bool:
    return (
        value is None
        or isinstance(value, (bool, int, str))
        or (isinstance(value, float) and math.isfinite(value))
    )


def _safe_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return {
        key: item
        for key, item in sorted(value.items())
        if isinstance(key, str)
        and not any(part in key.casefold() for part in _SENSITIVE_KEY_PARTS)
        and _safe_scalar(item)
    }


def _safe_text(value: object, fallback: str) -> str:
    if (
        isinstance(value, str)
        and value.strip()
        and "\n" not in value
        and "\r" not in value
    ):
        return value.strip()
    return fallback


def _json(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def render_place_relevance_study_documents(
    *,
    identity: Mapping[str, object],
    metrics: Mapping[str, object] | None,
    trackio_space_id: str | None = None,
) -> dict[str, str]:
    """Render the V2 protocol, results, and concise public study guide."""

    config = _safe_mapping(identity.get("training_config"))
    task_name = _safe_text(identity.get("task_name"), STUDY_ID)
    run_id = _safe_text(identity.get("run_id"), "not recorded")
    model_name = _safe_text(identity.get("model_name_or_path"), "not recorded")
    model_revision = _safe_text(identity.get("model_revision"), "not pinned")
    dataset_revision = _safe_text(
        identity.get("dataset_revision"),
        WORLDWIDE_PLACE_RELEVANCE_V2_CONTRACT.provenance.repository_revision,
    )
    trackio = _safe_text(trackio_space_id, "not enabled")
    study = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "task_name": task_name,
        "dataset": {
            "id": SOURCE_DATASET_ID,
            "config": WORLDWIDE_PLACE_RELEVANCE_V2_CONTRACT.config,
            "revision": dataset_revision,
            "parquet_sha256": WORLDWIDE_PLACE_RELEVANCE_V2_CONTRACT.provenance.parquet_sha256,
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
            "validation_fraction": config.get("validation_fraction", 0.1),
            "test_fraction": config.get("test_fraction", 0.1),
            "seed": config.get("seed", 42),
        },
        "data_audit": _DATA_AUDIT,
        "evaluation": {
            "validation": "at the end of each logical streamed epoch",
            "test": "once after training; never used for model selection",
            "metrics": ["accuracy", "balanced_accuracy", "precision", "recall", "f1"],
        },
        "model": {
            "name": model_name,
            "revision": model_revision,
            "trainable_layers": config.get("trainable_layers", "head"),
        },
        "tracking": {
            "project": config.get("tracking_project", STUDY_ID),
            "space": trackio,
        },
        "publication": {"repository": TARGET_MODEL_REPOSITORY_ID},
    }
    results = {
        "schema_version": 1,
        "study_id": STUDY_ID,
        "run_id": run_id,
        "dataset_revision": dataset_revision,
        "model_revision": model_revision,
        "metrics": _safe_mapping(metrics),
    }
    readme = (
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
        f"The exact dataset revision is `{dataset_revision}` and the model revision "
        f"is `{model_revision}`. The run identity is `{run_id}`.\n\n"
        "- Protocol: [`study.json`](study.json)\n"
        "- Data audit: [`data-audit.json`](data-audit.json)\n"
        "- Results: [`results.json`](results.json)\n"
        f"- Model repository: [{TARGET_MODEL_REPOSITORY_ID}]"
        f"(https://huggingface.co/{TARGET_MODEL_REPOSITORY_ID})\n"
        f"- Trackio: `{trackio}`\n"
    )
    return {
        "studies/place-relevance-v2/README.md": readme,
        "studies/place-relevance-v2/data-audit.json": _json(_DATA_AUDIT),
        "studies/place-relevance-v2/results.json": _json(results),
        "studies/place-relevance-v2/study.json": _json(study),
    }


__all__ = ["STUDY_ID", "render_place_relevance_study_documents"]
