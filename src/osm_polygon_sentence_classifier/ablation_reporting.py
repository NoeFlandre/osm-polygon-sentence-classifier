"""Render public documents for the landuse ablation study."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import cast


def _format_metric(metrics: Mapping[str, object], name: str) -> str:
    value = metrics.get(name)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.4f}"
    return "—"


def _format_count(metrics: Mapping[str, object], name: str) -> str:
    value = metrics.get(name)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(int(value)) if float(value).is_integer() else f"{float(value):.1f}"
    return "—"


def render_public_documents(
    state: Mapping[str, object],
    *,
    rows: Sequence[Mapping[str, object]],
    study_id: str,
    tracking_space_id: str,
    study_title: str | None = None,
    study_introduction: str | None = None,
    evaluation_note: str | None = None,
    root_scope: str | None = None,
    include_root_readme: bool = True,
) -> dict[str, str]:
    """Render the public study README and machine-readable manifests."""

    specification = state.get("specification")
    fingerprint = state.get("fingerprint")
    if not isinstance(specification, Mapping) or not isinstance(fingerprint, str):
        raise ValueError("ablation study state lacks its public specification")
    status = str(state.get("phase", "running"))
    study_status_label = "Completed" if status == "completed" else "In-progress"
    source_commit = specification.get("source_commit", "not recorded")
    model_revision = specification.get("model_revision", "not recorded")
    dataset_revision = specification.get("dataset_revision", "not recorded")
    effective_title = study_title or "Landuse classifier ablation study"
    effective_introduction = study_introduction or (
        "This study measures controlled changes to the landuse sentence classifier."
    )
    effective_evaluation_note = evaluation_note or (
        "Results are validation results; this study has no held-out test set."
    )
    effective_root_scope = root_scope or "landuse sentence-classification task."
    definition_rows = [
        "| Ablation | Change | Maximum length | Learning rate | "
        "Trainable layers | Class weighting |",
        "|---|---|---:|---:|---|---|",
    ]
    raw_definitions = specification.get("definitions", [])
    if isinstance(raw_definitions, list):
        for raw_definition in raw_definitions:
            if not isinstance(raw_definition, Mapping):
                continue
            definition_rows.append(
                f"| `{raw_definition.get('ablation_id', '—')}` | "
                f"{raw_definition.get('label', '—')} | "
                f"`{raw_definition.get('max_length', '—')}` | "
                f"`{raw_definition.get('learning_rate', '—')}` | "
                f"`{raw_definition.get('trainable_layers', '—')}` | "
                f"`{raw_definition.get('class_weight_mode', '—')}` |"
            )
    table_rows = [
        "| Run name | Ablation | Seed | Status | Accuracy | Precision | "
        "Recall | Positive F1 | Macro F1 | Balanced accuracy | "
        "Validation support (no / yes) | Final artifact |",
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        metrics = cast(Mapping[str, object], row["metrics"])
        model_path = row.get("model_path")
        model_cell = f"`{model_path}`" if isinstance(model_path, str) else "—"
        run_name = f"{study_id}|{row['ablation_id']}|seed-{row['seed']}"
        support = (
            f"{_format_count(metrics, 'eval_negative_support')} / "
            f"{_format_count(metrics, 'eval_positive_support')}"
        )
        table_rows.append(
            f"| `{run_name}` | `{row['ablation_id']}` | {row['seed']} | "
            f"{row['status']} | "
            f"{_format_metric(metrics, 'eval_accuracy')} | "
            f"{_format_metric(metrics, 'eval_precision')} | "
            f"{_format_metric(metrics, 'eval_recall')} | "
            f"{_format_metric(metrics, 'eval_f1')} | "
            f"{_format_metric(metrics, 'eval_macro_f1')} | "
            f"{_format_metric(metrics, 'eval_balanced_accuracy')} | "
            f"{support} | {model_cell} |"
        )
    results_payload = {
        "study_id": study_id,
        "fingerprint": fingerprint,
        "phase": status,
        "runs": rows,
    }
    source_commit_history = state.get("source_commit_history", [])
    if not isinstance(source_commit_history, list) or any(
        not isinstance(item, str) for item in source_commit_history
    ):
        source_commit_history = []
    source_history_note = (
        "\n- Earlier source commits retained for historical run provenance: "
        + ", ".join(f"`{item}`" for item in source_commit_history)
        if source_commit_history
        else ""
    )
    study_payload = {
        **dict(specification),
        "fingerprint": fingerprint,
        "source_commit_history": source_commit_history,
    }
    study_readme = (
        f"# {effective_title} `{study_id}`\n\n"
        f"{effective_introduction} "
        "The existing single-run baseline remains under `experiments/`; this study "
        f"uses the `studies/{study_id}/` namespace.\n\n"
        "## Protocol\n\n"
        "- Seven one-factor screening runs use seed `42`.\n"
        "- The baseline and the two highest positive-class F1 variants are replicated "
        "with seeds `43` and `44`.\n"
        "- Selection metric: positive-class F1 (`eval_f1`). Tie-break: macro-F1.\n"
        "- The polygon split, cleaned input, dataset revision, model revision, and "
        "training budget are fixed across runs.\n"
        f"- {effective_evaluation_note}\n\n"
        "## Provenance\n\n"
        f"- Dataset revision: `{dataset_revision}`\n"
        f"- Model revision: `{model_revision}`\n"
        f"- Source commit: `{source_commit}`\n"
        f"- Study specification SHA-256: `{fingerprint}`\n"
        f"- Trackio: [public dashboard](https://huggingface.co/spaces/"
        f"{tracking_space_id}){source_history_note}\n\n"
        "## How to read a run\n\n"
        f"- A public Trackio run name is `{study_id}|<ablation-id>|seed-<seed>`; "
        "it identifies the study variant and replication seed.\n"
        "- `run-<run-id>` is the immutable controller run identity. It is not an "
        "OAR job ID; short continuation jobs for one run keep this same ID.\n"
        "- `checkpoints/step-N/` contains the complete checkpoint at Trainer step "
        "`N`; `final/` contains that run's terminal model.\n"
        "- `results.json` is the machine-readable source for all scalar metrics; "
        "`study.json` is the immutable protocol and provenance record.\n\n"
        "## Ablation definitions\n\n" + "\n".join(definition_rows) + "\n\n"
        "## Validation metrics and run registry\n\n"
        "The table below is the human-readable registry. `Positive F1` is "
        "`eval_f1`; support is shown as `no / yes`. These are validation results, "
        "not held-out test results.\n\n" + "\n".join(table_rows) + "\n\n"
        "Grid'5000 resources were used for the computation. Publications based on "
        "this study should include the official Grid’5000 acknowledgment.\n"
    )
    root_readme = (
        "---\n"
        "library_name: transformers\n"
        "pipeline_tag: text-classification\n"
        "tags:\n"
        "- landuse\n"
        "- text-classification\n"
        "---\n\n"
        "# OSM Polygon Sentence Classifier\n\n"
        "Public model artifacts and experiment registry for the OSM polygon "
        f"{effective_root_scope} The repository root is "
        "documentation-only; model files live inside immutable experiment runs.\n\n"
        "## Start here\n\n"
        f"- {study_status_label} ablation study: [`studies/{study_id}/README.md`]"
        f"(studies/{study_id}/README.md)\n"
        f"- Machine-readable protocol: [`studies/{study_id}/study.json`]"
        f"(studies/{study_id}/study.json)\n"
        f"- Machine-readable metrics: [`studies/{study_id}/results.json`]"
        f"(studies/{study_id}/results.json)\n"
        f"- Static metrics: [Trackio dashboard](https://huggingface.co/spaces/"
        f"{tracking_space_id})\n\n"
        "## Artifact layout\n\n"
        "- Single-run outputs: `experiments/<experiment>/run-<run-id>/`.\n"
        f"- Ablation outputs: `studies/{study_id}/<ablation-id>/run-<run-id>/`.\n"
        "- Final model: the run's `final/` directory.\n"
        "- Resumable outputs: the same run's `checkpoints/step-N/` directories.\n\n"
        "The run registry explains how the study name, seed, controller run ID, "
        "checkpoint step, and final artifact path relate to one another.\n"
    )
    documents = {
        f"studies/{study_id}/README.md": study_readme,
        f"studies/{study_id}/study.json": (
            json.dumps(study_payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ),
        f"studies/{study_id}/results.json": (
            json.dumps(results_payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ),
    }
    if include_root_readme:
        documents["README.md"] = root_readme
    return documents


__all__ = ["render_public_documents"]
