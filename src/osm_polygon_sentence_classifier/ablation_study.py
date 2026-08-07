"""Reproducible planning and identity for the landuse ablation study."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from .config import PROJECT_NAME, TARGET_MODEL_REPOSITORY_ID
from .dataset_contract import LANDUSE_DATASET_CONTRACT
from .grid5000 import Grid5000RunIdentity
from .grid5000_autonomous import (
    DEFAULT_AUTONOMOUS_WALLTIME_SECONDS,
    AutonomousRunConfig,
    AutonomousRunController,
)
from .grid5000_sites import DEFAULT_SITES, SiteRequirements
from .grid5000_state import RunPhase
from .tracking import TRACKIO_BUCKET_ID, TRACKIO_STATIC_SPACE_ID
from .training import (
    DEFAULT_MODEL_NAME,
    ClassWeightMode,
    TrainableLayers,
    TrainingConfig,
    _training_config_payload,
)

ABLATION_STUDY_ID = "landuse-v1"
ABLATION_TRACKING_PROJECT = PROJECT_NAME
DEFAULT_MODEL_REVISION = "abc32620dd4f6ab06f5fbe905dc25f310618e09f"
PolicyType = Literal["auto", "day", "night"]
_CONTROL_ID = "a00-baseline-head-256-lr3e-4"
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_STUDY_STATE_SUBDIRECTORY = Path("grid5000/ablation-studies")


class AblationStudyError(ValueError):
    """Raised when the immutable ablation study plan is invalid."""


@dataclass(frozen=True, slots=True)
class AblationDefinition:
    """One controlled change from the frozen-head baseline."""

    ablation_id: str
    label: str
    max_length: int = 256
    learning_rate: float = 3e-4
    trainable_layers: TrainableLayers = "head"
    class_weight_mode: ClassWeightMode = "none"


@dataclass(frozen=True, slots=True)
class AblationRun:
    """One ablation definition and seed to execute."""

    definition: AblationDefinition
    seed: int

    @property
    def ablation_id(self) -> str:
        return self.definition.ablation_id


class AblationStudyStateStore:
    """Atomic study-level state beneath the approved external data root."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root) if root is not None else _default_state_root()

    def _path(self) -> Path:
        return self.root / ABLATION_STUDY_ID / "state.json"

    def load(self) -> dict[str, object] | None:
        path = self._path()
        _reject_symlinked_path(path)
        if not path.exists():
            return None
        if path.is_symlink():
            raise AblationStudyError("ablation study state cannot be a symlink")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise AblationStudyError("ablation study state cannot be read") from error
        if not isinstance(payload, Mapping):
            raise AblationStudyError("ablation study state must be a JSON object")
        return dict(payload)

    def save(self, payload: Mapping[str, object]) -> None:
        path = self._path()
        temporary = path.with_name(".state.json.tmp")
        _reject_symlinked_path(path)
        if path.is_symlink() or temporary.is_symlink():
            raise AblationStudyError("ablation study state cannot be a symlink")
        try:
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(path.parent, 0o700)
            _reject_symlinked_path(path.parent)
            if temporary.is_symlink():
                raise AblationStudyError("ablation study state cannot be a symlink")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.chmod(temporary, 0o600)
            os.replace(temporary, path)
            os.chmod(path, 0o600)
        except (OSError, TypeError, ValueError) as error:
            temporary.unlink(missing_ok=True)
            raise AblationStudyError("ablation study state cannot be saved") from error


def _default_state_root() -> Path:
    from .config import APPROVED_DATA_ROOT

    return APPROVED_DATA_ROOT / _STUDY_STATE_SUBDIRECTORY


def _reject_symlinked_path(path: Path) -> None:
    current = Path(path.anchor) if path.is_absolute() else Path()
    parts = path.parts[1:] if path.is_absolute() else path.parts
    for part in parts:
        current /= part
        if current.is_symlink():
            raise AblationStudyError(
                "ablation study state path cannot contain symlinks"
            )


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise AblationStudyError(
            "ablation study specification is not JSON-safe"
        ) from error


def _definition_payload(definition: AblationDefinition) -> dict[str, object]:
    return {
        "ablation_id": definition.ablation_id,
        "class_weight_mode": definition.class_weight_mode,
        "label": definition.label,
        "learning_rate": definition.learning_rate,
        "max_length": definition.max_length,
        "trainable_layers": definition.trainable_layers,
    }


def study_specification(
    *, source_commit: str, model_revision: str
) -> dict[str, object]:
    """Return the immutable study inputs recorded in public artifacts."""

    return {
        "study_id": ABLATION_STUDY_ID,
        "task_name": "landuse",
        "dataset_id": LANDUSE_DATASET_CONTRACT.dataset_id,
        "dataset_revision": LANDUSE_DATASET_CONTRACT.provenance.repository_revision,
        "model_revision": model_revision,
        "source_commit": source_commit,
        "screening_seed": 42,
        "replication_seeds": [43, 44],
        "selection_metric": "eval_f1",
        "tie_break_metric": "eval_macro_f1",
        "definitions": [
            _definition_payload(definition)
            for definition in baseline_ablation_definitions()
        ],
    }


def study_specification_fingerprint(specification: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(specification).encode("utf-8")).hexdigest()


def _metric_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return {
        key: item
        for key, item in value.items()
        if isinstance(key, str)
        and isinstance(item, (bool, int, float, str))
        and not isinstance(item, bool)
    }


class AblationStudyController:
    """Run the fixed study sequentially through the guarded run controller."""

    def __init__(
        self,
        *,
        source_commit: str,
        model_revision: str,
        model_name_or_path: str = DEFAULT_MODEL_NAME,
        sites: tuple[str, ...] = DEFAULT_SITES,
        gpu_memory_mb: int = 8_000,
        walltime_seconds: int = DEFAULT_AUTONOMOUS_WALLTIME_SECONDS,
        policy_type: PolicyType = "auto",
        max_workers: int = 4,
        max_continuations: int = 6,
        cleanup: bool = True,
        allow_source_commit_update: bool = False,
        state_root: Path | None = None,
        run_controller_factory: Callable[..., Any] = AutonomousRunController,
        publish_report: Callable[[Mapping[str, object]], None] | None = None,
        emit: Callable[[str], None] | None = None,
    ) -> None:
        if _REVISION_PATTERN.fullmatch(source_commit) is None:
            raise AblationStudyError("source_commit must be a pinned revision")
        if _REVISION_PATTERN.fullmatch(model_revision) is None:
            raise AblationStudyError("model_revision must be a pinned revision")
        if not sites:
            raise AblationStudyError("at least one Grid'5000 site is required")
        if gpu_memory_mb <= 0 or walltime_seconds <= 0:
            raise AblationStudyError("GPU memory and walltime must be positive")
        if max_workers <= 0 or max_continuations <= 0:
            raise AblationStudyError("worker and continuation limits must be positive")
        if not isinstance(allow_source_commit_update, bool):
            raise AblationStudyError(
                "source commit update permission must be a boolean"
            )
        if policy_type not in {"auto", "day", "night"}:
            raise AblationStudyError("policy_type must be auto, day, or night")
        self.source_commit = source_commit
        self.model_revision = model_revision
        self.model_name_or_path = model_name_or_path
        self.sites = sites
        self.gpu_memory_mb = gpu_memory_mb
        self.walltime_seconds = walltime_seconds
        self.policy_type = policy_type
        self.max_workers = max_workers
        self.max_continuations = max_continuations
        self.cleanup = cleanup
        self.allow_source_commit_update = allow_source_commit_update
        self.state = AblationStudyStateStore(state_root)
        self.run_state_root = self.state.root / "runs"
        self.run_controller_factory = run_controller_factory
        self.publish_report = publish_report or (lambda _state: None)
        self.emit = emit or (lambda _message: None)
        self.specification = study_specification(
            source_commit=source_commit,
            model_revision=model_revision,
        )
        self.fingerprint = study_specification_fingerprint(self.specification)

    def plan(self) -> dict[str, object]:
        """Return the next side-effect-free study plan."""

        state = self.state.load()
        if state is not None:
            self._validate_state(state)
        screening_results = self._screening_results(state or {})
        runs = planned_ablation_runs(screening_results=screening_results or None)
        records = self._records(state or {})
        pending = [
            {
                "ablation_id": run.ablation_id,
                "seed": run.seed,
                "status": records.get(self._run_key(run), {}).get("phase", "pending"),
            }
            for run in runs
            if records.get(self._run_key(run), {}).get("phase") != "completed"
        ]
        return {
            "study_id": ABLATION_STUDY_ID,
            "fingerprint": self.fingerprint,
            "model_repository_id": TARGET_MODEL_REPOSITORY_ID,
            "trackio_space_id": TRACKIO_STATIC_SPACE_ID,
            "trackio_bucket_id": TRACKIO_BUCKET_ID,
            "specification": self.specification,
            "next_runs": pending,
            "total_runs": len(runs),
        }

    def _new_state(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "study_id": ABLATION_STUDY_ID,
            "fingerprint": self.fingerprint,
            "specification": self.specification,
            "phase": "running",
            "runs": {},
        }

    def _validate_state(self, state: Mapping[str, object]) -> None:
        state_mismatch = (
            state.get("schema_version") != 1
            or state.get("study_id") != ABLATION_STUDY_ID
            or state.get("fingerprint") != self.fingerprint
        )
        if state_mismatch:
            if self.allow_source_commit_update and self._can_adopt_source_commit(state):
                return
            raise AblationStudyError(
                "existing ablation state does not match the immutable study specification"
            )
        if not isinstance(state.get("runs", {}), Mapping):
            raise AblationStudyError("ablation study run state is invalid")

    def _can_adopt_source_commit(self, state: Mapping[str, object]) -> bool:
        if (
            state.get("schema_version") != 1
            or state.get("study_id") != ABLATION_STUDY_ID
        ):
            return False
        raw_specification = state.get("specification")
        if not isinstance(raw_specification, Mapping):
            return False
        specification = cast(Mapping[str, object], raw_specification)
        old_source_commit = specification.get("source_commit")
        if (
            not isinstance(old_source_commit, str)
            or old_source_commit == self.source_commit
            or state.get("phase") == "completed"
        ):
            return False
        if study_specification_fingerprint(specification) != state.get("fingerprint"):
            return False
        current_without_source = {
            key: value
            for key, value in self.specification.items()
            if key != "source_commit"
        }
        stored_without_source = {
            key: value for key, value in specification.items() if key != "source_commit"
        }
        if current_without_source != stored_without_source:
            return False
        return not any(
            record.get("phase") == "running" for record in self._records(state).values()
        )

    def _adopt_source_commit_if_needed(
        self, state: dict[str, object]
    ) -> dict[str, object]:
        if not self.allow_source_commit_update or not self._can_adopt_source_commit(
            state
        ):
            return state
        specification = state.get("specification")
        if not isinstance(specification, Mapping):
            raise AblationStudyError("ablation study specification is invalid")
        old_source_commit = specification.get("source_commit")
        if not isinstance(old_source_commit, str):
            raise AblationStudyError("ablation study source commit is invalid")
        raw_history = state.get("source_commit_history", [])
        history: list[str] = (
            [item for item in raw_history if isinstance(item, str)]
            if isinstance(raw_history, list)
            else []
        )
        if old_source_commit not in history:
            history.append(old_source_commit)
        records = self._records(state)
        for record in records.values():
            record.setdefault("source_commit", old_source_commit)
        state["runs"] = records
        state["source_commit_history"] = history
        state["specification"] = self.specification
        state["fingerprint"] = self.fingerprint
        self.state.save(state)
        return state

    @staticmethod
    def _records(state: Mapping[str, object]) -> dict[str, dict[str, object]]:
        records = state.get("runs", {})
        if not isinstance(records, Mapping):
            raise AblationStudyError("ablation study run state is invalid")
        parsed: dict[str, dict[str, object]] = {}
        for key, value in records.items():
            if not isinstance(key, str) or not isinstance(value, Mapping):
                raise AblationStudyError("ablation study run state is invalid")
            parsed[key] = dict(cast(Mapping[str, object], value))
        return parsed

    def _screening_results(
        self,
        state: Mapping[str, object],
    ) -> dict[str, Mapping[str, object]]:
        results: dict[str, Mapping[str, object]] = {}
        for record in self._records(state).values():
            if record.get("phase") != "completed" or record.get("seed") != 42:
                continue
            ablation_id = record.get("ablation_id")
            metrics = record.get("metrics")
            if isinstance(ablation_id, str) and isinstance(metrics, Mapping):
                results[ablation_id] = cast(Mapping[str, object], metrics)
        return results

    @staticmethod
    def _run_key(run: AblationRun) -> str:
        return f"{run.ablation_id}|seed-{run.seed}"

    def _run_config(
        self, run: AblationRun
    ) -> tuple[TrainingConfig, Grid5000RunIdentity]:
        config = build_ablation_training_config(
            run.definition,
            seed=run.seed,
            model_revision=self.model_revision,
            model_name_or_path=self.model_name_or_path,
        )
        identity = Grid5000RunIdentity(
            source_commit=self.source_commit,
            dataset_revision=LANDUSE_DATASET_CONTRACT.provenance.repository_revision,
            model_name_or_path=config.model_name_or_path,
            model_revision=self.model_revision,
            training_config=_training_config_payload(config),
        )
        return config, identity

    def _autonomous_config(
        self,
        run: AblationRun,
    ) -> AutonomousRunConfig:
        config, identity = self._run_config(run)
        return AutonomousRunConfig(
            identity=identity,
            training_config=config,
            sites=self.sites,
            requirements=SiteRequirements(gpu_memory_mb=self.gpu_memory_mb),
            walltime_seconds=self.walltime_seconds,
            policy_type=self.policy_type,
            max_workers=self.max_workers,
            max_continuations=self.max_continuations,
            cleanup=self.cleanup,
        )

    def run(self) -> dict[str, object]:
        """Run or resume all planned ablations until the study is complete."""

        state = self.state.load()
        if state is None:
            state = self._new_state()
            self.state.save(state)
        else:
            self._validate_state(state)
            state = self._adopt_source_commit_if_needed(state)
        if state.get("phase") == "completed":
            return state

        while True:
            screening_results = self._screening_results(state)
            runs = planned_ablation_runs(screening_results=screening_results or None)
            records = self._records(state)
            pending = next(
                (
                    run
                    for run in runs
                    if records.get(self._run_key(run), {}).get("phase") != "completed"
                ),
                None,
            )
            if pending is None:
                state["phase"] = "completed"
                self.state.save(state)
                self.publish_report(state)
                self.emit(f"study {ABLATION_STUDY_ID}: completed")
                return state

            key = self._run_key(pending)
            config = self._autonomous_config(pending)
            records[key] = {
                "ablation_id": pending.ablation_id,
                "seed": pending.seed,
                "source_commit": self.source_commit,
                "run_id": config.identity.run_id,
                "phase": "running",
            }
            state["runs"] = records
            self.state.save(state)
            self.emit(
                f"study {ABLATION_STUDY_ID}: running {pending.ablation_id} "
                f"seed={pending.seed} run={config.identity.run_id}"
            )
            try:
                run_controller = self.run_controller_factory(
                    config,
                    state_root=self.run_state_root,
                    emit=self.emit,
                )
                run_state = run_controller.run()
            except Exception as error:
                records[key].update({"phase": "failed", "error": str(error)})
                state["runs"] = records
                self.state.save(state)
                raise AblationStudyError(
                    f"ablation {pending.ablation_id} seed {pending.seed} failed"
                ) from error
            if getattr(run_state, "phase", None) is not RunPhase.COMPLETED:
                phase = str(getattr(run_state, "phase", "unknown"))
                records[key].update({"phase": phase})
                state["runs"] = records
                self.state.save(state)
                raise AblationStudyError(
                    f"ablation {pending.ablation_id} seed {pending.seed} ended in {phase}"
                )
            facts = getattr(run_state, "facts", {})
            completion = (
                facts.get("completion", {}) if isinstance(facts, Mapping) else {}
            )
            metrics = (
                _metric_mapping(completion.get("metrics"))
                if isinstance(completion, Mapping)
                else {}
            )
            records[key].update({"phase": "completed", "metrics": metrics})
            state["runs"] = records
            self.state.save(state)
            self.publish_report(state)


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


def _report_runs(state: Mapping[str, object]) -> list[dict[str, object]]:
    records = AblationStudyController._records(state)
    specification = state.get("specification", {})
    if not isinstance(specification, Mapping):
        raise AblationStudyError("ablation study specification is invalid")
    screening = {
        str(record.get("ablation_id")): cast(
            Mapping[str, object], record.get("metrics")
        )
        for record in records.values()
        if record.get("seed") == 42
        and record.get("phase") == "completed"
        and isinstance(record.get("ablation_id"), str)
        and isinstance(record.get("metrics"), Mapping)
    }
    planned = planned_ablation_runs(screening_results=screening or None)
    result_rows: list[dict[str, object]] = []
    for run in planned:
        record = records.get(f"{run.ablation_id}|seed-{run.seed}", {})
        metrics = record.get("metrics", {})
        result_rows.append(
            {
                "ablation_id": run.ablation_id,
                "seed": run.seed,
                "status": record.get("phase", "pending"),
                "run_id": record.get("run_id"),
                "source_commit": record.get("source_commit"),
                "metrics": _metric_mapping(metrics),
                "model_path": (
                    f"studies/{ABLATION_STUDY_ID}/{run.ablation_id}/"
                    f"run-{record['run_id']}/final/"
                    if isinstance(record.get("run_id"), str)
                    and record.get("phase") == "completed"
                    else None
                ),
            }
        )
    return result_rows


def render_study_documents(state: Mapping[str, object]) -> dict[str, str]:
    """Render the small set of public study documents."""

    specification = state.get("specification")
    fingerprint = state.get("fingerprint")
    if not isinstance(specification, Mapping) or not isinstance(fingerprint, str):
        raise AblationStudyError("ablation study state lacks its public specification")
    rows = _report_runs(state)
    status = str(state.get("phase", "running"))
    study_status_label = "Completed" if status == "completed" else "In-progress"
    source_commit = specification.get("source_commit", "not recorded")
    model_revision = specification.get("model_revision", "not recorded")
    dataset_revision = specification.get("dataset_revision", "not recorded")
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
        run_name = f"{ABLATION_STUDY_ID}|{row['ablation_id']}|seed-{row['seed']}"
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
        "study_id": ABLATION_STUDY_ID,
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
        f"# Landuse classifier ablation study `{ABLATION_STUDY_ID}`\n\n"
        "This study measures controlled changes to the landuse sentence classifier. "
        "The existing single-run baseline remains under `experiments/`; this study "
        "uses the `studies/landuse-v1/` namespace.\n\n"
        "## Protocol\n\n"
        "- Seven one-factor screening runs use seed `42`.\n"
        "- The baseline and the two highest positive-class F1 variants are replicated "
        "with seeds `43` and `44`.\n"
        "- Selection metric: positive-class F1 (`eval_f1`). Tie-break: macro-F1.\n"
        "- The polygon split, cleaned input, dataset revision, model revision, and "
        "training budget are fixed across runs.\n"
        "- Results are validation results; this study has no held-out test set.\n\n"
        "## Provenance\n\n"
        f"- Dataset revision: `{dataset_revision}`\n"
        f"- Model revision: `{model_revision}`\n"
        f"- Source commit: `{source_commit}`\n"
        f"- Study specification SHA-256: `{fingerprint}`\n"
        f"- Trackio: [public dashboard](https://huggingface.co/spaces/"
        f"{TRACKIO_STATIC_SPACE_ID}){source_history_note}\n\n"
        "## How to read a run\n\n"
        "- A public Trackio run name is `landuse-v1|<ablation-id>|seed-<seed>`; "
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
        "landuse sentence-classification task. The repository root is "
        "documentation-only; model files live inside immutable experiment runs.\n\n"
        "## Start here\n\n"
        f"- {study_status_label} ablation study: [`studies/{ABLATION_STUDY_ID}/README.md`]"
        f"(studies/{ABLATION_STUDY_ID}/README.md)\n"
        f"- Machine-readable protocol: [`studies/{ABLATION_STUDY_ID}/study.json`]"
        f"(studies/{ABLATION_STUDY_ID}/study.json)\n"
        f"- Machine-readable metrics: [`studies/{ABLATION_STUDY_ID}/results.json`]"
        f"(studies/{ABLATION_STUDY_ID}/results.json)\n"
        f"- Static metrics: [Trackio dashboard](https://huggingface.co/spaces/"
        f"{TRACKIO_STATIC_SPACE_ID})\n\n"
        "## Artifact layout\n\n"
        "- Single-run outputs: `experiments/<experiment>/run-<run-id>/`.\n"
        f"- Ablation outputs: `studies/{ABLATION_STUDY_ID}/<ablation-id>/run-<run-id>/`.\n"
        "- Final model: the run's `final/` directory.\n"
        "- Resumable outputs: the same run's `checkpoints/step-N/` directories.\n\n"
        "The run registry explains how the study name, seed, controller run ID, "
        "checkpoint step, and final artifact path relate to one another.\n"
    )
    return {
        "README.md": root_readme,
        f"studies/{ABLATION_STUDY_ID}/README.md": study_readme,
        f"studies/{ABLATION_STUDY_ID}/study.json": (
            json.dumps(study_payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ),
        f"studies/{ABLATION_STUDY_ID}/results.json": (
            json.dumps(results_payload, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n"
        ),
    }


def publish_study_report(
    state: Mapping[str, object],
    *,
    hub_api: Any | None = None,
) -> None:
    """Publish the current study manifest and report to the existing model repo."""

    from .publication import ModelPublicationError, publish_study_documents

    try:
        publish_study_documents(
            TARGET_MODEL_REPOSITORY_ID,
            render_study_documents(state),
            hub_api=hub_api,
        )
    except ModelPublicationError as error:
        raise AblationStudyError("ablation study report publication failed") from error


def baseline_ablation_definitions() -> tuple[AblationDefinition, ...]:
    """Return the fixed one-factor screening matrix in execution order."""

    return (
        AblationDefinition(
            _CONTROL_ID,
            "Frozen encoder, 256 tokens, learning rate 3e-4",
        ),
        AblationDefinition(
            "a01-head-128", "Frozen encoder, 128 tokens", max_length=128
        ),
        AblationDefinition(
            "a02-head-512", "Frozen encoder, 512 tokens", max_length=512
        ),
        AblationDefinition(
            "a03-head-lr1e-4",
            "Frozen encoder, learning rate 1e-4",
            learning_rate=1e-4,
        ),
        AblationDefinition(
            "a04-head-lr1e-3",
            "Frozen encoder, learning rate 1e-3",
            learning_rate=1e-3,
        ),
        AblationDefinition(
            "a05-balanced-head",
            "Frozen encoder, class-balanced loss",
            class_weight_mode="balanced",
        ),
        AblationDefinition(
            "a06-last2-256",
            "Unfreeze the last two encoder layers",
            trainable_layers="last2",
        ),
    )


def build_ablation_training_config(
    definition: AblationDefinition,
    *,
    seed: int,
    model_revision: str,
    model_name_or_path: str = DEFAULT_MODEL_NAME,
    publish_to_hub: bool = True,
    sync_trackio: bool = True,
) -> TrainingConfig:
    """Create the fully explicit training configuration for one study run."""

    namespace = f"studies/{ABLATION_STUDY_ID}/{definition.ablation_id}"
    return TrainingConfig(
        model_name_or_path=model_name_or_path,
        output_subdirectory=Path(namespace) / "models/landuse",
        seed=seed,
        max_length=definition.max_length,
        learning_rate=definition.learning_rate,
        run_name=f"{ABLATION_STUDY_ID}|{definition.ablation_id}|seed-{seed}",
        model_revision=model_revision,
        trainable_layers=definition.trainable_layers,
        class_weight_mode=definition.class_weight_mode,
        tracking_project=ABLATION_TRACKING_PROJECT,
        artifact_namespace=namespace,
        publish_to_hub=publish_to_hub,
        sync_trackio=sync_trackio,
    )


def rank_screening_results(
    results: Mapping[str, Mapping[str, object]],
) -> list[str]:
    """Rank completed screening runs by positive F1, then macro-F1."""

    ranked: list[tuple[str, float, float]] = []
    for ablation_id, metrics in results.items():
        positive_f1 = metrics.get("eval_f1")
        macro_f1 = metrics.get("eval_macro_f1", 0.0)
        if not isinstance(positive_f1, (int, float)) or isinstance(positive_f1, bool):
            raise AblationStudyError(
                f"screening result for {ablation_id} lacks numeric eval_f1"
            )
        if not isinstance(macro_f1, (int, float)) or isinstance(macro_f1, bool):
            raise AblationStudyError(
                f"screening result for {ablation_id} has invalid eval_macro_f1"
            )
        ranked.append((ablation_id, float(positive_f1), float(macro_f1)))
    ranked.sort(key=lambda item: (-item[1], -item[2], item[0]))
    return [item[0] for item in ranked]


def planned_ablation_runs(
    *,
    screening_results: Mapping[str, Mapping[str, object]] | None = None,
) -> list[AblationRun]:
    """Return screening runs, or add seed replications after screening."""

    definitions = baseline_ablation_definitions()
    by_id = {definition.ablation_id: definition for definition in definitions}
    runs = [AblationRun(definition, 42) for definition in definitions]
    if screening_results is None or not all(
        definition.ablation_id in screening_results for definition in definitions
    ):
        return runs
    ranked_non_control = [
        ablation_id
        for ablation_id in rank_screening_results(screening_results)
        if ablation_id != _CONTROL_ID
    ]
    if len(ranked_non_control) < 2:
        raise AblationStudyError("screening requires two non-control finalists")
    finalists = [_CONTROL_ID, *ranked_non_control[:2]]
    runs.extend(
        AblationRun(by_id[ablation_id], seed)
        for ablation_id in finalists
        for seed in (43, 44)
    )
    return runs


__all__ = [
    "ABLATION_STUDY_ID",
    "ABLATION_TRACKING_PROJECT",
    "AblationDefinition",
    "AblationRun",
    "AblationStudyController",
    "AblationStudyError",
    "AblationStudyStateStore",
    "DEFAULT_MODEL_REVISION",
    "baseline_ablation_definitions",
    "build_ablation_training_config",
    "planned_ablation_runs",
    "publish_study_report",
    "rank_screening_results",
    "render_study_documents",
    "study_specification",
    "study_specification_fingerprint",
]
