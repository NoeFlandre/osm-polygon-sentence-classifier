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

from .ablation_reporting import render_public_documents
from .config import PROJECT_NAME, TARGET_MODEL_REPOSITORY_ID
from .dataset_contract import (
    LANDUSE_DATASET_CONTRACT,
    WORLDWIDE_PLACE_RELEVANCE_V2_CONTRACT,
    DatasetContract,
)
from .grid5000 import (
    ContainerRuntime,
    Grid5000ConfigurationError,
    Grid5000RunIdentity,
    _validate_container_settings,
)
from .grid5000_autonomous import (
    DEFAULT_AUTONOMOUS_WALLTIME_SECONDS,
    AutonomousRunConfig,
    AutonomousRunController,
)
from .grid5000_sites import DEFAULT_SITES, SiteRequirements
from .grid5000_state import RunPhase
from .tracking import (
    TRACKIO_BUCKET_ID,
    TRACKIO_STATIC_SPACE_ID,
    V2_TRACKIO_STATIC_SPACE_ID,
)
from .training import (
    DEFAULT_MODEL_NAME,
    ClassWeightMode,
    TrainableLayers,
    TrainingConfig,
    _training_config_payload,
)

ABLATION_STUDY_ID = "landuse-v1"
ABLATION_TRACKING_PROJECT = PROJECT_NAME
PLACE_RELEVANCE_V2_ABLATION_STUDY_ID = "place-relevance-v2-ablations"
PLACE_RELEVANCE_V2_ABLATION_TRACKING_PROJECT = "place-relevance-v2-ablations"
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


@dataclass(frozen=True, slots=True)
class AblationStudyProtocol:
    """Immutable settings that define one comparable ablation study."""

    study_id: str
    task_name: str
    dataset_contract: DatasetContract
    tracking_project: str
    definitions: tuple[AblationDefinition, ...]
    screening_seed: int = 42
    replication_seeds: tuple[int, ...] = (43, 44)
    selection_metric: str = "eval_f1"
    tie_break_metric: str = "eval_macro_f1"
    validation_fraction: float = 0.2
    test_fraction: float = 0.0
    eval_strategy: Literal["steps", "epoch"] = "steps"
    max_steps: int = 1_000
    output_task: str = "landuse"
    title: str = "Landuse classifier ablation study"
    introduction: str = (
        "This study measures controlled changes to the landuse sentence classifier."
    )
    evaluation_note: str = (
        "Results are validation results; this study has no held-out test set."
    )

    def __post_init__(self) -> None:
        if not self.study_id or any(char in self.study_id for char in "/\\\n\r"):
            raise AblationStudyError("study_id must be a clean single-line name")
        if not self.task_name or not self.tracking_project:
            raise AblationStudyError("study task and tracking project are required")
        if not self.definitions:
            raise AblationStudyError("at least one ablation definition is required")
        if not self.replication_seeds:
            raise AblationStudyError("at least one replication seed is required")
        if self.validation_fraction < 0 or self.test_fraction < 0:
            raise AblationStudyError("split fractions must be non-negative")
        if self.validation_fraction + self.test_fraction > 1:
            raise AblationStudyError("split fractions must sum to at most one")
        if self.max_steps <= 0:
            raise AblationStudyError("max_steps must be positive")


class AblationStudyStateStore:
    """Atomic study-level state beneath the approved external data root."""

    def __init__(
        self,
        root: Path | None = None,
        *,
        study_id: str = ABLATION_STUDY_ID,
    ) -> None:
        self.root = Path(root) if root is not None else _default_state_root()
        self.study_id = study_id

    def _path(self) -> Path:
        return self.root / self.study_id / "state.json"

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
    *,
    source_commit: str,
    model_revision: str,
    protocol: AblationStudyProtocol | None = None,
) -> dict[str, object]:
    """Return the immutable study inputs recorded in public artifacts."""

    effective_protocol = protocol or landuse_ablation_protocol()
    return {
        "study_id": effective_protocol.study_id,
        "task_name": effective_protocol.task_name,
        "dataset_id": effective_protocol.dataset_contract.dataset_id,
        "dataset_revision": effective_protocol.dataset_contract.provenance.repository_revision,
        "model_revision": model_revision,
        "source_commit": source_commit,
        "screening_seed": effective_protocol.screening_seed,
        "replication_seeds": list(effective_protocol.replication_seeds),
        "selection_metric": effective_protocol.selection_metric,
        "tie_break_metric": effective_protocol.tie_break_metric,
        "definitions": [
            _definition_payload(definition)
            for definition in effective_protocol.definitions
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
        container_image: str | None = None,
        container_runtime: ContainerRuntime = "auto",
        cleanup: bool = True,
        allow_source_commit_update: bool = False,
        state_root: Path | None = None,
        protocol: AblationStudyProtocol | None = None,
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
        try:
            _validate_container_settings(container_image, container_runtime)
        except Grid5000ConfigurationError as error:
            raise AblationStudyError(str(error)) from error
        self.source_commit = source_commit
        self.model_revision = model_revision
        self.model_name_or_path = model_name_or_path
        self.sites = sites
        self.gpu_memory_mb = gpu_memory_mb
        self.walltime_seconds = walltime_seconds
        self.policy_type = policy_type
        self.max_workers = max_workers
        self.max_continuations = max_continuations
        self.container_image = container_image
        self.container_runtime = container_runtime
        self.cleanup = cleanup
        self.allow_source_commit_update = allow_source_commit_update
        self.protocol = protocol or landuse_ablation_protocol()
        self.state = AblationStudyStateStore(
            state_root,
            study_id=self.protocol.study_id,
        )
        self.run_state_root = self.state.root / "runs"
        self.run_controller_factory = run_controller_factory
        self.publish_report = publish_report or (lambda _state: None)
        self.emit = emit or (lambda _message: None)
        self.specification = study_specification(
            source_commit=source_commit,
            model_revision=model_revision,
            protocol=self.protocol,
        )
        self.fingerprint = study_specification_fingerprint(self.specification)

    def plan(self) -> dict[str, object]:
        """Return the next side-effect-free study plan."""

        state = self.state.load()
        if state is not None:
            self._validate_state(state)
        screening_results = self._screening_results(state or {})
        runs = self._planned_runs(screening_results or None)
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
            "study_id": self.protocol.study_id,
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
            "study_id": self.protocol.study_id,
            "fingerprint": self.fingerprint,
            "specification": self.specification,
            "phase": "running",
            "runs": {},
        }

    def _validate_state(self, state: Mapping[str, object]) -> None:
        state_mismatch = (
            state.get("schema_version") != 1
            or state.get("study_id") != self.protocol.study_id
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
            or state.get("study_id") != self.protocol.study_id
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
            if (
                record.get("phase") != "completed"
                or record.get("seed") != self.protocol.screening_seed
            ):
                continue
            ablation_id = record.get("ablation_id")
            metrics = record.get("metrics")
            if isinstance(ablation_id, str) and isinstance(metrics, Mapping):
                results[ablation_id] = cast(Mapping[str, object], metrics)
        return results

    def _planned_runs(
        self,
        screening_results: Mapping[str, Mapping[str, object]] | None,
    ) -> list[AblationRun]:
        return planned_ablation_runs(
            definitions=self.protocol.definitions,
            screening_results=screening_results,
            screening_seed=self.protocol.screening_seed,
            replication_seeds=self.protocol.replication_seeds,
            selection_metric=self.protocol.selection_metric,
            tie_break_metric=self.protocol.tie_break_metric,
        )

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
            protocol=self.protocol,
        )
        identity = Grid5000RunIdentity(
            source_commit=self.source_commit,
            dataset_revision=(
                self.protocol.dataset_contract.provenance.repository_revision
            ),
            model_name_or_path=config.model_name_or_path,
            model_revision=self.model_revision,
            task_name=self.protocol.task_name,
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
            container_image=self.container_image,
            container_runtime=self.container_runtime,
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
            runs = self._planned_runs(screening_results or None)
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
                self.emit(f"study {self.protocol.study_id}: completed")
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
                f"study {self.protocol.study_id}: running {pending.ablation_id} "
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


def _report_runs(
    state: Mapping[str, object],
    *,
    protocol: AblationStudyProtocol | None = None,
) -> list[dict[str, object]]:
    effective_protocol = protocol or landuse_ablation_protocol()
    records = AblationStudyController._records(state)
    specification = state.get("specification", {})
    if not isinstance(specification, Mapping):
        raise AblationStudyError("ablation study specification is invalid")
    screening = {
        str(record.get("ablation_id")): cast(
            Mapping[str, object], record.get("metrics")
        )
        for record in records.values()
        if record.get("seed") == effective_protocol.screening_seed
        and record.get("phase") == "completed"
        and isinstance(record.get("ablation_id"), str)
        and isinstance(record.get("metrics"), Mapping)
    }
    planned = planned_ablation_runs(
        definitions=effective_protocol.definitions,
        screening_results=screening or None,
        screening_seed=effective_protocol.screening_seed,
        replication_seeds=effective_protocol.replication_seeds,
        selection_metric=effective_protocol.selection_metric,
        tie_break_metric=effective_protocol.tie_break_metric,
    )
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
                    f"studies/{effective_protocol.study_id}/{run.ablation_id}/"
                    f"run-{record['run_id']}/final/"
                    if isinstance(record.get("run_id"), str)
                    and record.get("phase") == "completed"
                    else None
                ),
            }
        )
    return result_rows


def render_study_documents(
    state: Mapping[str, object],
    *,
    protocol: AblationStudyProtocol | None = None,
) -> dict[str, str]:
    """Render the public ablation documents from durable study state."""

    specification = state.get("specification")
    fingerprint = state.get("fingerprint")
    if not isinstance(specification, Mapping) or not isinstance(fingerprint, str):
        raise AblationStudyError("ablation study state lacks its public specification")
    effective_protocol = protocol or landuse_ablation_protocol()
    tracking_space_id = (
        V2_TRACKIO_STATIC_SPACE_ID
        if effective_protocol.task_name == "place-relevance-v2"
        else TRACKIO_STATIC_SPACE_ID
    )
    return render_public_documents(
        state,
        rows=_report_runs(state, protocol=effective_protocol),
        study_id=effective_protocol.study_id,
        tracking_space_id=tracking_space_id,
        study_title=effective_protocol.title,
        study_introduction=effective_protocol.introduction,
        evaluation_note=effective_protocol.evaluation_note,
        root_scope=(
            "worldwide V2 place-relevance sentence-classification task."
            if effective_protocol.task_name == "place-relevance-v2"
            else None
        ),
        include_root_readme=effective_protocol.task_name != "place-relevance-v2",
    )


def publish_study_report(
    state: Mapping[str, object],
    *,
    hub_api: Any | None = None,
    protocol: AblationStudyProtocol | None = None,
) -> None:
    """Publish the current study manifest and report to the existing model repo."""

    from .publication import ModelPublicationError, publish_study_documents

    try:
        publish_study_documents(
            TARGET_MODEL_REPOSITORY_ID,
            render_study_documents(state, protocol=protocol),
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


def landuse_ablation_protocol() -> AblationStudyProtocol:
    """Return the original Afghanistan landuse ablation protocol."""

    return AblationStudyProtocol(
        study_id=ABLATION_STUDY_ID,
        task_name="landuse",
        dataset_contract=LANDUSE_DATASET_CONTRACT,
        tracking_project=ABLATION_TRACKING_PROJECT,
        definitions=baseline_ablation_definitions(),
    )


def place_relevance_v2_ablation_protocol() -> AblationStudyProtocol:
    """Return the separate worldwide V2 screening and replication protocol."""

    return AblationStudyProtocol(
        study_id=PLACE_RELEVANCE_V2_ABLATION_STUDY_ID,
        task_name="place-relevance-v2",
        dataset_contract=WORLDWIDE_PLACE_RELEVANCE_V2_CONTRACT,
        tracking_project=PLACE_RELEVANCE_V2_ABLATION_TRACKING_PROJECT,
        definitions=baseline_ablation_definitions(),
        validation_fraction=0.1,
        test_fraction=0.1,
        eval_strategy="epoch",
        max_steps=17_661,
        output_task="place-relevance-v2",
        title="Worldwide V2 place-relevance ablation study",
        introduction=(
            "This study measures controlled changes to the worldwide V2 "
            "place-relevance sentence classifier."
        ),
        evaluation_note=(
            "Validation metrics select finalists; the held-out test set is "
            "evaluated once per run after training and is not used for selection."
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
    protocol: AblationStudyProtocol | None = None,
) -> TrainingConfig:
    """Create the fully explicit training configuration for one study run."""

    effective_protocol = protocol or landuse_ablation_protocol()
    namespace = f"studies/{effective_protocol.study_id}/{definition.ablation_id}"
    return TrainingConfig(
        model_name_or_path=model_name_or_path,
        output_subdirectory=Path(namespace)
        / f"models/{effective_protocol.output_task}",
        validation_fraction=effective_protocol.validation_fraction,
        test_fraction=effective_protocol.test_fraction,
        seed=seed,
        max_length=definition.max_length,
        learning_rate=definition.learning_rate,
        eval_strategy=effective_protocol.eval_strategy,
        max_steps=effective_protocol.max_steps,
        run_name=(
            f"{effective_protocol.study_id}|{definition.ablation_id}|seed-{seed}"
        ),
        model_revision=model_revision,
        trainable_layers=definition.trainable_layers,
        class_weight_mode=definition.class_weight_mode,
        tracking_project=effective_protocol.tracking_project,
        artifact_namespace=namespace,
        publish_to_hub=publish_to_hub,
        sync_trackio=sync_trackio,
    )


def rank_screening_results(
    results: Mapping[str, Mapping[str, object]],
    *,
    selection_metric: str = "eval_f1",
    tie_break_metric: str = "eval_macro_f1",
) -> list[str]:
    """Rank completed screening runs by two explicit scalar metrics."""

    ranked: list[tuple[str, float, float]] = []
    for ablation_id, metrics in results.items():
        positive_f1 = metrics.get(selection_metric)
        macro_f1 = metrics.get(tie_break_metric, 0.0)
        if not isinstance(positive_f1, (int, float)) or isinstance(positive_f1, bool):
            raise AblationStudyError(
                f"screening result for {ablation_id} lacks numeric {selection_metric}"
            )
        if not isinstance(macro_f1, (int, float)) or isinstance(macro_f1, bool):
            raise AblationStudyError(
                f"screening result for {ablation_id} has invalid {tie_break_metric}"
            )
        ranked.append((ablation_id, float(positive_f1), float(macro_f1)))
    ranked.sort(key=lambda item: (-item[1], -item[2], item[0]))
    return [item[0] for item in ranked]


def planned_ablation_runs(
    *,
    definitions: tuple[AblationDefinition, ...] | None = None,
    screening_results: Mapping[str, Mapping[str, object]] | None = None,
    screening_seed: int = 42,
    replication_seeds: tuple[int, ...] = (43, 44),
    selection_metric: str = "eval_f1",
    tie_break_metric: str = "eval_macro_f1",
) -> list[AblationRun]:
    """Return screening runs, or add seed replications after screening."""

    effective_definitions = definitions or baseline_ablation_definitions()
    by_id = {definition.ablation_id: definition for definition in effective_definitions}
    runs = [
        AblationRun(definition, screening_seed) for definition in effective_definitions
    ]
    if screening_results is None or not all(
        definition.ablation_id in screening_results
        for definition in effective_definitions
    ):
        return runs
    ranked_non_control = [
        ablation_id
        for ablation_id in rank_screening_results(
            screening_results,
            selection_metric=selection_metric,
            tie_break_metric=tie_break_metric,
        )
        if ablation_id != effective_definitions[0].ablation_id
    ]
    if len(ranked_non_control) < 2:
        raise AblationStudyError("screening requires two non-control finalists")
    finalists = [effective_definitions[0].ablation_id, *ranked_non_control[:2]]
    runs.extend(
        AblationRun(by_id[ablation_id], seed)
        for ablation_id in finalists
        for seed in replication_seeds
    )
    return runs


__all__ = [
    "ABLATION_STUDY_ID",
    "ABLATION_TRACKING_PROJECT",
    "PLACE_RELEVANCE_V2_ABLATION_STUDY_ID",
    "PLACE_RELEVANCE_V2_ABLATION_TRACKING_PROJECT",
    "AblationDefinition",
    "AblationRun",
    "AblationStudyController",
    "AblationStudyError",
    "AblationStudyProtocol",
    "AblationStudyStateStore",
    "DEFAULT_MODEL_REVISION",
    "baseline_ablation_definitions",
    "build_ablation_training_config",
    "landuse_ablation_protocol",
    "planned_ablation_runs",
    "place_relevance_v2_ablation_protocol",
    "publish_study_report",
    "rank_screening_results",
    "render_study_documents",
    "study_specification",
    "study_specification_fingerprint",
]
