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
        _validate_protocol_identity(self)
        _validate_protocol_contents(self)
        _validate_protocol_splits(self)
        _validate_protocol_budget(self)


def _validate_protocol_identity(protocol: AblationStudyProtocol) -> None:
    _validate_clean_study_id(protocol.study_id)
    _validate_protocol_names(protocol.task_name, protocol.tracking_project)


def _validate_clean_study_id(value: str) -> None:
    if not value or any(char in value for char in "/\\\n\r"):
        raise AblationStudyError("study_id must be a clean single-line name")


def _validate_protocol_names(task_name: str, tracking_project: str) -> None:
    if not task_name or not tracking_project:
        raise AblationStudyError("study task and tracking project are required")


def _validate_protocol_contents(protocol: AblationStudyProtocol) -> None:
    if not protocol.definitions:
        raise AblationStudyError("at least one ablation definition is required")
    if not protocol.replication_seeds:
        raise AblationStudyError("at least one replication seed is required")


def _validate_protocol_splits(protocol: AblationStudyProtocol) -> None:
    if protocol.validation_fraction < 0 or protocol.test_fraction < 0:
        raise AblationStudyError("split fractions must be non-negative")
    if protocol.validation_fraction + protocol.test_fraction > 1:
        raise AblationStudyError("split fractions must sum to at most one")


def _validate_protocol_budget(protocol: AblationStudyProtocol) -> None:
    if protocol.max_steps <= 0:
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
    return hashlib.sha256(_canonical_json(specification).encode()).hexdigest()


def _metric_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return {
        cast(str, key): item
        for key, item in value.items()
        if _is_metric_entry(key, item)
    }


def _is_metric_entry(key: object, value: object) -> bool:
    return isinstance(key, str) and _is_metric_value(value)


def _is_metric_value(value: object) -> bool:
    return isinstance(value, (int, float, str)) and not isinstance(value, bool)


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
        _validate_controller_arguments(
            source_commit=source_commit,
            model_revision=model_revision,
            sites=sites,
            gpu_memory_mb=gpu_memory_mb,
            walltime_seconds=walltime_seconds,
            max_workers=max_workers,
            max_continuations=max_continuations,
            allow_source_commit_update=allow_source_commit_update,
            policy_type=policy_type,
            container_image=container_image,
            container_runtime=container_runtime,
        )
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
        pending = _pending_plan_records(runs, records, self._run_key)
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
        if not _state_identity_matches(
            state,
            study_id=self.protocol.study_id,
            fingerprint=self.fingerprint,
        ):
            if self.allow_source_commit_update and self._can_adopt_source_commit(state):
                return
            raise AblationStudyError(
                "existing ablation state does not match the immutable study specification"
            )
        _validate_run_records(state)

    def _can_adopt_source_commit(self, state: Mapping[str, object]) -> bool:
        specification = _compatible_adoption_specification(
            state,
            study_id=self.protocol.study_id,
            current_source_commit=self.source_commit,
            current_specification=self.specification,
        )
        if specification is None:
            return False
        return not _has_running_records(self._records(state))

    def _adopt_source_commit_if_needed(
        self, state: dict[str, object]
    ) -> dict[str, object]:
        if not self.allow_source_commit_update or not self._can_adopt_source_commit(
            state
        ):
            return state
        old_source_commit = _adoption_source(state)
        history = _source_commit_history(state, old_source_commit)
        records = self._records(state)
        _add_source_commit_to_records(records, old_source_commit)
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
            result = _screening_result(record, self.protocol.screening_seed)
            if result is not None:
                ablation_id, metrics = result
                results[ablation_id] = metrics
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

        state = self._load_run_state()
        if state.get("phase") == "completed":
            return state

        while True:
            pending = self._next_pending_run(state)
            if pending is None:
                return self._complete_study(state)
            self._execute_pending_run(state, pending)

    def _load_run_state(self) -> dict[str, object]:
        state = self.state.load()
        if state is None:
            state = self._new_state()
            self.state.save(state)
            return state
        self._validate_state(state)
        return self._adopt_source_commit_if_needed(state)

    def _next_pending_run(self, state: Mapping[str, object]) -> AblationRun | None:
        screening_results = self._screening_results(state)
        runs = self._planned_runs(screening_results or None)
        records = self._records(state)
        return next(
            (
                run
                for run in runs
                if records.get(self._run_key(run), {}).get("phase") != "completed"
            ),
            None,
        )

    def _complete_study(self, state: dict[str, object]) -> dict[str, object]:
        state["phase"] = "completed"
        self.state.save(state)
        self.publish_report(state)
        self.emit(f"study {self.protocol.study_id}: completed")
        return state

    def _execute_pending_run(
        self, state: dict[str, object], pending: AblationRun
    ) -> None:
        key = self._run_key(pending)
        config = self._autonomous_config(pending)
        records = self._records(state)
        self._record_run_started(state, records, key, pending, config.identity.run_id)
        try:
            run_state = self._run_controller(config)
        except Exception as error:
            self._record_run_failure(state, records, key, error)
            raise AblationStudyError(
                f"ablation {pending.ablation_id} seed {pending.seed} failed"
            ) from error
        if getattr(run_state, "phase", None) is not RunPhase.COMPLETED:
            self._record_run_phase(state, records, key, run_state)
            phase = str(getattr(run_state, "phase", "unknown"))
            raise AblationStudyError(
                f"ablation {pending.ablation_id} seed {pending.seed} ended in {phase}"
            )
        metrics = _completion_metrics(run_state)
        records[key].update({"phase": "completed", "metrics": metrics})
        state["runs"] = records
        self.state.save(state)
        self.publish_report(state)

    def _record_run_started(
        self,
        state: dict[str, object],
        records: dict[str, dict[str, object]],
        key: str,
        pending: AblationRun,
        run_id: str,
    ) -> None:
        records[key] = {
            "ablation_id": pending.ablation_id,
            "seed": pending.seed,
            "source_commit": self.source_commit,
            "run_id": run_id,
            "phase": "running",
        }
        state["runs"] = records
        self.state.save(state)
        self.emit(
            f"study {self.protocol.study_id}: running {pending.ablation_id} "
            f"seed={pending.seed} run={run_id}"
        )

    def _run_controller(self, config: AutonomousRunConfig) -> Any:
        run_controller = self.run_controller_factory(
            config,
            state_root=self.run_state_root,
            emit=self.emit,
        )
        return run_controller.run()

    def _record_run_failure(
        self,
        state: dict[str, object],
        records: dict[str, dict[str, object]],
        key: str,
        error: Exception,
    ) -> None:
        records[key].update({"phase": "failed", "error": str(error)})
        state["runs"] = records
        self.state.save(state)

    def _record_run_phase(
        self,
        state: dict[str, object],
        records: dict[str, dict[str, object]],
        key: str,
        run_state: Any,
    ) -> None:
        phase = str(getattr(run_state, "phase", "unknown"))
        records[key].update({"phase": phase})
        state["runs"] = records
        self.state.save(state)


def _completion_metrics(run_state: Any) -> dict[str, object]:
    try:
        facts = run_state.facts
    except AttributeError:
        return {}
    if not isinstance(facts, Mapping):
        return {}
    completion = facts.get("completion")
    if not isinstance(completion, Mapping):
        return {}
    return _metric_mapping(completion.get("metrics"))


def _pending_plan_records(
    runs: list[AblationRun],
    records: Mapping[str, Mapping[str, object]],
    run_key: Callable[[AblationRun], str],
) -> list[dict[str, object]]:
    pending: list[dict[str, object]] = []
    for run in runs:
        record = records.get(run_key(run), {})
        if record.get("phase") == "completed":
            continue
        pending.append(
            {
                "ablation_id": run.ablation_id,
                "seed": run.seed,
                "status": record.get("phase", "pending"),
            }
        )
    return pending


def _state_identity_matches(
    state: Mapping[str, object], *, study_id: str, fingerprint: str
) -> bool:
    return (
        state.get("schema_version") == 1
        and state.get("study_id") == study_id
        and state.get("fingerprint") == fingerprint
    )


def _validate_run_records(state: Mapping[str, object]) -> None:
    if not isinstance(state.get("runs", {}), Mapping):
        raise AblationStudyError("ablation study run state is invalid")


def _adoptable_state_identity(state: Mapping[str, object], study_id: str) -> bool:
    return state.get("schema_version") == 1 and state.get("study_id") == study_id


def _compatible_adoption_specification(
    state: Mapping[str, object],
    *,
    study_id: str,
    current_source_commit: str,
    current_specification: Mapping[str, object],
) -> Mapping[str, object] | None:
    if not _adoptable_state_identity(state, study_id):
        return None
    specification = _stored_specification(state)
    if specification is None:
        return None
    if not _adoption_specification_matches(
        state,
        specification,
        current_source_commit=current_source_commit,
        current_specification=current_specification,
    ):
        return None
    return specification


def _stored_specification(state: Mapping[str, object]) -> Mapping[str, object] | None:
    raw_specification = state.get("specification")
    if not isinstance(raw_specification, Mapping):
        return None
    return cast(Mapping[str, object], raw_specification)


def _adoption_specification_matches(
    state: Mapping[str, object],
    specification: Mapping[str, object],
    *,
    current_source_commit: str,
    current_specification: Mapping[str, object],
) -> bool:
    return (
        _adoptable_source_commit(
            specification.get("source_commit"),
            current_source_commit=current_source_commit,
            phase=state.get("phase"),
        )
        and study_specification_fingerprint(specification) == state.get("fingerprint")
        and _specifications_match_without_source(current_specification, specification)
    )


def _adoptable_source_commit(
    value: object,
    *,
    current_source_commit: str,
    phase: object,
) -> bool:
    return (
        isinstance(value, str)
        and value != current_source_commit
        and phase != "completed"
    )


def _specifications_match_without_source(
    current: Mapping[str, object], stored: Mapping[str, object]
) -> bool:
    current_without_source = {
        key: value for key, value in current.items() if key != "source_commit"
    }
    stored_without_source = {
        key: value for key, value in stored.items() if key != "source_commit"
    }
    return current_without_source == stored_without_source


def _has_running_records(records: Mapping[str, Mapping[str, object]]) -> bool:
    return any(record.get("phase") == "running" for record in records.values())


def _screening_result(
    record: Mapping[str, object], screening_seed: int
) -> tuple[str, Mapping[str, object]] | None:
    if record.get("phase") != "completed" or record.get("seed") != screening_seed:
        return None
    ablation_id = record.get("ablation_id")
    metrics = record.get("metrics")
    if not isinstance(ablation_id, str) or not isinstance(metrics, Mapping):
        return None
    return ablation_id, cast(Mapping[str, object], metrics)


def _adoption_source(state: Mapping[str, object]) -> str:
    specification = state.get("specification")
    if not isinstance(specification, Mapping):
        raise AblationStudyError("ablation study specification is invalid")
    old_source_commit = specification.get("source_commit")
    if not isinstance(old_source_commit, str):
        raise AblationStudyError("ablation study source commit is invalid")
    return old_source_commit


def _source_commit_history(
    state: Mapping[str, object], old_source_commit: str
) -> list[str]:
    raw_history = state.get("source_commit_history", [])
    history = (
        [item for item in raw_history if isinstance(item, str)]
        if isinstance(raw_history, list)
        else []
    )
    if old_source_commit not in history:
        history.append(old_source_commit)
    return history


def _add_source_commit_to_records(
    records: Mapping[str, dict[str, object]], old_source_commit: str
) -> None:
    for record in records.values():
        record.setdefault("source_commit", old_source_commit)


def _validate_controller_arguments(
    *,
    source_commit: str,
    model_revision: str,
    sites: tuple[str, ...],
    gpu_memory_mb: int,
    walltime_seconds: int,
    max_workers: int,
    max_continuations: int,
    allow_source_commit_update: bool,
    policy_type: PolicyType,
    container_image: str | None,
    container_runtime: ContainerRuntime,
) -> None:
    _validate_revisions(source_commit, model_revision)
    _validate_controller_limits(
        sites=sites,
        gpu_memory_mb=gpu_memory_mb,
        walltime_seconds=walltime_seconds,
        max_workers=max_workers,
        max_continuations=max_continuations,
    )
    _validate_controller_policy(allow_source_commit_update, policy_type)
    _validate_container_arguments(container_image, container_runtime)


def _validate_revisions(source_commit: str, model_revision: str) -> None:
    if _REVISION_PATTERN.fullmatch(source_commit) is None:
        raise AblationStudyError("source_commit must be a pinned revision")
    if _REVISION_PATTERN.fullmatch(model_revision) is None:
        raise AblationStudyError("model_revision must be a pinned revision")


def _validate_controller_limits(
    *,
    sites: tuple[str, ...],
    gpu_memory_mb: int,
    walltime_seconds: int,
    max_workers: int,
    max_continuations: int,
) -> None:
    if not sites:
        raise AblationStudyError("at least one Grid'5000 site is required")
    _validate_resource_limits(gpu_memory_mb, walltime_seconds)
    _validate_worker_limits(max_workers, max_continuations)


def _validate_resource_limits(gpu_memory_mb: int, walltime_seconds: int) -> None:
    if gpu_memory_mb <= 0 or walltime_seconds <= 0:
        raise AblationStudyError("GPU memory and walltime must be positive")


def _validate_worker_limits(max_workers: int, max_continuations: int) -> None:
    if max_workers <= 0 or max_continuations <= 0:
        raise AblationStudyError("worker and continuation limits must be positive")


def _validate_controller_policy(
    allow_source_commit_update: bool,
    policy_type: PolicyType,
) -> None:
    if not isinstance(allow_source_commit_update, bool):
        raise AblationStudyError("source commit update permission must be a boolean")
    if policy_type not in {"auto", "day", "night"}:
        raise AblationStudyError("policy_type must be auto, day, or night")


def _validate_container_arguments(
    container_image: str | None,
    container_runtime: ContainerRuntime,
) -> None:
    try:
        _validate_container_settings(container_image, container_runtime)
    except Grid5000ConfigurationError as error:
        raise AblationStudyError(str(error)) from error


def _report_runs(
    state: Mapping[str, object],
    *,
    protocol: AblationStudyProtocol | None = None,
) -> list[dict[str, object]]:
    effective_protocol = protocol or landuse_ablation_protocol()
    records = AblationStudyController._records(state)
    _validate_report_specification(state)
    screening = _report_screening_results(records, effective_protocol.screening_seed)
    planned = planned_ablation_runs(
        definitions=effective_protocol.definitions,
        screening_results=screening or None,
        screening_seed=effective_protocol.screening_seed,
        replication_seeds=effective_protocol.replication_seeds,
        selection_metric=effective_protocol.selection_metric,
        tie_break_metric=effective_protocol.tie_break_metric,
    )
    return [_report_row(run, records, effective_protocol.study_id) for run in planned]


def _validate_report_specification(state: Mapping[str, object]) -> None:
    if not isinstance(state.get("specification", {}), Mapping):
        raise AblationStudyError("ablation study specification is invalid")


def _report_screening_results(
    records: Mapping[str, Mapping[str, object]], screening_seed: int
) -> dict[str, Mapping[str, object]]:
    results: dict[str, Mapping[str, object]] = {}
    for record in records.values():
        result = _screening_result(record, screening_seed)
        if result is not None:
            ablation_id, metrics = result
            results[ablation_id] = metrics
    return results


def _report_row(
    run: AblationRun,
    records: Mapping[str, Mapping[str, object]],
    study_id: str,
) -> dict[str, object]:
    record = records.get(f"{run.ablation_id}|seed-{run.seed}", {})
    return {
        "ablation_id": run.ablation_id,
        "seed": run.seed,
        "status": record.get("phase", "pending"),
        "run_id": record.get("run_id"),
        "source_commit": record.get("source_commit"),
        "metrics": _metric_mapping(record.get("metrics")),
        "model_path": _model_path(record, run, study_id),
    }


def _model_path(
    record: Mapping[str, object], run: AblationRun, study_id: str
) -> str | None:
    run_id = record.get("run_id")
    if not isinstance(run_id, str) or record.get("phase") != "completed":
        return None
    return f"studies/{study_id}/{run.ablation_id}/run-{run_id}/final/"


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
    tracking_space_id, root_scope, include_root_readme = _document_options(
        effective_protocol
    )
    return render_public_documents(
        state,
        rows=_report_runs(state, protocol=effective_protocol),
        study_id=effective_protocol.study_id,
        tracking_space_id=tracking_space_id,
        study_title=effective_protocol.title,
        study_introduction=effective_protocol.introduction,
        evaluation_note=effective_protocol.evaluation_note,
        root_scope=root_scope,
        include_root_readme=include_root_readme,
    )


def _document_options(
    protocol: AblationStudyProtocol,
) -> tuple[str, str | None, bool]:
    if protocol.task_name == "place-relevance-v2":
        return (
            V2_TRACKIO_STATIC_SPACE_ID,
            "worldwide V2 place-relevance sentence-classification task.",
            False,
        )
    return TRACKIO_STATIC_SPACE_ID, None, True


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

    ranked = [
        _ranking_tuple(
            ablation_id,
            metrics,
            selection_metric=selection_metric,
            tie_break_metric=tie_break_metric,
        )
        for ablation_id, metrics in results.items()
    ]
    ranked.sort(key=lambda item: (-item[1], -item[2], item[0]))
    return [item[0] for item in ranked]


def _ranking_tuple(
    ablation_id: str,
    metrics: Mapping[str, object],
    *,
    selection_metric: str,
    tie_break_metric: str,
) -> tuple[str, float, float]:
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
    return ablation_id, float(positive_f1), float(macro_f1)


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
    runs = _screening_runs(effective_definitions, screening_seed)
    if not _has_all_screening_results(effective_definitions, screening_results):
        return runs
    runs.extend(
        _planned_replication_runs(
            effective_definitions,
            screening_results,
            replication_seeds=replication_seeds,
            selection_metric=selection_metric,
            tie_break_metric=tie_break_metric,
        )
    )
    return runs


def _screening_runs(
    definitions: tuple[AblationDefinition, ...],
    seed: int,
) -> list[AblationRun]:
    return [AblationRun(definition, seed) for definition in definitions]


def _planned_replication_runs(
    definitions: tuple[AblationDefinition, ...],
    screening_results: Mapping[str, Mapping[str, object]] | None,
    *,
    replication_seeds: tuple[int, ...],
    selection_metric: str,
    tie_break_metric: str,
) -> list[AblationRun]:
    assert screening_results is not None
    ranked_non_control = _ranked_non_control(
        screening_results,
        control_id=definitions[0].ablation_id,
        selection_metric=selection_metric,
        tie_break_metric=tie_break_metric,
    )
    if len(ranked_non_control) < 2:
        raise AblationStudyError("screening requires two non-control finalists")
    finalists = _finalist_ids(definitions[0].ablation_id, ranked_non_control)
    by_id = {definition.ablation_id: definition for definition in definitions}
    return _replication_runs(by_id, finalists, replication_seeds)


def _has_all_screening_results(
    definitions: tuple[AblationDefinition, ...],
    screening_results: Mapping[str, Mapping[str, object]] | None,
) -> bool:
    return screening_results is not None and all(
        definition.ablation_id in screening_results for definition in definitions
    )


def _ranked_non_control(
    screening_results: Mapping[str, Mapping[str, object]],
    *,
    control_id: str,
    selection_metric: str,
    tie_break_metric: str,
) -> list[str]:
    return [
        ablation_id
        for ablation_id in rank_screening_results(
            screening_results,
            selection_metric=selection_metric,
            tie_break_metric=tie_break_metric,
        )
        if ablation_id != control_id
    ]


def _finalist_ids(control_id: str, ranked_non_control: list[str]) -> list[str]:
    return [control_id, *ranked_non_control[:2]]


def _replication_runs(
    definitions: Mapping[str, AblationDefinition],
    finalists: list[str],
    seeds: tuple[int, ...],
) -> list[AblationRun]:
    return [
        AblationRun(definitions[ablation_id], seed)
        for ablation_id in finalists
        for seed in seeds
    ]


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
