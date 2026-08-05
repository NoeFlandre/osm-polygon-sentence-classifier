"""Secure durable state and recoverable legacy reconciliation."""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Final, cast

from .config import ProjectConfig

STATE_ROOT_SUBDIRECTORY: Final[Path] = Path("grid5000/runs")
_RUN_ID_PATTERN = re.compile(r"[0-9a-f]{20}")
_UNSAFE_FACT_KEYS = frozenset(
    {
        "token",
        "access_token",
        "authorization",
        "password",
        "secret",
        "hf_token",
    }
)


class RunPhase(StrEnum):
    """Autonomous lifecycle phases persisted locally."""

    CREATED = "created"
    PROBING = "probing"
    PREPARED = "prepared"
    SUBMITTING = "submitting"
    SUBMITTED = "submitted"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StateError(RuntimeError):
    """Base class for durable state failures."""


class StateSecurityError(StateError):
    """State or factual evidence violates the local security contract."""


class LegacyAmbiguousStateError(StateError):
    """An older submission state cannot safely be resumed or overwritten."""


JSONValue = str | int | float | bool | None | list["JSONValue"] | dict[str, "JSONValue"]


@dataclass(frozen=True, slots=True)
class AutonomousRunState:
    """Serializable identity and lifecycle facts for one run."""

    run_id: str
    phase: RunPhase | str
    identity: Mapping[str, object]
    site: str | None = None
    job_id: int | None = None
    facts: Mapping[str, object] | None = None
    updated_at: str = ""

    def __post_init__(self) -> None:
        if _RUN_ID_PATTERN.fullmatch(self.run_id) is None:
            raise StateError("run ID is invalid")
        try:
            phase = RunPhase(self.phase)
        except ValueError as error:
            raise StateError("run phase is invalid") from error
        object.__setattr__(self, "phase", phase)
        if self.site is not None and (
            not isinstance(self.site, str) or not self.site.strip()
        ):
            raise StateError("state site is invalid")
        if self.job_id is not None and (
            isinstance(self.job_id, bool)
            or not isinstance(self.job_id, int)
            or self.job_id <= 0
        ):
            raise StateError("state job ID is invalid")
        identity = _sanitize_mapping(self.identity)
        facts = _sanitize_mapping(self.facts or {})
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "facts", facts)
        timestamp = self.updated_at or _now()
        _parse_timestamp(timestamp)
        object.__setattr__(self, "updated_at", timestamp)

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-safe state document."""

        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "phase": RunPhase(self.phase).value,
            "identity": dict(self.identity),
            "site": self.site,
            "job_id": self.job_id,
            "facts": dict(self.facts or {}),
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> AutonomousRunState:
        if payload.get("schema_version") != 1:
            raise StateError("state schema version is unsupported")
        try:
            run_id = payload["run_id"]
            phase = payload["phase"]
            identity = payload["identity"]
        except KeyError as error:
            raise StateError("state document is incomplete") from error
        if not isinstance(run_id, str) or not isinstance(phase, str):
            raise StateError("state identity or phase is invalid")
        if not isinstance(identity, Mapping):
            raise StateError("state identity is invalid")
        identity_mapping = cast(Mapping[str, object], identity)
        site = payload.get("site")
        if site is not None and not isinstance(site, str):
            raise StateError("state site is invalid")
        job_id = payload.get("job_id")
        if job_id is not None and (
            isinstance(job_id, bool) or not isinstance(job_id, int)
        ):
            raise StateError("state job ID is invalid")
        facts = payload.get("facts", {})
        if not isinstance(facts, Mapping):
            raise StateError("state facts are invalid")
        facts_mapping = cast(Mapping[str, object], facts)
        updated_at = payload.get("updated_at")
        if not isinstance(updated_at, str):
            raise StateError("state timestamp is invalid")
        return cls(
            run_id=run_id,
            phase=phase,
            identity=identity_mapping,
            site=site,
            job_id=job_id,
            facts=facts_mapping,
            updated_at=updated_at,
        )


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _parse_timestamp(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise StateError("state timestamp is not ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise StateError("state timestamp must be timezone-aware")


def _sanitize_scalar(value: object) -> JSONValue:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise StateError("state facts must contain finite JSON values")
        return value
    raise StateError("state facts must be JSON-compatible values")


def _sanitize_mapping_value(value: Mapping[object, object]) -> dict[str, JSONValue]:
    result: dict[str, JSONValue] = {}
    for key, child in value.items():
        if not isinstance(key, str):
            raise StateError("state facts must use string keys")
        if key.casefold() in _UNSAFE_FACT_KEYS:
            raise StateSecurityError(f"unsafe state fact key rejected: {key}")
        result[key] = _sanitize(child)
    return result


def _sanitize(value: object) -> JSONValue:
    if isinstance(value, Mapping):
        return _sanitize_mapping_value(cast(Mapping[object, object], value))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_sanitize(child) for child in value]
    return _sanitize_scalar(value)


def _sanitize_mapping(value: Mapping[str, object]) -> dict[str, JSONValue]:
    if not isinstance(value, Mapping):
        raise StateError("state identity/facts must be mappings")
    result = _sanitize(value)
    if not isinstance(result, dict):
        raise StateError("state identity/facts must be mappings")
    return result


def _reject_symlinks(path: Path) -> None:
    if not path.is_absolute():
        raise StateSecurityError("state root must be absolute")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise StateSecurityError("state path cannot contain symlinks")


def _check_mode(path: Path, expected: int, label: str) -> None:
    if path.is_symlink() or not path.exists():
        raise StateSecurityError(f"{label} is missing or symlinked")
    if path.stat().st_mode & 0o777 != expected:
        raise StateSecurityError(f"{label} has unsafe permissions")


class AutonomousStateStore:
    """Atomic state store beneath the approved external data root."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (
            Path(root)
            if root is not None
            else ProjectConfig().data_root / STATE_ROOT_SUBDIRECTORY
        )
        _reject_symlinks(self.root)

    def _validate_run_id(self, run_id: str) -> None:
        if _RUN_ID_PATTERN.fullmatch(run_id) is None:
            raise StateError("run ID is invalid")

    def _directory(self, run_id: str) -> Path:
        self._validate_run_id(run_id)
        return self.root / run_id

    def _ensure_root(self) -> None:
        _reject_symlinks(self.root)
        try:
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(self.root, 0o700)
        except OSError as error:
            raise StateSecurityError("state root cannot be created securely") from error

    def _ensure_directory(self, run_id: str) -> Path:
        self._ensure_root()
        directory = self._directory(run_id)
        if directory.is_symlink():
            raise StateSecurityError("run state directory cannot be a symlink")
        try:
            directory.mkdir(mode=0o700)
            os.chmod(directory, 0o700)
        except FileExistsError as error:
            raise StateError("run already has durable state") from error
        except OSError as error:
            raise StateSecurityError("run state directory cannot be created") from error
        return directory

    def _write_state(self, directory: Path, state: AutonomousRunState) -> None:
        temporary = directory / ".state.json.tmp"
        try:
            temporary.write_text(
                json.dumps(
                    state.to_dict(),
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            os.chmod(temporary, 0o600)
            os.replace(temporary, directory / "state.json")
        except (OSError, TypeError, ValueError) as error:
            temporary.unlink(missing_ok=True)
            raise StateError("state document cannot be written") from error

    def create(self, state: AutonomousRunState) -> None:
        """Create one new state document without overwriting an existing run."""

        directory = self._ensure_directory(state.run_id)
        self._write_state(directory, state)

    def save(self, state: AutonomousRunState) -> None:
        """Atomically update an existing state document."""

        directory = self._directory(state.run_id)
        if directory.is_symlink() or not directory.is_dir():
            raise StateError("run state directory is missing or unsafe")
        _check_mode(directory, 0o700, "run state directory")
        self._write_state(directory, state)

    def load(self, run_id: str) -> AutonomousRunState | None:
        """Load current state, explicitly surfacing legacy ambiguity."""

        directory = self._directory(run_id)
        if not directory.exists():
            return None
        if directory.is_symlink():
            raise StateSecurityError("run state directory cannot be a symlink")
        _check_mode(directory, 0o700, "run state directory")
        path = directory / "state.json"
        if not path.exists() or path.is_symlink():
            raise StateError("state document is missing or unsafe")
        _check_mode(path, 0o600, "state document")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise StateError("state document cannot be read") from error
        if isinstance(payload, Mapping) and payload.get("schema_version") is None:
            raise LegacyAmbiguousStateError(
                "legacy Grid'5000 state is ambiguous; reconcile before starting"
            )
        if not isinstance(payload, Mapping):
            raise StateError("state document must be an object")
        state = AutonomousRunState.from_dict(payload)
        if state.run_id != run_id:
            raise StateError("state identity does not match its directory")
        return state

    def append_event(
        self,
        run_id: str,
        event: str,
        facts: Mapping[str, object] | None = None,
    ) -> None:
        """Append one credential-free JSON event."""

        if not event or "\n" in event or "\r" in event:
            raise StateError("event name is invalid")
        directory = self._directory(run_id)
        if not directory.is_dir() or directory.is_symlink():
            raise StateError("run state directory is missing or unsafe")
        _check_mode(directory, 0o700, "run state directory")
        event_payload = {
            "schema_version": 1,
            "timestamp": _now(),
            "event": event,
            "facts": _sanitize_mapping(facts or {}),
        }
        path = directory / "events.jsonl"
        if path.exists() and path.is_symlink():
            raise StateSecurityError("events document cannot be a symlink")
        try:
            with path.open("a", encoding="utf-8") as handle:
                os.chmod(path, 0o600)
                handle.write(json.dumps(event_payload, sort_keys=True) + "\n")
        except OSError as error:
            raise StateError("event document cannot be written") from error

    def reconcile_legacy(
        self,
        run_id: str,
        *,
        active_job_ids: Sequence[int],
    ) -> Path:
        """Move legacy state to timestamped evidence only when the scheduler is idle."""

        directory = self._directory(run_id)
        if active_job_ids:
            raise LegacyAmbiguousStateError(
                "legacy Grid'5000 state cannot be reconciled while jobs are active"
            )
        _check_mode(directory, 0o700, "legacy run state directory")
        path = directory / "state.json"
        if not path.is_file() or path.is_symlink():
            raise StateError("legacy state document is missing or unsafe")
        _check_mode(path, 0o600, "legacy state document")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise StateError("legacy state document cannot be read") from error
        if (
            not isinstance(payload, Mapping)
            or payload.get("schema_version") is not None
        ):
            raise StateError("state is not a legacy document")
        archive_root = self.root / "legacy-ambiguous"
        if archive_root.exists():
            _check_mode(archive_root, 0o700, "legacy archive directory")
        else:
            archive_root.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(archive_root, 0o700)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        archived = archive_root / f"{run_id}-{stamp}"
        if archived.exists():
            raise StateError("legacy archive target already exists")
        try:
            shutil.move(str(directory), str(archived))
        except OSError as error:
            raise StateError("legacy state could not be archived") from error
        return archived


__all__ = [
    "AutonomousRunState",
    "AutonomousStateStore",
    "LegacyAmbiguousStateError",
    "RunPhase",
    "StateError",
    "StateSecurityError",
]
