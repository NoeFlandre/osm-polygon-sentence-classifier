"""Normalized OAR lifecycle operations for autonomous Grid'5000 runs."""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol
from zoneinfo import ZoneInfo

from .grid5000 import CommandResult, Grid5000ExecutionError

GRID5000_TIMEZONE = ZoneInfo("Europe/Paris")
_JOB_ID_PATTERN = re.compile(r"(?:OAR_JOB_ID=|^)([1-9][0-9]*)$", re.MULTILINE)
_TIMESTAMP_PATTERN = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}"
)


class JobState(StrEnum):
    """Normalized scheduler state."""

    QUEUED = "queued"
    RUNNING = "running"
    FINISHING = "finishing"
    TERMINATED = "terminated"
    ERROR = "error"
    MISSING = "missing"


LIVE_STATES = frozenset({JobState.QUEUED, JobState.RUNNING, JobState.FINISHING})
_OAR_STATES = {
    "waiting": JobState.QUEUED,
    "hold": JobState.QUEUED,
    "launching": JobState.QUEUED,
    "running": JobState.RUNNING,
    "finishing": JobState.FINISHING,
    "terminated": JobState.TERMINATED,
    "error": JobState.ERROR,
}


class OarError(RuntimeError):
    """Raised for malformed or unsafe OAR responses."""


class RemoteOar(Protocol):
    """The subset of the remote boundary required by OAR operations."""

    def raw(self, command: str) -> CommandResult: ...

    def run(self, command: str) -> CommandResult: ...


@dataclass(frozen=True, slots=True)
class JobStatus:
    """Facts needed to monitor one scheduler allocation."""

    job_id: int
    state: JobState
    exit_code: int | None = None
    message: str = ""
    node: str | None = None
    scheduled_start: str | None = None
    walltime_seconds: int | None = None


def parse_job_id(output: str) -> int:
    """Extract one unambiguous positive OAR job ID."""

    matches = _JOB_ID_PATTERN.findall(output.strip())
    if len(matches) != 1:
        raise OarError("submission did not return one job ID")
    return int(matches[0])


def _parse_walltime(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    return _parse_walltime_string(value) if isinstance(value, str) else None


def _parse_walltime_string(value: str) -> int | None:
    match = re.fullmatch(r"([0-9]+):([0-9]{2}):([0-9]{2})", value)
    if match is not None:
        hours, minutes, seconds = (int(part) for part in match.groups())
        return hours * 3_600 + minutes * 60 + seconds
    if value.isdigit():
        parsed = int(value)
        return parsed if parsed > 0 else None
    return None


def _parse_scheduled_start(value: object) -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return _format_timestamp(value)
    return (
        value
        if isinstance(value, str) and _TIMESTAMP_PATTERN.fullmatch(value)
        else None
    )


def _format_timestamp(value: int) -> str | None:
    if value <= 0:
        return None
    return datetime.fromtimestamp(value, tz=GRID5000_TIMEZONE).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def _parse_exit_code(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and re.fullmatch(r"-?[0-9]+", value):
        return int(value)
    return None


def _parse_status(job_id: int, payload: object) -> JobStatus:
    if not isinstance(payload, Mapping):
        raise OarError("OAR status JSON must be an object")
    record = _status_record(payload, job_id)
    state = _OAR_STATES.get(str(record.get("state", "")).casefold())
    if state is None:
        raise OarError("unsupported OAR job state")
    return JobStatus(
        job_id=job_id,
        state=state,
        exit_code=_parse_exit_code(record.get("exit_code")),
        message=str(record.get("message", "")),
        node=_status_node(record),
        scheduled_start=_parse_scheduled_start(record.get("scheduled_start")),
        walltime_seconds=_parse_walltime(record.get("walltime")),
    )


def _status_record(payload: Mapping[Any, Any], job_id: int) -> Mapping[Any, Any]:
    record = payload.get(str(job_id), payload)
    if not isinstance(record, Mapping):
        raise OarError("OAR status record must be an object")
    return record


def _status_node(record: Mapping[object, object]) -> str | None:
    node = record.get("assigned_network_address")
    return node if node is None or isinstance(node, str) else str(node)


class OarClient:
    """Perform fixed-argv OAR operations over the remote boundary."""

    def __init__(self, remote: RemoteOar) -> None:
        self.remote = remote

    def submit(self, command: Sequence[str]) -> int:
        """Submit exactly one already-audited scheduler command."""

        _validate_submission_command(command)
        try:
            result = self.remote.run(_shell_command(command))
            return parse_job_id(result.stdout)
        except OarError:
            raise
        except Exception as error:
            raise OarError("OAR submission failed") from error

    def status(self, job_id: int) -> JobStatus:
        """Read one job status; OAR exit code 6 is an explicit missing state."""

        _validate_job_id(job_id)
        result = self.remote.raw(f"oarstat -fj {job_id} -J")
        if result.returncode == 6:
            return JobStatus(job_id=job_id, state=JobState.MISSING)
        if result.returncode != 0:
            raise OarError(f"OAR status failed with exit code {result.returncode}")
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise OarError("invalid OAR status JSON") from error
        return _parse_status(job_id, payload)

    def cancel(self, job_id: int) -> None:
        """Cancel one positive job ID without retrying."""

        _validate_job_id(job_id)
        try:
            self.remote.run(f"oardel {job_id}")
        except Grid5000ExecutionError as error:
            raise OarError("OAR cancellation failed") from error

    def user_job_ids(self) -> tuple[int, ...]:
        """List the current user's jobs for safe legacy-state reconciliation."""

        result = self.remote.raw("oarstat -u -J")
        if result.returncode != 0:
            raise OarError(f"user OAR status failed with exit code {result.returncode}")
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise OarError("invalid user OAR status JSON") from error
        if not isinstance(payload, Mapping):
            raise OarError("user OAR status JSON must be an object")
        identifiers: list[int] = []
        identifiers.extend(_job_ids_from_payload(payload))
        return tuple(sorted(identifiers))


def _validate_submission_command(command: Sequence[str]) -> None:
    if not command or any(not isinstance(value, str) or not value for value in command):
        raise OarError("submission command cannot be empty")


def _shell_command(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(value) for value in command)


def _validate_job_id(job_id: object) -> None:
    if isinstance(job_id, bool) or not isinstance(job_id, int) or job_id <= 0:
        raise ValueError("job_id must be positive")


def _job_ids_from_payload(payload: Mapping[object, object]) -> tuple[int, ...]:
    return tuple(
        int(str(key)) for key in payload if re.fullmatch(r"[1-9][0-9]*", str(key))
    )


def format_job_status(status: JobStatus) -> str:
    """Render concise factual status for terminal output."""

    if status.state is JobState.QUEUED:
        return _format_queued_status(status)
    if status.state is JobState.RUNNING:
        return _format_running_status(status)
    return _format_terminal_status(status)


def _format_queued_status(status: JobStatus) -> str:
    if status.scheduled_start is None:
        return "queued; scheduler has no start-time prediction"
    return f"queued; scheduled start {status.scheduled_start} Europe/Paris"


def _format_running_status(status: JobStatus) -> str:
    suffix = f" on {status.node}" if status.node else ""
    return f"running{suffix}"


def _format_terminal_status(status: JobStatus) -> str:
    if status.exit_code is not None:
        return f"{status.state.value} (exit {status.exit_code})"
    return status.state.value


def is_live_state(state: JobState) -> bool:
    """Return whether an allocation can still consume resources."""

    return state in LIVE_STATES


__all__ = [
    "GRID5000_TIMEZONE",
    "LIVE_STATES",
    "JobState",
    "JobStatus",
    "OarClient",
    "OarError",
    "format_job_status",
    "is_live_state",
    "parse_job_id",
]
