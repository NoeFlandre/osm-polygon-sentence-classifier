"""Bounded remote checkpoint-evidence probes for Grid'5000 runs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Protocol

from .grid5000 import Grid5000ExecutionError

CHECKPOINT_PROBE_ATTEMPTS = 3
CHECKPOINT_PROBE_RETRY_SECONDS = 5.0


class CheckpointProbeRemote(Protocol):
    """Remote boundary required to verify a completed checkpoint."""

    def has_complete_checkpoint(
        self,
        run_id: str,
        *,
        output_subdirectory: str | Path,
        identity: Mapping[str, object],
        allow_failed_status: bool,
    ) -> bool: ...


class CheckpointProbeError(RuntimeError):
    """Raised when checkpoint evidence cannot be verified safely."""


def _detail_suffix(error: BaseException | None) -> str:
    if error is None:
        return ""
    detail = str(error).strip()
    return f": {detail[:240]}" if detail else ""


def probe_complete_checkpoint(
    remote: CheckpointProbeRemote,
    *,
    run_id: str,
    output_subdirectory: str | Path,
    identity: Mapping[str, object],
    allow_failed_status: bool,
    site: str,
    job_id: int,
    poll_seconds: float,
    emit: Callable[[str], None],
    sleep: Callable[[float], None],
    attempts: int = CHECKPOINT_PROBE_ATTEMPTS,
    retry_seconds: float = CHECKPOINT_PROBE_RETRY_SECONDS,
) -> bool:
    """Probe checkpoint evidence with bounded retries for SSH outages."""

    last_error: Grid5000ExecutionError | None = None
    for attempt in range(1, attempts + 1):
        try:
            return bool(
                remote.has_complete_checkpoint(
                    run_id,
                    output_subdirectory=output_subdirectory,
                    identity=identity,
                    allow_failed_status=allow_failed_status,
                )
            )
        except Grid5000ExecutionError as error:
            last_error = error
            if attempt == attempts:
                break
            delay = min(poll_seconds, retry_seconds)
            emit(
                f"{site} job {job_id}: checkpoint probe failed "
                f"(attempt {attempt}/{attempts}); "
                f"retrying in {delay:g}s"
            )
            sleep(delay)
        except Exception as error:
            raise CheckpointProbeError(
                f"checkpoint availability could not be verified{_detail_suffix(error)}"
            ) from error

    raise CheckpointProbeError(
        f"checkpoint availability could not be verified{_detail_suffix(last_error)}"
    ) from last_error


__all__ = ["CheckpointProbeError", "probe_complete_checkpoint"]
