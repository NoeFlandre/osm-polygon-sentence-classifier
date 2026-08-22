"""Grid'5000 policy-window and bounded short-job replacement rules."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from .grid5000_oar import GRID5000_TIMEZONE, JobState, JobStatus

IMMEDIATE_START_LIMIT = timedelta(minutes=10)
SHORT_TRIAL_WALLTIME_SECONDS = 20 * 60
TRIAL_TIMEOUT_SECONDS = 10 * 60
TRIAL_POLL_SECONDS = 30.0


def _forecast_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return parsed.replace(tzinfo=GRID5000_TIMEZONE)


def policy_type_for(now: datetime, *, walltime_seconds: int) -> str:
    """Return the policy window that can contain a complete short allocation."""

    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    if walltime_seconds <= 0:
        raise ValueError("walltime_seconds must be positive")
    local = now.astimezone(GRID5000_TIMEZONE)
    if local.weekday() >= 5:
        return "night"
    return _weekday_policy_type(local, timedelta(seconds=walltime_seconds))


def _weekday_policy_type(local: datetime, duration: timedelta) -> str:
    day_start = local.replace(hour=9, minute=0, second=0, microsecond=0)
    day_end = local.replace(hour=19, minute=0, second=0, microsecond=0)
    if local < day_start:
        return "night" if local + duration <= day_start else "day"
    if local < day_end:
        return "day" if local + duration <= day_end else "night"
    return "night"


def should_seek_replacement(
    status: JobStatus,
    *,
    now: datetime,
    immediate_start_limit: timedelta = IMMEDIATE_START_LIMIT,
) -> bool:
    """Return whether a queued fallback needs a bounded replacement trial.

    OAR may leave ``scheduled_start`` unset while a job remains queued. That is
    not evidence of an imminent start, so the controller treats it as eligible
    for one bounded replacement round. A forecast that has already passed is
    also stale evidence while the job is still queued. Trial candidates use
    :func:`forecast_exceeds_immediate_window` instead, which preserves their
    full observation window when their own forecast is unknown.
    """

    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    if immediate_start_limit <= timedelta(0):
        raise ValueError("immediate_start_limit must be positive")
    if status.state is not JobState.QUEUED:
        return False
    forecast = _forecast_datetime(status.scheduled_start)
    return _forecast_is_outside_window(forecast, now, immediate_start_limit)


def _forecast_is_outside_window(
    forecast: datetime | None,
    local_now: datetime,
    immediate_start_limit: timedelta,
) -> bool:
    if forecast is None:
        return True
    return forecast <= local_now or forecast > local_now + immediate_start_limit


@dataclass(frozen=True, slots=True)
class QueuedReplacementDecision:
    """Pure action selected for a queued fallback allocation."""

    action: Literal["wait", "replace", "fail"]
    attempt_count: int | None = None
    attempt_timestamp: str | None = None
    message: str | None = None


def decide_queued_replacement(
    status: JobStatus,
    *,
    site: str,
    job_id: int,
    now: datetime,
    attempt_count: int,
    retry_due: bool,
    max_attempts: int,
) -> QueuedReplacementDecision:
    """Choose the next bounded action for a queued fallback job."""

    if not should_seek_replacement(status, now=now):
        return QueuedReplacementDecision(action="wait")
    if attempt_count >= max_attempts:
        if status.scheduled_start is None:
            message = (
                f"{site} job {job_id} remained queued with no start-time "
                f"prediction after {max_attempts} replacement rounds"
            )
        else:
            message = (
                f"{site} job {job_id} remained queued with scheduled start "
                f"{status.scheduled_start} after {max_attempts} replacement rounds"
            )
        return QueuedReplacementDecision(action="fail", message=message)
    if not retry_due:
        return QueuedReplacementDecision(action="wait")
    return QueuedReplacementDecision(
        action="replace",
        attempt_count=attempt_count + 1,
        attempt_timestamp=now.isoformat(),
    )


def forecast_exceeds_immediate_window(
    status: JobStatus,
    *,
    now: datetime,
    immediate_start_limit: timedelta = IMMEDIATE_START_LIMIT,
) -> bool:
    """Return whether a replacement trial is known to be too late."""

    if not _has_queued_forecast(status):
        return False
    return should_seek_replacement(
        status,
        now=now,
        immediate_start_limit=immediate_start_limit,
    )


def _has_queued_forecast(status: JobStatus) -> bool:
    return (
        status.state is JobState.QUEUED
        and _forecast_datetime(status.scheduled_start) is not None
    )


@dataclass(frozen=True, slots=True)
class ReplacementOutcome:
    """The one allocation retained after a replacement trial."""

    site: str
    job_id: int
    replaced: bool


@dataclass(frozen=True, slots=True)
class ReplacementCandidate:
    """A site and scheduler allocation known to be currently compatible."""

    site: str
    allocation: dict[str, str]


def attempt_immediate_replacement(
    *,
    fallback_site: str,
    fallback_job_id: int,
    candidates: Sequence[ReplacementCandidate],
    submit: Callable[[ReplacementCandidate], int],
    status: Callable[[str, int], JobStatus],
    cancel: Callable[[str, int], None],
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    wall_clock: Callable[[], datetime] = lambda: datetime.now().astimezone(),
    trial_seconds: float = TRIAL_TIMEOUT_SECONDS,
    poll_seconds: float = TRIAL_POLL_SECONDS,
) -> ReplacementOutcome:
    """Try candidates sequentially and keep whichever allocation starts first."""

    _validate_replacement_arguments(
        fallback_job_id,
        trial_seconds=trial_seconds,
        poll_seconds=poll_seconds,
    )
    for candidate in candidates:
        outcome = _try_replacement_candidate(
            candidate,
            fallback_site=fallback_site,
            fallback_job_id=fallback_job_id,
            submit=submit,
            status=status,
            cancel=cancel,
            sleep=sleep,
            monotonic=monotonic,
            wall_clock=wall_clock,
            trial_seconds=trial_seconds,
            poll_seconds=poll_seconds,
        )
        if outcome is not None:
            return outcome
    return ReplacementOutcome(
        site=fallback_site,
        job_id=fallback_job_id,
        replaced=False,
    )


def _validate_replacement_arguments(
    fallback_job_id: int,
    *,
    trial_seconds: float,
    poll_seconds: float,
) -> None:
    if fallback_job_id <= 0:
        raise ValueError("fallback_job_id must be positive")
    if trial_seconds <= 0 or poll_seconds <= 0:
        raise ValueError("trial_seconds and poll_seconds must be positive")


def _try_replacement_candidate(
    candidate: ReplacementCandidate,
    *,
    fallback_site: str,
    fallback_job_id: int,
    submit: Callable[[ReplacementCandidate], int],
    status: Callable[[str, int], JobStatus],
    cancel: Callable[[str, int], None],
    sleep: Callable[[float], None],
    monotonic: Callable[[], float],
    wall_clock: Callable[[], datetime],
    trial_seconds: float,
    poll_seconds: float,
) -> ReplacementOutcome | None:
    trial_job_id = submit(candidate)
    deadline = monotonic() + trial_seconds
    while monotonic() < deadline:
        outcome, observed = _observe_replacement_candidate(
            candidate,
            trial_job_id=trial_job_id,
            fallback_site=fallback_site,
            fallback_job_id=fallback_job_id,
            status=status,
            cancel=cancel,
        )
        if outcome is not None:
            return outcome
        assert observed is not None
        if _candidate_should_stop(observed, wall_clock):
            cancel(candidate.site, trial_job_id)
            return None
        sleep(min(poll_seconds, max(0.0, deadline - monotonic())))
    cancel(candidate.site, trial_job_id)
    return None


def _observe_replacement_candidate(
    candidate: ReplacementCandidate,
    *,
    trial_job_id: int,
    fallback_site: str,
    fallback_job_id: int,
    status: Callable[[str, int], JobStatus],
    cancel: Callable[[str, int], None],
) -> tuple[ReplacementOutcome | None, JobStatus | None]:
    fallback = status(fallback_site, fallback_job_id)
    if fallback.state is JobState.RUNNING:
        cancel(candidate.site, trial_job_id)
        return ReplacementOutcome(fallback_site, fallback_job_id, False), None
    observed = status(candidate.site, trial_job_id)
    if observed.state is JobState.RUNNING:
        cancel(fallback_site, fallback_job_id)
        return ReplacementOutcome(candidate.site, trial_job_id, True), observed
    return None, observed


def _candidate_should_stop(
    observed: JobStatus,
    wall_clock: Callable[[], datetime],
) -> bool:
    return observed.state in {
        JobState.TERMINATED,
        JobState.ERROR,
        JobState.MISSING,
    } or forecast_exceeds_immediate_window(observed, now=wall_clock())


__all__ = [
    "IMMEDIATE_START_LIMIT",
    "ReplacementCandidate",
    "ReplacementOutcome",
    "QueuedReplacementDecision",
    "SHORT_TRIAL_WALLTIME_SECONDS",
    "TRIAL_POLL_SECONDS",
    "TRIAL_TIMEOUT_SECONDS",
    "attempt_immediate_replacement",
    "decide_queued_replacement",
    "forecast_exceeds_immediate_window",
    "policy_type_for",
    "should_seek_replacement",
]
