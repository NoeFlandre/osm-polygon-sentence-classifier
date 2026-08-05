"""Grid'5000 policy-window and bounded short-job replacement rules."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta

from .grid5000_oar import GRID5000_TIMEZONE, JobState, JobStatus

IMMEDIATE_START_LIMIT = timedelta(minutes=10)
SHORT_TRIAL_WALLTIME_SECONDS = 20 * 60
TRIAL_TIMEOUT_SECONDS = 20 * 60
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
    if local.weekday() < 5:
        duration = timedelta(seconds=walltime_seconds)
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
    """Return whether a queued fallback forecast exceeds the short exception."""

    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")
    if immediate_start_limit <= timedelta(0):
        raise ValueError("immediate_start_limit must be positive")
    if status.state is not JobState.QUEUED:
        return False
    forecast = _forecast_datetime(status.scheduled_start)
    if forecast is None:
        return False
    return forecast > now.astimezone(GRID5000_TIMEZONE) + immediate_start_limit


def forecast_exceeds_immediate_window(
    status: JobStatus,
    *,
    now: datetime,
    immediate_start_limit: timedelta = IMMEDIATE_START_LIMIT,
) -> bool:
    """Return whether a replacement trial is no longer policy-immediate."""

    return should_seek_replacement(
        status,
        now=now,
        immediate_start_limit=immediate_start_limit,
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
    """Try candidates one at a time and adopt only a visibly running trial."""

    if fallback_job_id <= 0:
        raise ValueError("fallback_job_id must be positive")
    if trial_seconds <= 0 or poll_seconds <= 0:
        raise ValueError("trial_seconds and poll_seconds must be positive")
    for candidate in candidates:
        trial_job_id = submit(candidate)
        deadline = monotonic() + trial_seconds
        while monotonic() < deadline:
            fallback = status(fallback_site, fallback_job_id)
            if fallback.state is JobState.RUNNING:
                cancel(candidate.site, trial_job_id)
                return ReplacementOutcome(
                    site=fallback_site,
                    job_id=fallback_job_id,
                    replaced=False,
                )
            observed = status(candidate.site, trial_job_id)
            if observed.state is JobState.RUNNING:
                cancel(fallback_site, fallback_job_id)
                return ReplacementOutcome(
                    site=candidate.site,
                    job_id=trial_job_id,
                    replaced=True,
                )
            if observed.state in {
                JobState.TERMINATED,
                JobState.ERROR,
                JobState.MISSING,
            } or forecast_exceeds_immediate_window(
                observed,
                now=wall_clock(),
            ):
                cancel(candidate.site, trial_job_id)
                break
            sleep(min(poll_seconds, max(0.0, deadline - monotonic())))
        else:
            cancel(candidate.site, trial_job_id)
    return ReplacementOutcome(
        site=fallback_site,
        job_id=fallback_job_id,
        replaced=False,
    )


__all__ = [
    "IMMEDIATE_START_LIMIT",
    "ReplacementCandidate",
    "ReplacementOutcome",
    "SHORT_TRIAL_WALLTIME_SECONDS",
    "TRIAL_POLL_SECONDS",
    "TRIAL_TIMEOUT_SECONDS",
    "attempt_immediate_replacement",
    "forecast_exceeds_immediate_window",
    "policy_type_for",
    "should_seek_replacement",
]
