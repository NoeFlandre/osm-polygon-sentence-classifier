from datetime import UTC, datetime, timedelta

import pytest

from osm_polygon_sentence_classifier.grid5000_oar import JobState, JobStatus
from osm_polygon_sentence_classifier.grid5000_policy import (
    IMMEDIATE_START_LIMIT,
    ReplacementCandidate,
    ReplacementOutcome,
    attempt_immediate_replacement,
    policy_type_for,
    should_seek_replacement,
)


def test_policy_type_fits_a_short_job_inside_the_day_window() -> None:
    now = datetime(2026, 8, 5, 10, 0, tzinfo=UTC)

    assert policy_type_for(now, walltime_seconds=1_800) == "day"


def test_policy_type_uses_night_when_a_job_would_cross_day_end() -> None:
    now = datetime(2026, 8, 5, 18, 45, tzinfo=UTC)

    assert policy_type_for(now, walltime_seconds=1_800) == "night"


def test_distant_queued_forecast_is_eligible_for_one_trial() -> None:
    now = datetime(2026, 8, 5, 10, 0, tzinfo=UTC)
    status = JobStatus(
        job_id=1,
        state=JobState.QUEUED,
        scheduled_start="2026-08-05 19:00:00",
    )

    assert should_seek_replacement(status, now=now)
    assert timedelta(minutes=10) == IMMEDIATE_START_LIMIT


def test_running_or_unknown_forecast_is_not_replaced() -> None:
    now = datetime(2026, 8, 5, 10, 0, tzinfo=UTC)

    assert not should_seek_replacement(
        JobStatus(job_id=1, state=JobState.RUNNING), now=now
    )
    assert not should_seek_replacement(
        JobStatus(job_id=1, state=JobState.QUEUED), now=now
    )


def test_replacement_cancels_a_trial_that_reaches_its_deadline() -> None:
    clock_values = iter((0.0, 0.0, 2.0, 2.0))
    cancelled: list[tuple[str, int]] = []

    def monotonic() -> float:
        return next(clock_values)

    def status(site: str, job_id: int) -> JobStatus:
        return JobStatus(job_id=job_id, state=JobState.QUEUED)

    outcome = attempt_immediate_replacement(
        fallback_site="nancy",
        fallback_job_id=10,
        candidates=(ReplacementCandidate("grenoble", {"queue": "production"}),),
        submit=lambda _candidate: 11,
        status=status,
        cancel=lambda site, job_id: cancelled.append((site, job_id)),
        sleep=lambda _seconds: None,
        monotonic=monotonic,
        trial_seconds=1.0,
        poll_seconds=0.1,
    )

    assert outcome == ReplacementOutcome("nancy", 10, False)
    assert cancelled == [("grenoble", 11)]


def test_replacement_keeps_the_fallback_when_it_starts_first() -> None:
    cancelled: list[tuple[str, int]] = []

    def status(site: str, job_id: int) -> JobStatus:
        state = JobState.RUNNING if site == "nancy" else JobState.QUEUED
        return JobStatus(job_id=job_id, state=state)

    outcome = attempt_immediate_replacement(
        fallback_site="nancy",
        fallback_job_id=10,
        candidates=(ReplacementCandidate("grenoble", {"queue": "production"}),),
        submit=lambda _candidate: 11,
        status=status,
        cancel=lambda site, job_id: cancelled.append((site, job_id)),
        sleep=lambda _seconds: None,
        trial_seconds=1.0,
        poll_seconds=0.1,
    )

    assert outcome == ReplacementOutcome("nancy", 10, False)
    assert cancelled == [("grenoble", 11)]


def test_replacement_does_not_swallow_a_submission_error() -> None:
    def submit(_candidate: ReplacementCandidate) -> int:
        raise RuntimeError("ambiguous scheduler response")

    with pytest.raises(RuntimeError, match="ambiguous"):
        attempt_immediate_replacement(
            fallback_site="nancy",
            fallback_job_id=10,
            candidates=(ReplacementCandidate("grenoble", {"queue": "production"}),),
            submit=submit,
            status=lambda _site, _job_id: JobStatus(job_id=10, state=JobState.QUEUED),
            cancel=lambda _site, _job_id: None,
            sleep=lambda _seconds: None,
            trial_seconds=1.0,
            poll_seconds=0.1,
        )
