from datetime import UTC, datetime, timedelta

import pytest

from osm_polygon_sentence_classifier.grid5000_oar import JobState, JobStatus
from osm_polygon_sentence_classifier.grid5000_policy import (
    IMMEDIATE_START_LIMIT,
    ReplacementCandidate,
    ReplacementOutcome,
    attempt_immediate_replacement,
    decide_queued_replacement,
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


def test_queued_job_without_a_forecast_is_eligible_for_one_trial() -> None:
    now = datetime(2026, 8, 5, 10, 0, tzinfo=UTC)
    status = JobStatus(job_id=1, state=JobState.QUEUED)

    assert should_seek_replacement(status, now=now)


def test_running_job_is_not_replaced() -> None:
    now = datetime(2026, 8, 5, 10, 0, tzinfo=UTC)

    assert not should_seek_replacement(
        JobStatus(job_id=1, state=JobState.RUNNING), now=now
    )


def test_queued_replacement_decision_waits_for_an_imminent_job() -> None:
    now = datetime(2026, 8, 5, 8, 0, tzinfo=UTC)

    decision = decide_queued_replacement(
        JobStatus(
            job_id=1,
            state=JobState.QUEUED,
            scheduled_start="2026-08-05 10:05:00",
        ),
        site="grenoble",
        job_id=1,
        now=now,
        attempt_count=0,
        retry_due=True,
        max_attempts=3,
    )

    assert decision.action == "wait"


def test_queued_replacement_decision_starts_the_next_due_round() -> None:
    now = datetime(2026, 8, 5, 10, 0, tzinfo=UTC)

    decision = decide_queued_replacement(
        JobStatus(job_id=1, state=JobState.QUEUED),
        site="grenoble",
        job_id=1,
        now=now,
        attempt_count=1,
        retry_due=True,
        max_attempts=3,
    )

    assert decision.action == "replace"
    assert decision.attempt_count == 2
    assert decision.attempt_timestamp == now.isoformat()


def test_queued_replacement_decision_fails_after_the_bounded_rounds() -> None:
    now = datetime(2026, 8, 5, 19, 0, tzinfo=UTC)

    decision = decide_queued_replacement(
        JobStatus(
            job_id=1,
            state=JobState.QUEUED,
            scheduled_start="2026-08-06 08:02:02",
        ),
        site="grenoble",
        job_id=1,
        now=now,
        attempt_count=3,
        retry_due=False,
        max_attempts=3,
    )

    assert decision.action == "fail"
    assert decision.message == (
        "grenoble job 1 remained queued with scheduled start 2026-08-06 08:02:02 "
        "after 3 replacement rounds"
    )


def test_replacement_cancels_a_trial_that_reaches_its_deadline() -> None:
    elapsed = 0.0
    cancelled: list[tuple[str, int]] = []
    observed: list[tuple[str, int]] = []

    def monotonic() -> float:
        return elapsed

    def sleep(seconds: float) -> None:
        nonlocal elapsed
        elapsed += seconds

    def status(site: str, job_id: int) -> JobStatus:
        observed.append((site, job_id))
        return JobStatus(job_id=job_id, state=JobState.QUEUED)

    outcome = attempt_immediate_replacement(
        fallback_site="nancy",
        fallback_job_id=10,
        candidates=(ReplacementCandidate("grenoble", {"queue": "production"}),),
        submit=lambda _candidate: 11,
        status=status,
        cancel=lambda site, job_id: cancelled.append((site, job_id)),
        sleep=sleep,
        monotonic=monotonic,
        trial_seconds=1.0,
        poll_seconds=0.5,
    )

    assert outcome == ReplacementOutcome("nancy", 10, False)
    assert cancelled == [("grenoble", 11)]
    assert observed == [("nancy", 10), ("grenoble", 11)] * 2


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
