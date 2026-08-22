from datetime import UTC, datetime, timedelta, timezone, tzinfo
from typing import NoReturn

import pytest

import osm_polygon_sentence_classifier.grid5000_policy as grid5000_policy
from osm_polygon_sentence_classifier.grid5000_oar import JobState, JobStatus
from osm_polygon_sentence_classifier.grid5000_policy import (
    GRID5000_TIMEZONE,
    IMMEDIATE_START_LIMIT,
    ReplacementCandidate,
    ReplacementOutcome,
    _weekday_policy_type,
    attempt_immediate_replacement,
    decide_queued_replacement,
    policy_type_for,
    should_seek_replacement,
)


def test_policy_type_fits_a_short_job_inside_the_day_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime(2026, 8, 5, 10, 0, tzinfo=UTC)

    assert policy_type_for(now, walltime_seconds=1_800) == "day"

    import osm_polygon_sentence_classifier.grid5000_policy as policy_module

    target_timezone = timezone(timedelta(hours=10))
    observed: list[datetime] = []

    def record_local(local: datetime, duration: timedelta) -> str:
        observed.append(local)
        assert duration == timedelta(seconds=1)
        return "day"

    monkeypatch.setattr(policy_module, "GRID5000_TIMEZONE", target_timezone)
    monkeypatch.setattr(policy_module, "_weekday_policy_type", record_local)
    policy_now = datetime(2026, 8, 3, 12, tzinfo=UTC)

    assert policy_module.policy_type_for(policy_now, walltime_seconds=1) == "day"
    assert observed == [policy_now.astimezone(target_timezone)]


def test_policy_type_uses_night_when_a_job_would_cross_day_end() -> None:
    now = datetime(2026, 8, 5, 18, 45, tzinfo=UTC)

    assert policy_type_for(now, walltime_seconds=1_800) == "night"


def test_policy_type_rejects_a_timezone_without_an_offset_exactly() -> None:
    class MissingOffset(tzinfo):
        def utcoffset(self, _value: datetime | None) -> timedelta | None:
            return None

        def dst(self, _value: datetime | None) -> timedelta | None:
            return None

    with pytest.raises(ValueError, match="^now must be timezone-aware$") as error:
        policy_type_for(
            datetime(2026, 8, 5, 10, tzinfo=MissingOffset()),
            walltime_seconds=1,
        )

    assert str(error.value) == "now must be timezone-aware"


@pytest.mark.parametrize("walltime_seconds", [0, -1])
def test_policy_type_rejects_non_positive_walltime_exactly(
    walltime_seconds: int,
) -> None:
    with pytest.raises(
        ValueError, match="^walltime_seconds must be positive$"
    ) as error:
        policy_type_for(
            datetime(2026, 8, 5, 10, tzinfo=UTC),
            walltime_seconds=walltime_seconds,
        )

    assert str(error.value) == "walltime_seconds must be positive"


def test_policy_type_uses_the_configured_grid5000_timezone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import osm_polygon_sentence_classifier.grid5000_policy as policy_module

    target_timezone = timezone(timedelta(hours=10))
    observed: list[datetime] = []

    def record_local(local: datetime, duration: timedelta) -> str:
        observed.append(local)
        assert duration == timedelta(seconds=1)
        return "day"

    monkeypatch.setattr(policy_module, "GRID5000_TIMEZONE", target_timezone)
    monkeypatch.setattr(policy_module, "_weekday_policy_type", record_local)
    now = datetime(2026, 8, 3, 12, tzinfo=UTC)

    assert policy_module.policy_type_for(now, walltime_seconds=1) == "day"
    assert observed == [now.astimezone(target_timezone)]


def test_policy_type_applies_the_configured_timezone_to_window_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_timezone = timezone(timedelta(hours=14))
    monkeypatch.setattr(grid5000_policy, "GRID5000_TIMEZONE", target_timezone)

    assert (
        grid5000_policy.policy_type_for(
            datetime(2026, 8, 3, 7, tzinfo=UTC),
            walltime_seconds=1,
        )
        == "night"
    )


def test_policy_type_treats_saturday_as_night() -> None:
    now = datetime(2026, 8, 8, 10, tzinfo=UTC)

    assert policy_type_for(now, walltime_seconds=1) == "night"


def test_replacement_argument_validation_accepts_the_smallest_valid_values() -> None:
    assert (
        grid5000_policy._validate_replacement_arguments(
            1,
            trial_seconds=1,
            poll_seconds=1,
        )
        is None
    )


def test_replacement_argument_validation_reports_invalid_job_ids_exactly() -> None:
    with pytest.raises(ValueError, match="^fallback_job_id must be positive$") as error:
        grid5000_policy._validate_replacement_arguments(
            0,
            trial_seconds=1,
            poll_seconds=1,
        )

    assert str(error.value) == "fallback_job_id must be positive"


@pytest.mark.parametrize(
    ("trial_seconds", "poll_seconds"),
    [(0, 1), (1, 0)],
)
def test_replacement_argument_validation_reports_non_positive_durations_exactly(
    trial_seconds: float,
    poll_seconds: float,
) -> None:
    with pytest.raises(
        ValueError, match="^trial_seconds and poll_seconds must be positive$"
    ) as error:
        grid5000_policy._validate_replacement_arguments(
            1,
            trial_seconds=trial_seconds,
            poll_seconds=poll_seconds,
        )

    assert str(error.value) == "trial_seconds and poll_seconds must be positive"


@pytest.mark.parametrize(
    ("local", "duration", "expected"),
    [
        (datetime(2026, 8, 5, 9, 30), timedelta(seconds=1), "day"),
        (datetime(2026, 8, 5, 9, 0, 30), timedelta(seconds=1), "day"),
        (datetime(2026, 8, 5, 9, 0), timedelta(seconds=1), "day"),
        (datetime(2026, 8, 5, 9, 0), timedelta(microseconds=1), "day"),
        (datetime(2026, 8, 5, 19, 30), timedelta(seconds=1), "night"),
        (datetime(2026, 8, 5, 19, 0, 30), timedelta(seconds=1), "night"),
        (datetime(2026, 8, 5, 19, 0), timedelta(microseconds=1), "night"),
        (datetime(2026, 8, 5, 19, 0, 0, 500_000), timedelta(microseconds=1), "night"),
        (datetime(2026, 8, 5, 8, 59), timedelta(minutes=1), "night"),
        (datetime(2026, 8, 5, 18, 59), timedelta(minutes=1), "day"),
        (datetime(2026, 8, 5, 8, 0), timedelta(hours=2), "day"),
        (datetime(2026, 8, 5, 18, 0), timedelta(hours=2), "night"),
    ],
)
def test_weekday_policy_uses_the_complete_boundary_and_duration_rules(
    local: datetime,
    duration: timedelta,
    expected: str,
) -> None:
    assert (
        _weekday_policy_type(local.replace(tzinfo=GRID5000_TIMEZONE), duration)
        == expected
    )


def test_weekday_policy_zeroes_subminute_values_before_day_start() -> None:
    local = datetime(2026, 8, 5, 8, 59, 59, 500_000)

    assert (
        _weekday_policy_type(
            local.replace(tzinfo=GRID5000_TIMEZONE), timedelta(seconds=1)
        )
        == "day"
    )


def test_weekday_policy_treats_exact_day_start_as_day_even_for_zero_duration() -> None:
    local = datetime(2026, 8, 5, 9, 0)

    assert (
        _weekday_policy_type(local.replace(tzinfo=GRID5000_TIMEZONE), timedelta(0))
        == "day"
    )


def test_weekday_policy_zeroes_subminute_values_before_day_end() -> None:
    local = datetime(2026, 8, 5, 18, 59, 59, 500_000)

    assert (
        _weekday_policy_type(
            local.replace(tzinfo=GRID5000_TIMEZONE), timedelta(seconds=1)
        )
        == "night"
    )


def test_weekday_policy_treats_exact_day_end_as_night_even_for_zero_duration() -> None:
    local = datetime(2026, 8, 5, 19, 0)

    assert (
        _weekday_policy_type(local.replace(tzinfo=GRID5000_TIMEZONE), timedelta(0))
        == "night"
    )


def test_distant_queued_forecast_is_eligible_for_one_trial() -> None:
    now = datetime(2026, 8, 5, 10, 0, tzinfo=UTC)
    status = JobStatus(
        job_id=1,
        state=JobState.QUEUED,
        scheduled_start="2026-08-05 19:00:00",
    )

    assert should_seek_replacement(status, now=now)
    assert timedelta(minutes=10) == IMMEDIATE_START_LIMIT


def test_should_seek_replacement_reports_invalid_time_arguments_exactly() -> None:
    status = JobStatus(job_id=1, state=JobState.QUEUED)

    with pytest.raises(ValueError, match="^now must be timezone-aware$") as error:
        should_seek_replacement(status, now=datetime(2026, 8, 5, 10, 0))
    assert str(error.value) == "now must be timezone-aware"

    class MissingOffset(tzinfo):
        def utcoffset(self, _value: datetime | None) -> timedelta | None:
            return None

        def dst(self, _value: datetime | None) -> timedelta | None:
            return None

    with pytest.raises(ValueError, match="^now must be timezone-aware$") as error:
        should_seek_replacement(
            status,
            now=datetime(2026, 8, 5, 10, 0, tzinfo=MissingOffset()),
        )
    assert str(error.value) == "now must be timezone-aware"

    with pytest.raises(
        ValueError, match="^immediate_start_limit must be positive$"
    ) as error:
        should_seek_replacement(
            status,
            now=datetime(2026, 8, 5, 10, 0, tzinfo=UTC),
            immediate_start_limit=timedelta(0),
        )
    assert str(error.value) == "immediate_start_limit must be positive"


def test_queued_job_without_a_forecast_is_eligible_for_one_trial() -> None:
    now = datetime(2026, 8, 5, 10, 0, tzinfo=UTC)
    status = JobStatus(job_id=1, state=JobState.QUEUED)

    assert should_seek_replacement(status, now=now)


def test_should_seek_replacement_uses_the_configured_grid5000_timezone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_timezone = timezone(timedelta(hours=14))
    monkeypatch.setattr(grid5000_policy, "GRID5000_TIMEZONE", target_timezone)
    status = JobStatus(
        job_id=1,
        state=JobState.QUEUED,
        scheduled_start="2026-08-03 21:05:00",
    )

    assert not should_seek_replacement(
        status,
        now=datetime(2026, 8, 3, 7, tzinfo=UTC),
    )


def test_past_due_queued_forecast_is_eligible_for_replacement() -> None:
    now = datetime(2026, 8, 5, 10, 0, tzinfo=UTC)
    status = JobStatus(
        job_id=1,
        state=JobState.QUEUED,
        scheduled_start="2026-08-05 09:55:00",
    )

    assert should_seek_replacement(status, now=now)


@pytest.mark.parametrize(
    ("scheduled_start", "expected"),
    [
        ("2026-08-05 10:00:00", True),
        ("2026-08-05 10:10:00", False),
    ],
)
def test_replacement_window_treats_both_endpoints_as_not_immediate(
    scheduled_start: str, expected: bool
) -> None:
    assert (
        should_seek_replacement(
            JobStatus(job_id=1, state=JobState.QUEUED, scheduled_start=scheduled_start),
            now=datetime(2026, 8, 5, 8, 0, tzinfo=UTC),
        )
        is expected
    )


def test_forecast_helper_forwards_all_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = JobStatus(
        job_id=1,
        state=JobState.QUEUED,
        scheduled_start="2026-08-05 19:00:00",
    )
    now = datetime(2026, 8, 5, 10, 0, tzinfo=UTC)
    limit = timedelta(minutes=7)
    observed: list[tuple[JobStatus, datetime, timedelta]] = []

    def replacement(
        candidate: JobStatus,
        *,
        now: datetime,
        immediate_start_limit: timedelta,
    ) -> bool:
        observed.append((candidate, now, immediate_start_limit))
        return True

    monkeypatch.setattr(grid5000_policy, "should_seek_replacement", replacement)

    assert (
        grid5000_policy.forecast_exceeds_immediate_window(
            status,
            now=now,
            immediate_start_limit=limit,
        )
        is True
    )
    assert observed == [(status, now, limit)]


def test_has_queued_forecast_requires_both_queue_state_and_valid_forecast() -> None:
    assert grid5000_policy._has_queued_forecast(
        JobStatus(
            job_id=1,
            state=JobState.QUEUED,
            scheduled_start="2026-08-05 19:00:00",
        )
    )
    assert not grid5000_policy._has_queued_forecast(
        JobStatus(job_id=1, state=JobState.QUEUED)
    )
    assert not grid5000_policy._has_queued_forecast(
        JobStatus(
            job_id=1,
            state=JobState.RUNNING,
            scheduled_start="2026-08-05 19:00:00",
        )
    )


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


def test_replacement_keeps_a_candidate_that_starts_first() -> None:
    cancelled: list[tuple[str, int]] = []

    def status(site: str, job_id: int) -> JobStatus:
        state = JobState.RUNNING if site == "grenoble" else JobState.QUEUED
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

    assert outcome == ReplacementOutcome("grenoble", 11, True)
    assert cancelled == [("nancy", 10)]


def test_replacement_stops_a_candidate_with_a_late_forecast() -> None:
    cancelled: list[tuple[str, int]] = []
    now = datetime(2026, 8, 5, 10, tzinfo=UTC)

    def fail_sleep(_seconds: float) -> NoReturn:
        raise AssertionError("late candidate should stop immediately")

    outcome = attempt_immediate_replacement(
        fallback_site="nancy",
        fallback_job_id=10,
        candidates=(ReplacementCandidate("grenoble", {"queue": "production"}),),
        submit=lambda _candidate: 11,
        status=lambda _site, job_id: JobStatus(
            job_id=job_id,
            state=JobState.QUEUED,
            scheduled_start="2026-08-05 19:00:00",
        ),
        cancel=lambda site, job_id: cancelled.append((site, job_id)),
        sleep=fail_sleep,
        wall_clock=lambda: now,
        trial_seconds=1.0,
        poll_seconds=0.1,
    )

    assert outcome == ReplacementOutcome("nancy", 10, False)
    assert cancelled == [("grenoble", 11)]


def test_replacement_uses_only_the_remaining_trial_time_for_sleep() -> None:
    elapsed = 0.0
    sleeps: list[float] = []

    def monotonic() -> float:
        return elapsed

    def sleep(seconds: float) -> None:
        nonlocal elapsed
        sleeps.append(seconds)
        elapsed += seconds

    outcome = attempt_immediate_replacement(
        fallback_site="nancy",
        fallback_job_id=10,
        candidates=(ReplacementCandidate("grenoble", {"queue": "production"}),),
        submit=lambda _candidate: 11,
        status=lambda _site, job_id: JobStatus(job_id=job_id, state=JobState.QUEUED),
        cancel=lambda _site, _job_id: None,
        sleep=sleep,
        monotonic=monotonic,
        trial_seconds=1.5,
        poll_seconds=1.0,
    )

    assert outcome == ReplacementOutcome("nancy", 10, False)
    assert sleeps == [1.0, 0.5]
    assert elapsed == 1.5


def test_candidate_stop_uses_the_supplied_wall_clock() -> None:
    observed = JobStatus(
        job_id=11,
        state=JobState.QUEUED,
        scheduled_start="2026-08-05 19:00:00",
    )

    assert grid5000_policy._candidate_should_stop(
        observed,
        lambda: datetime(2026, 8, 5, 10, tzinfo=UTC),
    )


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
