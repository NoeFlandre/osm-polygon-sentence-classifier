from collections.abc import Mapping
from pathlib import Path

import pytest

import osm_polygon_sentence_classifier.grid5000_checkpointing as checkpointing
from osm_polygon_sentence_classifier.grid5000 import Grid5000ExecutionError
from osm_polygon_sentence_classifier.grid5000_checkpointing import (
    CheckpointProbeError,
    probe_complete_checkpoint,
)

type ProbeOutcome = bool | Exception


class _ProbeRemote:
    def __init__(self, outcomes: list[ProbeOutcome]) -> None:
        self.outcomes = outcomes
        self.calls: list[tuple[str, str | Path, Mapping[str, object], bool]] = []

    def has_complete_checkpoint(
        self,
        run_id: str,
        *,
        output_subdirectory: str | Path,
        identity: Mapping[str, object],
        allow_failed_status: bool,
    ) -> bool:
        self.calls.append((run_id, output_subdirectory, identity, allow_failed_status))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_checkpoint_probe_retries_transient_error_with_bounded_delay() -> None:
    remote = _ProbeRemote(
        [
            Grid5000ExecutionError("connection timed out"),
            True,
        ]
    )
    messages: list[str] = []
    sleeps: list[float] = []
    identity = {"run_id": "a" * 20}

    result = probe_complete_checkpoint(
        remote,
        run_id="a" * 20,
        output_subdirectory=Path("models/landuse"),
        identity=identity,
        allow_failed_status=True,
        site="lille",
        job_id=123,
        poll_seconds=30.0,
        emit=messages.append,
        sleep=sleeps.append,
    )

    assert result is True
    assert remote.calls == [
        ("a" * 20, Path("models/landuse"), identity, True),
        ("a" * 20, Path("models/landuse"), identity, True),
    ]
    assert messages == [
        "lille job 123: checkpoint probe failed (attempt 1/3); retrying in 5s"
    ]
    assert sleeps == [5.0]


def test_checkpoint_probe_raises_after_three_transient_errors() -> None:
    remote = _ProbeRemote(
        [
            Grid5000ExecutionError("connection timed out"),
            Grid5000ExecutionError("connection timed out"),
            Grid5000ExecutionError("connection timed out"),
        ]
    )
    sleeps: list[float] = []

    with pytest.raises(
        CheckpointProbeError,
        match="checkpoint availability could not be verified: connection timed out",
    ):
        probe_complete_checkpoint(
            remote,
            run_id="b" * 20,
            output_subdirectory="models/landuse",
            identity={"run_id": "b" * 20},
            allow_failed_status=False,
            site="nancy",
            job_id=456,
            poll_seconds=30.0,
            emit=lambda _message: None,
            sleep=sleeps.append,
        )

    assert len(remote.calls) == 3
    assert sleeps == [5.0, 5.0]


def test_checkpoint_probe_does_not_retry_unexpected_errors() -> None:
    remote = _ProbeRemote([ValueError("malformed checkpoint marker")])
    sleeps: list[float] = []

    with pytest.raises(
        CheckpointProbeError,
        match="checkpoint availability could not be verified: malformed checkpoint marker",
    ):
        probe_complete_checkpoint(
            remote,
            run_id="c" * 20,
            output_subdirectory="models/landuse",
            identity={"run_id": "c" * 20},
            allow_failed_status=False,
            site="grenoble",
            job_id=789,
            poll_seconds=30.0,
            emit=lambda _message: None,
            sleep=sleeps.append,
        )

    assert len(remote.calls) == 1
    assert sleeps == []


def test_checkpoint_probe_detail_suffix_is_empty_without_an_error() -> None:
    assert checkpointing._detail_suffix(None) == ""
    assert checkpointing._detail_suffix(ValueError("   ")) == ""


def test_checkpoint_probe_detail_suffix_is_bounded_to_240_characters() -> None:
    detail = "x" * 241

    assert checkpointing._detail_suffix(ValueError(detail)) == ": " + "x" * 240


def test_checkpoint_probe_with_zero_attempts_has_no_exception_cause() -> None:
    remote = _ProbeRemote([])

    with pytest.raises(CheckpointProbeError) as caught:
        probe_complete_checkpoint(
            remote,
            run_id="d" * 20,
            output_subdirectory="models/landuse",
            identity={"run_id": "d" * 20},
            allow_failed_status=False,
            site="nancy",
            job_id=321,
            poll_seconds=30.0,
            emit=lambda _message: None,
            sleep=lambda _seconds: None,
            attempts=0,
        )

    assert caught.value.__cause__ is None


def test_checkpoint_probe_does_not_exceed_the_attempt_budget() -> None:
    remote = _ProbeRemote([Grid5000ExecutionError("connection timed out")])

    with pytest.raises(CheckpointProbeError):
        probe_complete_checkpoint(
            remote,
            run_id="e" * 20,
            output_subdirectory="models/landuse",
            identity={"run_id": "e" * 20},
            allow_failed_status=False,
            site="nancy",
            job_id=654,
            poll_seconds=30.0,
            emit=lambda _message: None,
            sleep=lambda _seconds: None,
            attempts=1,
        )

    assert len(remote.calls) == 1
