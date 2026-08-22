import json
from collections.abc import Sequence
from typing import Any, cast
from zoneinfo import ZoneInfo

import pytest

import osm_polygon_sentence_classifier.grid5000_oar as oar_module
from osm_polygon_sentence_classifier.grid5000 import (
    CommandResult,
    Grid5000ExecutionError,
)
from osm_polygon_sentence_classifier.grid5000_oar import (
    JobState,
    JobStatus,
    OarClient,
    OarError,
    _parse_exit_code,
    _parse_scheduled_start,
    _parse_status,
    _parse_walltime,
    _parse_walltime_string,
    format_job_status,
    parse_job_id,
)


class _FakeRemote:
    def __init__(self, responses: Sequence[CommandResult]) -> None:
        self.responses = list(responses)
        self.commands: list[str] = []

    def raw(self, command: str) -> CommandResult:
        self.commands.append(command)
        return self.responses.pop(0)

    def run(self, command: str) -> CommandResult:
        self.commands.append(command)
        return self.responses.pop(0)


class _RaisingRemote:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def raw(self, _: str) -> CommandResult:
        raise self.error

    def run(self, _: str) -> CommandResult:
        raise self.error


def test_parse_job_id_rejects_ambiguous_scheduler_output() -> None:
    assert parse_job_id("OAR_JOB_ID=123\n") == 123
    with pytest.raises(
        OarError,
        match="^submission did not return one job ID$",
    ):
        parse_job_id("OAR_JOB_ID=123\nOAR_JOB_ID=124\n")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (1, 1),
        (0, None),
        (-1, None),
        (True, None),
        ("01:02:03", 3_723),
        ("00:00:01", 1),
        ("1", 1),
        ("0", None),
        ("", None),
    ],
)
def test_walltime_parser_preserves_units_and_positive_boundaries(
    value: object,
    expected: int | None,
) -> None:
    assert _parse_walltime(value) == expected


def test_walltime_string_parser_rejects_malformed_fields() -> None:
    assert _parse_walltime_string("1:2:03") is None
    assert _parse_walltime_string("1:60:00") == 7_200
    assert _parse_walltime_string("1:00:60") == 3_660


def test_scheduled_start_parser_converts_positive_epoch_and_rejects_zero() -> None:
    assert _parse_scheduled_start(1) == "1970-01-01 01:00:01"
    assert _parse_scheduled_start(0) is None
    assert _parse_scheduled_start(-1) is None
    assert _parse_scheduled_start(True) is None


def test_scheduled_start_uses_the_grid5000_timezone(monkeypatch) -> None:
    monkeypatch.setattr(oar_module, "GRID5000_TIMEZONE", ZoneInfo("UTC"))

    assert oar_module._parse_scheduled_start(1) == "1970-01-01 00:00:01"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0),
        (-7, -7),
        ("0", 0),
        ("-7", -7),
        ("+7", None),
        ("7x", None),
        (True, None),
        (1.0, None),
    ],
)
def test_exit_code_parser_accepts_signed_decimal_values_only(
    value: object,
    expected: int | None,
) -> None:
    assert _parse_exit_code(value) == expected


def test_status_parser_preserves_all_scheduler_fields() -> None:
    status = OarClient(
        _FakeRemote(
            [
                CommandResult(
                    returncode=0,
                    stdout=json.dumps(
                        {
                            "42": {
                                "state": "Finishing",
                                "exit_code": "-7",
                                "message": "wrapping up",
                                "assigned_network_address": 123,
                                "scheduled_start": 1,
                                "walltime": 1,
                            }
                        }
                    ),
                )
            ]
        )
    ).status(42)

    assert status == JobStatus(
        job_id=42,
        state=JobState.FINISHING,
        exit_code=-7,
        message="wrapping up",
        node="123",
        scheduled_start="1970-01-01 01:00:01",
        walltime_seconds=1,
    )


def test_status_parser_preserves_missing_message_and_node_defaults() -> None:
    status = _parse_status(42, {"state": "Running"})

    assert status.message == ""
    assert status.node is None


class _DefaultAwareMapping(dict[str, Any]):
    def get(self, key: object, default: object = None) -> object:
        if key == "state" and key not in self:
            return "Running" if default == "" else "unknown"
        return super().get(key, default)


def test_status_parser_passes_the_empty_state_default_to_mapping() -> None:
    assert _parse_status(42, _DefaultAwareMapping()).state is JobState.RUNNING


def test_status_parser_uses_inline_record_when_job_key_is_absent() -> None:
    remote = _FakeRemote(
        [
            CommandResult(
                returncode=0,
                stdout=json.dumps({"state": "Running"}),
            )
        ]
    )

    assert OarClient(remote).status(42).state is JobState.RUNNING


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ([], "^OAR status JSON must be an object$"),
        ({"42": []}, "^OAR status record must be an object$"),
        ({"42": {"state": "unknown"}}, "^unsupported OAR job state$"),
        ({"42": {}}, "^unsupported OAR job state$"),
    ],
)
def test_status_parser_rejects_malformed_or_unknown_records(
    payload: object,
    message: str,
) -> None:
    remote = _FakeRemote([CommandResult(returncode=0, stdout=json.dumps(payload))])

    with pytest.raises(OarError, match=message):
        OarClient(remote).status(42)


def test_format_job_status_covers_queued_running_and_terminal_states() -> None:
    assert format_job_status(JobStatus(1, JobState.QUEUED)) == (
        "queued; scheduler has no start-time prediction"
    )
    assert (
        format_job_status(
            JobStatus(1, JobState.QUEUED, scheduled_start="2026-08-05 19:00:00")
        )
        == "queued; scheduled start 2026-08-05 19:00:00 Europe/Paris"
    )
    assert format_job_status(JobStatus(1, JobState.RUNNING)) == "running"
    assert format_job_status(JobStatus(1, JobState.RUNNING, node="gpu-1")) == (
        "running on gpu-1"
    )
    assert format_job_status(JobStatus(1, JobState.TERMINATED)) == "terminated"
    assert format_job_status(JobStatus(1, JobState.ERROR, exit_code=256)) == (
        "error (exit 256)"
    )
    assert format_job_status(JobStatus(1, JobState.FINISHING)) == "finishing"


@pytest.mark.parametrize("command", [(), ("oarsub", ""), ("oarsub", 1)])
def test_submit_rejects_empty_or_non_text_commands(command: object) -> None:
    with pytest.raises(OarError, match="^submission command cannot be empty$"):
        OarClient(cast(Any, _FakeRemote([]))).submit(cast(Any, command))


@pytest.mark.parametrize("job_id", [0, -1, True, False, "1", None])
def test_client_rejects_non_positive_or_non_integer_job_ids(job_id: object) -> None:
    with pytest.raises(ValueError, match="^job_id must be positive$"):
        OarClient(cast(Any, _FakeRemote([]))).status(cast(Any, job_id))


def test_submit_wraps_unexpected_remote_errors_with_the_original_cause() -> None:
    error = RuntimeError("transport")

    with pytest.raises(OarError, match="^OAR submission failed$") as caught:
        OarClient(cast(Any, _RaisingRemote(error))).submit(("oarsub",))

    assert caught.value.__cause__ is error


def test_status_reports_remote_and_json_errors_with_causes() -> None:
    remote_error = _FakeRemote([CommandResult(returncode=4)])
    with pytest.raises(OarError, match="^OAR status failed with exit code 4$"):
        OarClient(remote_error).status(42)

    invalid_json = _FakeRemote([CommandResult(returncode=0, stdout="{")])
    with pytest.raises(OarError, match="^invalid OAR status JSON$") as caught:
        OarClient(invalid_json).status(42)
    assert isinstance(caught.value.__cause__, json.JSONDecodeError)


def test_cancel_wraps_scheduler_failures_with_the_original_cause() -> None:
    error = Grid5000ExecutionError("transport")

    with pytest.raises(OarError, match="^OAR cancellation failed$") as caught:
        OarClient(cast(Any, _RaisingRemote(error))).cancel(42)

    assert caught.value.__cause__ is error


@pytest.mark.parametrize(
    ("result", "message"),
    [
        (CommandResult(returncode=4), "^user OAR status failed with exit code 4$"),
        (
            CommandResult(returncode=0, stdout="{"),
            "^invalid user OAR status JSON$",
        ),
        (
            CommandResult(returncode=0, stdout="[]"),
            "^user OAR status JSON must be an object$",
        ),
    ],
)
def test_user_job_ids_reports_scheduler_response_errors(
    result: CommandResult,
    message: str,
) -> None:
    with pytest.raises(OarError, match=message):
        OarClient(_FakeRemote([result])).user_job_ids()


def test_status_normalizes_oar_json_and_missing_job() -> None:
    payload = {
        "42": {
            "state": "Waiting",
            "exit_code": None,
            "message": "queued",
            "assigned_network_address": None,
            "scheduled_start": "2026-08-05 19:00:00",
            "walltime": "00:30:00",
        }
    }
    remote = _FakeRemote(
        [
            CommandResult(returncode=0, stdout=json.dumps(payload)),
            CommandResult(returncode=6),
        ]
    )
    client = OarClient(remote)

    status = client.status(42)

    assert status.state is JobState.QUEUED
    assert status.scheduled_start == "2026-08-05 19:00:00"
    assert status.walltime_seconds == 1_800
    missing = client.status(42)
    assert missing.state is JobState.MISSING
    assert missing.job_id == 42


def test_status_accepts_job_id_one() -> None:
    status = OarClient(_FakeRemote([CommandResult(returncode=6)])).status(1)

    assert status.job_id == 1
    assert status.state is JobState.MISSING


@pytest.mark.parametrize(
    ("walltime", "expected"),
    [("3600", 3_600), ("0", None), ("not-a-duration", None), (0, None), (True, None)],
)
def test_status_parses_or_rejects_alternate_walltime_values(
    walltime: object,
    expected: int | None,
) -> None:
    remote = _FakeRemote(
        [
            CommandResult(
                returncode=0,
                stdout=json.dumps({"42": {"state": "Running", "walltime": walltime}}),
            )
        ]
    )

    assert OarClient(remote).status(42).walltime_seconds == expected


def test_submit_and_cancel_use_one_fixed_shell_command_each() -> None:
    remote = _FakeRemote(
        [
            CommandResult(returncode=0, stdout="OAR_JOB_ID=99\n"),
            CommandResult(returncode=0),
        ]
    )
    client = OarClient(remote)

    job_id = client.submit(("oarsub", "-q", "production", "payload with spaces"))
    client.cancel(job_id)

    assert job_id == 99
    assert remote.commands == [
        "oarsub -q production 'payload with spaces'",
        "oardel 99",
    ]


def test_user_jobs_returns_ids_without_inventing_scheduler_state() -> None:
    remote = _FakeRemote(
        [
            CommandResult(
                returncode=0,
                stdout=json.dumps({"11": {"state": "Waiting"}, "12": {}}),
            )
        ]
    )

    assert OarClient(remote).user_job_ids() == (11, 12)
