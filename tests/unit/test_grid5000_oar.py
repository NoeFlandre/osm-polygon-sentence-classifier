import json
from collections.abc import Sequence

import pytest

from osm_polygon_sentence_classifier.grid5000 import CommandResult
from osm_polygon_sentence_classifier.grid5000_oar import (
    JobState,
    OarClient,
    OarError,
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


def test_parse_job_id_rejects_ambiguous_scheduler_output() -> None:
    assert parse_job_id("OAR_JOB_ID=123\n") == 123
    with pytest.raises(OarError, match="one job ID"):
        parse_job_id("OAR_JOB_ID=123\nOAR_JOB_ID=124\n")


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
    assert client.status(42).state is JobState.MISSING


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
