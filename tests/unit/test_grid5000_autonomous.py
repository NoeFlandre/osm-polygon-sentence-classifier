import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from osm_polygon_sentence_classifier.checkpoint_hub import PublishedCheckpoint
from osm_polygon_sentence_classifier.grid5000 import (
    CommandResult,
    Grid5000ExecutionError,
    Grid5000Plan,
    Grid5000RunIdentity,
)
from osm_polygon_sentence_classifier.grid5000_autonomous import (
    MAX_REPLACEMENT_ATTEMPTS,
    REPLACEMENT_RETRY_INTERVAL,
    AutonomousRunConfig,
    AutonomousRunController,
    AutonomousRunError,
    _continuation_facts,
    _replacement_attempt_count_for_job,
)
from osm_polygon_sentence_classifier.grid5000_oar import JobState, JobStatus
from osm_polygon_sentence_classifier.grid5000_remote import RemotePreparationResult
from osm_polygon_sentence_classifier.grid5000_sites import (
    GpuResource,
    SiteProbe,
    SiteRequirements,
)
from osm_polygon_sentence_classifier.grid5000_state import (
    AutonomousRunState,
    AutonomousStateStore,
)
from osm_polygon_sentence_classifier.tracking import TRACKIO_SPACE_ID
from osm_polygon_sentence_classifier.training import TrainingConfig


class _FakeHub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def create_repo(self, **kwargs: object) -> None:
        self.calls.append(("repo", kwargs))

    def create_bucket(self, **kwargs: object) -> None:
        self.calls.append(("bucket", kwargs))

    def model_info(self, **kwargs: object) -> object:
        self.calls.append(("model_info", kwargs))
        return object()

    def space_info(self, **kwargs: object) -> object:
        self.calls.append(("space_info", kwargs))
        return object()


class _FakeRemote:
    def __init__(self) -> None:
        self.prepared = False
        self.prepared_allow_failed_run: bool | None = None
        self.installed_token: str | None = None
        self.cleaned = False
        self.marked: list[str] = []
        self.status_calls = 0

    def prepare(
        self,
        *,
        run_id: str,
        source_commit: str,
        allow_failed_run: bool = False,
    ) -> RemotePreparationResult:
        del source_commit
        self.prepared = True
        self.prepared_allow_failed_run = allow_failed_run
        return RemotePreparationResult("grenoble", run_id, reused_checkout=False)

    def install_hugging_face_token(self, token: str) -> None:
        self.installed_token = token

    def run(self, command: str, *, input_text: str | None = None) -> CommandResult:
        del input_text
        if "oarsub" in command:
            return CommandResult(returncode=0, stdout="OAR_JOB_ID=99\n")
        if "quota" in command:
            return CommandResult(returncode=0, stdout="0 100000000 100000001\n")
        return CommandResult(returncode=0, stdout="ok\n")

    def raw(self, command: str) -> CommandResult:
        if command.startswith("oarstat -fj"):
            self.status_calls += 1
            if self.status_calls == 1:
                payload = {"99": {"state": "Waiting"}}
            elif self.status_calls == 2:
                payload = {"99": {"state": "Running"}}
            else:
                payload = {"99": {"state": "Terminated", "exit_code": 0}}
            return CommandResult(returncode=0, stdout=json.dumps(payload))
        raise AssertionError(command)

    def read_completion(self, run_id: str) -> dict[str, object]:
        return {
            "schema_version": 1,
            "run_id": run_id,
            "source_commit": "a" * 40,
            "dataset_revision": "b" * 40,
            "model_name_or_path": "test-model",
            "model_revision": "c" * 40,
            "model_publication": {
                "repository_id": "NoeFlandre/osm-polygon-sentence-classifier",
                "commit_id": "d" * 40,
            },
            "tracking_space_id": TRACKIO_SPACE_ID,
        }

    def mark_status(self, run_id: str, status: str) -> None:
        del run_id
        self.marked.append(status)

    def cleanup(self, run_id: str) -> None:
        del run_id
        self.cleaned = True


class _CheckpointContinuationRemote(_FakeRemote):
    def __init__(self) -> None:
        super().__init__()
        self.submission_count = 0
        self.checkpoint_allow_failed_status: bool | None = None

    def run(self, command: str, *, input_text: str | None = None) -> CommandResult:
        del input_text
        if "oarsub" in command:
            self.submission_count += 1
            return CommandResult(
                returncode=0,
                stdout=f"OAR_JOB_ID={99 + self.submission_count - 1}\n",
            )
        if "quota" in command:
            return CommandResult(returncode=0, stdout="0 100000000 100000001\n")
        return CommandResult(returncode=0, stdout="ok\n")

    def raw(self, command: str) -> CommandResult:
        if command.startswith("oarstat -fj"):
            job_id = int(command.split()[2])
            if job_id == 99:
                payload = (
                    {"99": {"state": "Running"}}
                    if self.status_calls == 0
                    else {"99": {"state": "Terminated", "exit_code": -15}}
                )
            else:
                payload = (
                    {"100": {"state": "Running"}}
                    if self.status_calls < 3
                    else {"100": {"state": "Terminated", "exit_code": 0}}
                )
            self.status_calls += 1
            return CommandResult(returncode=0, stdout=json.dumps(payload))
        raise AssertionError(command)

    def has_complete_checkpoint(
        self,
        run_id: str,
        *,
        output_subdirectory: str,
        identity: dict[str, object],
        allow_failed_status: bool = False,
    ) -> bool:
        del run_id, output_subdirectory, identity
        self.checkpoint_allow_failed_status = allow_failed_status
        return True


class _NeverCompletesContinuationRemote(_CheckpointContinuationRemote):
    def __init__(self) -> None:
        super().__init__()
        self.job_status_calls: dict[int, int] = {}

    def raw(self, command: str) -> CommandResult:
        if command.startswith("oarstat -fj"):
            job_id = int(command.split()[2])
            calls = self.job_status_calls.get(job_id, 0)
            self.job_status_calls[job_id] = calls + 1
            state = "Running" if calls == 0 else "Terminated"
            exit_code = None if calls == 0 else -15
            return CommandResult(
                returncode=0,
                stdout=json.dumps(
                    {str(job_id): {"state": state, "exit_code": exit_code}}
                ),
            )
        raise AssertionError(command)


class _FailedRunContinuationRemote(_CheckpointContinuationRemote):
    def __init__(self, *, old_job_state: str = "Missing") -> None:
        super().__init__()
        self.old_job_state = old_job_state
        self.job_status_calls: dict[int, int] = {}

    def run(self, command: str, *, input_text: str | None = None) -> CommandResult:
        del input_text
        if "oarsub" in command:
            self.submission_count += 1
            return CommandResult(returncode=0, stdout="OAR_JOB_ID=100\n")
        if "quota" in command:
            return CommandResult(returncode=0, stdout="0 100000000 100000001\n")
        return CommandResult(returncode=0, stdout="ok\n")

    def raw(self, command: str) -> CommandResult:
        if command.startswith("oarstat -fj"):
            job_id = int(command.split()[2])
            if job_id == 99 and self.old_job_state == "Missing":
                return CommandResult(returncode=6)
            calls = self.job_status_calls.get(job_id, 0)
            self.job_status_calls[job_id] = calls + 1
            state = "Running" if calls == 0 else "Terminated"
            return CommandResult(
                returncode=0,
                stdout=json.dumps(
                    {
                        str(job_id): {
                            "state": self.old_job_state if job_id == 99 else state,
                            "exit_code": 0 if state == "Terminated" else None,
                        }
                    }
                ),
            )
        raise AssertionError(command)


class _FlakyCheckpointProbeRemote(_FailedRunContinuationRemote):
    def __init__(self) -> None:
        super().__init__()
        self.checkpoint_probe_calls = 0

    def has_complete_checkpoint(
        self,
        run_id: str,
        *,
        output_subdirectory: str,
        identity: dict[str, object],
        allow_failed_status: bool = False,
    ) -> bool:
        del run_id, output_subdirectory, identity
        self.checkpoint_allow_failed_status = allow_failed_status
        self.checkpoint_probe_calls += 1
        if self.checkpoint_probe_calls == 1:
            raise Grid5000ExecutionError(
                "remote command failed with exit code 255: connection timed out"
            )
        return True


class _MissingCheckpointRemote(_FailedRunContinuationRemote):
    def has_complete_checkpoint(
        self,
        run_id: str,
        *,
        output_subdirectory: str,
        identity: dict[str, object],
        allow_failed_status: bool = False,
    ) -> bool:
        del run_id, output_subdirectory, identity, allow_failed_status
        return False


class _ReplacementRemote:
    def __init__(self, site: str, state: str) -> None:
        self.site = site
        self.state = state
        self.cancelled: list[int] = []

    def prepare(
        self,
        *,
        run_id: str,
        source_commit: str,
        allow_failed_run: bool = False,
    ) -> RemotePreparationResult:
        del source_commit
        return RemotePreparationResult(self.site, run_id, reused_checkout=True)

    def install_hugging_face_token(self, token: str) -> None:
        del token

    def run(self, command: str, *, input_text: str | None = None) -> CommandResult:
        del input_text
        if "oarsub" in command:
            return CommandResult(returncode=0, stdout="OAR_JOB_ID=11\n")
        if "quota" in command:
            return CommandResult(returncode=0, stdout="0 100000000 100000001\n")
        return CommandResult(returncode=0, stdout="ok\n")

    def raw(self, command: str) -> CommandResult:
        if command.startswith("oarstat -fj"):
            job_id = command.split()[2]
            return CommandResult(
                returncode=0,
                stdout=json.dumps({job_id: {"state": self.state}}),
            )
        raise AssertionError(command)

    def cancel(self, job_id: int) -> None:
        self.cancelled.append(job_id)

    def mark_status(self, run_id: str, status: str) -> None:
        del run_id, status

    def cleanup(self, run_id: str) -> None:
        del run_id


def _config() -> AutonomousRunConfig:
    training = TrainingConfig(
        model_name_or_path="test-model",
        model_revision="c" * 40,
        publish_to_hub=True,
        sync_trackio=True,
    )
    identity = Grid5000RunIdentity(
        source_commit="a" * 40,
        dataset_revision="b" * 40,
        model_name_or_path="test-model",
        model_revision="c" * 40,
        training_config={
            "model_name_or_path": "test-model",
            "model_revision": "c" * 40,
            "publish_to_hub": True,
            "sync_trackio": True,
        },
    )
    return AutonomousRunConfig(
        identity=identity,
        training_config=training,
        sites=("grenoble",),
        requirements=SiteRequirements(gpu_memory_mb=8_000),
        walltime_seconds=1_800,
        cleanup=True,
    )


def test_continuation_helper_does_not_require_unused_status(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config()
    controller = AutonomousRunController(config, state_root=tmp_path / "runs")
    current = AutonomousRunState(
        run_id=config.identity.run_id,
        phase="queued",
        identity=config.identity.canonical_payload,
        site="grenoble",
        job_id=99,
        facts={"continuation_count": config.max_continuations},
    )
    sentinel = object()
    monkeypatch.setattr(
        controller,
        "_fail_terminal",
        lambda *_args, **_kwargs: sentinel,
    )

    result = controller._continue_after_incomplete(
        current,
        site="grenoble",
        job_id=99,
        remote=object(),
        reason="test continuation",
    )

    assert result is sentinel


def test_checkpoint_verification_failure_persists_terminal_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config()
    controller = AutonomousRunController(config, state_root=tmp_path / "runs")
    current = AutonomousRunState(
        run_id=config.identity.run_id,
        phase="running",
        identity=config.identity.canonical_payload,
        site="grenoble",
        job_id=99,
        facts={"continuation_count": 0},
    )
    controller.state.create(current)
    remote = _FakeRemote()

    def fail_checkpoint_verification(*_args, **_kwargs) -> bool:
        raise AutonomousRunError(
            "published checkpoint availability could not be verified"
        )

    monkeypatch.setattr(
        controller,
        "_has_complete_checkpoint",
        fail_checkpoint_verification,
    )

    with pytest.raises(
        AutonomousRunError,
        match="published checkpoint availability could not be verified",
    ):
        controller._continue_after_incomplete(
            current,
            site="grenoble",
            job_id=99,
            remote=remote,
            reason="job ended",
        )

    failed = controller.state.load(config.identity.run_id)
    assert failed is not None
    assert failed.phase == "failed"
    assert dict(failed.facts or {})["error"] == (
        "published checkpoint availability could not be verified"
    )
    assert remote.marked == ["failed"]


def test_continuation_probe_falls_back_to_another_configured_site(
    tmp_path: Path,
) -> None:
    config = replace(_config(), sites=("nancy", "nantes"))
    compatible_resource = GpuResource(
        gpu_memory_mb=16_000,
        cuda_capability=(8, 0),
        jobs_assigned=0,
        production=True,
        exotic=False,
    )
    calls: list[tuple[str, ...]] = []

    def probe_sites(*, sites, **_kwargs):
        requested = tuple(sites)
        calls.append(requested)
        return tuple(
            SiteProbe(
                name=site,
                reachable=site == "nantes",
                resources=(compatible_resource,) if site == "nantes" else (),
                persistent_free_bytes=10 * 1024**3 if site == "nantes" else 0,
                queued_jobs=0,
            )
            for site in requested
        )

    controller = AutonomousRunController(
        config,
        state_root=tmp_path / "runs",
        probe_sites=probe_sites,
    )

    selected = controller._continuation_probe("nancy")

    assert selected.name == "nantes"
    assert calls == [("nancy", "nantes")]


def test_continuation_probe_accepts_resume_headroom_below_fresh_run_requirement(
    tmp_path: Path,
) -> None:
    config = replace(_config(), sites=("nancy", "nantes"))
    compatible_resource = GpuResource(
        gpu_memory_mb=16_000,
        cuda_capability=(8, 0),
        jobs_assigned=0,
        production=True,
        exotic=False,
    )

    def probe_sites(*, sites, **_kwargs):
        return tuple(
            SiteProbe(
                name=site,
                reachable=site == "nantes",
                resources=(compatible_resource,) if site == "nantes" else (),
                persistent_free_bytes=(1 * 1024**3) if site == "nantes" else 0,
                queued_jobs=0,
            )
            for site in sites
        )

    controller = AutonomousRunController(
        config,
        state_root=tmp_path / "runs",
        probe_sites=probe_sites,
    )

    selected = controller._continuation_probe("nancy")

    assert selected.name == "nantes"


def test_continuation_uses_the_selected_site_remote(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    config = replace(_config(), sites=("nancy", "nantes"))
    compatible_resource = GpuResource(
        gpu_memory_mb=16_000,
        cuda_capability=(8, 0),
        jobs_assigned=0,
        production=True,
        exotic=False,
    )
    probe = SiteProbe(
        name="nantes",
        reachable=True,
        resources=(compatible_resource,),
        persistent_free_bytes=10 * 1024**3,
        queued_jobs=0,
    )
    previous_remote = _CheckpointContinuationRemote()
    successor_remote = _CheckpointContinuationRemote()
    remotes = {"nantes": successor_remote}
    controller = AutonomousRunController(
        config,
        state_root=tmp_path / "runs",
        probe_sites=lambda **_: (probe,),
        remote_factory=lambda site: remotes[site],
        hub_api=_FakeHub(),
        poll_seconds=0,
    )
    current = AutonomousRunState(
        run_id=config.identity.run_id,
        phase="running",
        identity=config.identity.canonical_payload,
        site="nancy",
        job_id=99,
        facts={"continuation_count": 0},
    )
    controller.state.create(current)

    result = controller._continue_after_incomplete(
        current,
        site="nancy",
        job_id=99,
        remote=previous_remote,
        reason="walltime",
    )

    assert result.phase == "completed"
    assert result.site == "nantes"
    assert previous_remote.submission_count == 0
    assert successor_remote.submission_count > 0


def test_legacy_replacement_state_is_upgraded_for_the_current_job() -> None:
    assert (
        _replacement_attempt_count_for_job(
            {"replacement_attempted": True},
            job_id=99,
        )
        == 0
    )
    assert (
        _replacement_attempt_count_for_job(
            {"replacement_attempted": True, "replacement_attempted_job_id": 99},
            job_id=99,
        )
        == 0
    )
    assert (
        _replacement_attempt_count_for_job(
            {
                "replacement_attempted": True,
                "replacement_attempted_job_id": 99,
                "replacement_attempt_count": 2,
            },
            job_id=99,
        )
        == 2
    )


def test_queued_replacement_is_retried_after_a_cooldown_but_is_bounded(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config()
    controller = AutonomousRunController(config, state_root=tmp_path / "runs")
    monkeypatch.setattr(
        controller,
        "_has_complete_checkpoint",
        lambda *args, **kwargs: False,
    )
    now = datetime(2026, 8, 5, 19, 0, tzinfo=UTC)
    monkeypatch.setattr(
        "osm_polygon_sentence_classifier.grid5000_autonomous._now",
        lambda: now,
    )
    current = AutonomousRunState(
        run_id=config.identity.run_id,
        phase="queued",
        identity=config.identity.canonical_payload,
        site="grenoble",
        job_id=99,
        facts={"resume_from_checkpoint": True},
    )
    controller.state.create(current)
    remote = _FakeRemote()
    status = JobStatus(
        job_id=99,
        state=JobState.QUEUED,
        scheduled_start="2026-08-06 08:02:02",
    )
    calls: list[tuple[int, bool]] = []

    def try_replacement(**kwargs: object) -> tuple[str, int, object]:
        calls.append((len(calls) + 1, kwargs["resume_from_checkpoint"] is True))
        return "grenoble", 99, kwargs["fallback_remote"]

    monkeypatch.setattr(controller, "_try_replacement", try_replacement)

    queued = controller._handle_queued_status(
        current,
        status=status,
        site="grenoble",
        job_id=99,
        remote=remote,
    )
    assert isinstance(queued, tuple)
    current, *_ = queued
    assert calls == [(1, True)]
    assert dict(current.facts or {})["replacement_attempt_count"] == 1

    queued = controller._handle_queued_status(
        current,
        status=status,
        site="grenoble",
        job_id=99,
        remote=remote,
    )
    assert isinstance(queued, tuple)
    current, *_ = queued
    assert calls == [(1, True)]

    now += REPLACEMENT_RETRY_INTERVAL
    for expected_count in range(2, MAX_REPLACEMENT_ATTEMPTS + 1):
        queued = controller._handle_queued_status(
            current,
            status=status,
            site="grenoble",
            job_id=99,
            remote=remote,
        )
        assert isinstance(queued, tuple)
        current, *_ = queued
        assert calls == [(count, True) for count in range(1, expected_count + 1)]

        if expected_count < MAX_REPLACEMENT_ATTEMPTS:
            now += REPLACEMENT_RETRY_INTERVAL

    now += REPLACEMENT_RETRY_INTERVAL
    with pytest.raises(AutonomousRunError, match="scheduled start"):
        controller._handle_queued_status(
            current,
            status=status,
            site="grenoble",
            job_id=99,
            remote=remote,
        )
    assert calls == [(count, True) for count in range(1, MAX_REPLACEMENT_ATTEMPTS + 1)]
    assert remote.marked == ["failed"]


def test_queued_job_without_a_forecast_starts_a_bounded_replacement_round(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config()
    controller = AutonomousRunController(config, state_root=tmp_path / "runs")
    now = datetime(2026, 8, 5, 19, 0, tzinfo=UTC)
    monkeypatch.setattr(
        "osm_polygon_sentence_classifier.grid5000_autonomous._now",
        lambda: now,
    )
    current = AutonomousRunState(
        run_id=config.identity.run_id,
        phase="queued",
        identity=config.identity.canonical_payload,
        site="grenoble",
        job_id=99,
        updated_at=now.isoformat(),
    )
    controller.state.create(current)
    calls: list[int] = []

    def try_replacement(**kwargs: object) -> tuple[str, int, object]:
        calls.append(len(calls) + 1)
        return "grenoble", 99, kwargs["fallback_remote"]

    monkeypatch.setattr(controller, "_try_replacement", try_replacement)

    queued = controller._handle_queued_status(
        current,
        status=JobStatus(job_id=99, state=JobState.QUEUED),
        site="grenoble",
        job_id=99,
        remote=object(),
    )
    assert isinstance(queued, tuple)
    current, *_ = queued

    assert calls == [1]
    assert dict(current.facts or {})["replacement_attempt_count"] == 1


def test_unpredicted_fallback_is_canceled_after_bounded_replacement_rounds(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = _config()
    controller = AutonomousRunController(config, state_root=tmp_path / "runs")
    monkeypatch.setattr(
        controller,
        "_has_complete_checkpoint",
        lambda *args, **kwargs: False,
    )
    now = datetime(2026, 8, 5, 19, 0, tzinfo=UTC)
    monkeypatch.setattr(
        "osm_polygon_sentence_classifier.grid5000_autonomous._now",
        lambda: now,
    )

    class CancelRecordingRemote:
        def __init__(self) -> None:
            self.commands: list[str] = []
            self.marked: list[str] = []

        def run(
            self,
            command: str,
            *,
            input_text: str | None = None,
        ) -> CommandResult:
            del input_text
            self.commands.append(command)
            return CommandResult(returncode=0, stdout="ok\n")

        def mark_status(self, run_id: str, status: str) -> None:
            del run_id
            self.marked.append(status)

    remote = CancelRecordingRemote()
    current = AutonomousRunState(
        run_id=config.identity.run_id,
        phase="queued",
        identity=config.identity.canonical_payload,
        site="grenoble",
        job_id=99,
        updated_at=now.isoformat(),
    )
    controller.state.create(current)
    monkeypatch.setattr(
        controller,
        "_try_replacement",
        lambda **kwargs: ("grenoble", 99, kwargs["fallback_remote"]),
    )

    for _ in range(MAX_REPLACEMENT_ATTEMPTS):
        queued = controller._handle_queued_status(
            current,
            status=JobStatus(job_id=99, state=JobState.QUEUED),
            site="grenoble",
            job_id=99,
            remote=remote,
        )
        assert isinstance(queued, tuple)
        current, *_ = queued
        now += REPLACEMENT_RETRY_INTERVAL

    with pytest.raises(AutonomousRunError, match="no start-time prediction"):
        controller._handle_queued_status(
            current,
            status=JobStatus(job_id=99, state=JobState.QUEUED),
            site="grenoble",
            job_id=99,
            remote=remote,
        )

    assert remote.commands == ["oardel 99"]
    assert remote.marked == ["failed"]


def test_stale_queued_job_restarts_from_a_complete_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    config = replace(_config(), max_continuations=41)
    remote = _CheckpointContinuationRemote()
    probe = SiteProbe(
        name="grenoble",
        reachable=True,
        resources=(
            GpuResource(
                gpu_memory_mb=16_000,
                cuda_capability=(8, 0),
                jobs_assigned=0,
                production=True,
                exotic=False,
            ),
        ),
        persistent_free_bytes=10 * 1024**3,
        queued_jobs=0,
    )
    current = AutonomousRunState(
        run_id=config.identity.run_id,
        phase="queued",
        identity=config.identity.canonical_payload,
        site="grenoble",
        job_id=99,
        facts={
            "continuation_count": 14,
            "max_continuations": 41,
            "replacement_attempted": True,
            "replacement_attempted_job_id": 99,
            "replacement_attempt_count": MAX_REPLACEMENT_ATTEMPTS,
            "replacement_last_attempt_at": "2026-08-18T19:00:00+02:00",
        },
    )
    controller = AutonomousRunController(
        config,
        state_root=tmp_path / "runs",
        probe_sites=lambda **_: (probe,),
        remote_factory=lambda _site: remote,
        hub_api=_FakeHub(),
        poll_seconds=0,
    )
    controller.state.create(current)

    result = controller._handle_queued_status(
        current,
        status=JobStatus(
            job_id=99,
            state=JobState.QUEUED,
            scheduled_start="2026-08-19 10:34:02",
        ),
        site="grenoble",
        job_id=99,
        remote=remote,
    )

    assert isinstance(result, AutonomousRunState)
    assert result.phase == "completed"
    assert remote.submission_count > 0


def test_controller_runs_prepare_submit_monitor_publish_verify_and_cleanup(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    remote = _FakeRemote()
    hub = _FakeHub()
    messages: list[str] = []
    probe = SiteProbe(
        name="grenoble",
        reachable=True,
        resources=(
            GpuResource(
                gpu_memory_mb=16_000,
                cuda_capability=(8, 0),
                jobs_assigned=0,
                production=True,
                exotic=False,
            ),
        ),
        persistent_free_bytes=10 * 1024**3,
        queued_jobs=0,
    )

    controller = AutonomousRunController(
        _config(),
        state_root=tmp_path / "runs",
        probe_sites=lambda **_: (probe,),
        remote_factory=lambda _site: remote,
        hub_api=hub,
        poll_seconds=0,
        emit=messages.append,
    )

    result = controller.run()

    assert result.phase == "completed"
    assert remote.prepared
    assert remote.installed_token == "hf_test_token"
    assert remote.marked == ["complete"]
    assert remote.cleaned
    assert any(kind == "repo" for kind, _ in hub.calls)
    assert any(kind == "bucket" for kind, _ in hub.calls)
    assert any(kind == "model_info" for kind, _ in hub.calls)
    assert any("phase=running" in message for message in messages)
    assert any("grenoble job 99: running" in message for message in messages)
    assert any("verifying completion manifest" in message for message in messages)


def test_controller_continues_from_a_checkpoint_after_walltime_termination(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    remote = _CheckpointContinuationRemote()
    probe = SiteProbe(
        name="grenoble",
        reachable=True,
        resources=(
            GpuResource(
                gpu_memory_mb=16_000,
                cuda_capability=(8, 0),
                jobs_assigned=0,
                production=True,
                exotic=False,
            ),
        ),
        persistent_free_bytes=10 * 1024**3,
        queued_jobs=0,
    )
    controller = AutonomousRunController(
        _config(),
        state_root=tmp_path / "runs",
        probe_sites=lambda **_: (probe,),
        remote_factory=lambda _site: remote,
        hub_api=_FakeHub(),
        poll_seconds=0,
    )

    result = controller.run()

    assert result.phase == "completed"
    assert remote.submission_count == 2
    assert remote.marked == ["complete"]
    assert remote.cleaned
    state = AutonomousStateStore(tmp_path / "runs").load(
        controller.config.identity.run_id
    )
    assert state is not None
    facts = dict(state.facts or {})
    assert facts["continuation_count"] == 1
    assert facts["resume_from_checkpoint"] is True
    assert facts["replacement_attempted"] is False


def test_continuation_facts_preserve_the_resume_state_contract() -> None:
    assert _continuation_facts(
        continuation_count=2,
        max_continuations=3,
        worker_source_commit="a" * 40,
        continuation_pending=False,
        continuation_reason="walltime",
        last_terminal_job_id=99,
        resume_from_checkpoint=True,
        scheduler_command=("oarsub", "-l", "nodes=1"),
    ) == {
        "continuation_count": 2,
        "max_continuations": 3,
        "worker_source_commit": "a" * 40,
        "continuation_pending": False,
        "continuation_reason": "walltime",
        "last_terminal_job_id": 99,
        "replacement_attempted": False,
        "replacement_attempted_job_id": None,
        "replacement_attempt_count": 0,
        "replacement_last_attempt_at": None,
        "resume_from_checkpoint": True,
        "scheduler_command": ["oarsub", "-l", "nodes=1"],
    }


def test_pending_continuation_facts_omit_submission_only_fields() -> None:
    assert _continuation_facts(
        continuation_count=2,
        max_continuations=3,
        worker_source_commit=None,
        continuation_pending=True,
        continuation_reason="walltime",
        last_terminal_job_id=99,
    ) == {
        "continuation_count": 2,
        "max_continuations": 3,
        "worker_source_commit": None,
        "continuation_pending": True,
        "continuation_reason": "walltime",
        "last_terminal_job_id": 99,
        "replacement_attempted": False,
        "replacement_attempted_job_id": None,
        "replacement_attempt_count": 0,
        "replacement_last_attempt_at": None,
    }


def test_controller_stops_after_the_continuation_limit(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    remote = _NeverCompletesContinuationRemote()
    probe = SiteProbe(
        name="grenoble",
        reachable=True,
        resources=(
            GpuResource(
                gpu_memory_mb=16_000,
                cuda_capability=(8, 0),
                jobs_assigned=0,
                production=True,
                exotic=False,
            ),
        ),
        persistent_free_bytes=10 * 1024**3,
        queued_jobs=0,
    )
    controller = AutonomousRunController(
        replace(_config(), max_continuations=1),
        state_root=tmp_path / "runs",
        probe_sites=lambda **_: (probe,),
        remote_factory=lambda _site: remote,
        hub_api=_FakeHub(),
        poll_seconds=0,
    )

    with pytest.raises(AutonomousRunError, match="after 1 checkpoint continuations"):
        controller.run()

    assert remote.submission_count == 2
    assert remote.marked == ["failed"]
    assert not remote.cleaned
    state = AutonomousStateStore(tmp_path / "runs").load(
        controller.config.identity.run_id
    )
    assert state is not None
    assert state.phase == "failed"
    assert dict(state.facts or {})["continuation_count"] == 1


def test_failed_run_can_extend_from_a_retained_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    config = replace(_config(), max_continuations=2)
    remote = _FailedRunContinuationRemote()
    probe = SiteProbe(
        name="grenoble",
        reachable=True,
        resources=(
            GpuResource(
                gpu_memory_mb=16_000,
                cuda_capability=(8, 0),
                jobs_assigned=0,
                production=True,
                exotic=False,
            ),
        ),
        persistent_free_bytes=10 * 1024**3,
        queued_jobs=0,
    )
    current = AutonomousRunState(
        run_id=config.identity.run_id,
        phase="failed",
        identity=config.identity.canonical_payload,
        site="grenoble",
        job_id=99,
        facts={
            "continuation_count": 1,
            "max_continuations": 1,
            "error": "job ended without completion after 1 checkpoint continuations",
        },
    )
    controller = AutonomousRunController(
        config,
        state_root=tmp_path / "runs",
        probe_sites=lambda **_: (probe,),
        remote_factory=lambda _site: remote,
        hub_api=_FakeHub(),
        poll_seconds=0,
    )
    controller.state.create(current)

    result = controller.run()

    assert result.phase == "completed"
    assert remote.submission_count == 1
    assert remote.marked == ["complete"]
    state = AutonomousStateStore(tmp_path / "runs").load(config.identity.run_id)
    assert state is not None
    facts = dict(state.facts or {})
    assert facts["continuation_count"] == 2
    assert facts["max_continuations"] == 2
    assert remote.checkpoint_allow_failed_status is True
    assert remote.prepared_allow_failed_run is True


def test_failed_run_can_restart_after_a_stale_queued_job(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    config = replace(_config(), max_continuations=41)
    remote = _FailedRunContinuationRemote()
    probe = SiteProbe(
        name="grenoble",
        reachable=True,
        resources=(
            GpuResource(
                gpu_memory_mb=16_000,
                cuda_capability=(8, 0),
                jobs_assigned=0,
                production=True,
                exotic=False,
            ),
        ),
        persistent_free_bytes=10 * 1024**3,
        queued_jobs=0,
    )
    current = AutonomousRunState(
        run_id=config.identity.run_id,
        phase="failed",
        identity=config.identity.canonical_payload,
        site="grenoble",
        job_id=99,
        facts={
            "continuation_count": 14,
            "max_continuations": 41,
            "error": (
                "grenoble job 99 remained queued with scheduled start "
                "2026-08-19 10:34:02 after 3 replacement rounds"
            ),
        },
    )
    controller = AutonomousRunController(
        config,
        state_root=tmp_path / "runs",
        probe_sites=lambda **_: (probe,),
        remote_factory=lambda _site: remote,
        hub_api=_FakeHub(),
        poll_seconds=0,
    )
    controller.state.create(current)

    result = controller.run()

    assert result.phase == "completed"
    assert remote.submission_count == 1


def test_failed_run_can_recover_from_a_published_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    config = replace(_config(), max_continuations=2)
    remote = _MissingCheckpointRemote()
    probe = SiteProbe(
        name="grenoble",
        reachable=True,
        resources=(
            GpuResource(
                gpu_memory_mb=16_000,
                cuda_capability=(8, 0),
                jobs_assigned=0,
                production=True,
                exotic=False,
            ),
        ),
        persistent_free_bytes=10 * 1024**3,
        queued_jobs=0,
    )
    monkeypatch.setattr(
        "osm_polygon_sentence_classifier.grid5000_autonomous.latest_published_checkpoint",
        lambda *_args, **_kwargs: PublishedCheckpoint(
            repository_id="NoeFlandre/osm-polygon-sentence-classifier",
            prefix="studies/landuse-v1/run-test",
            step=12,
            files=(),
        ),
    )
    current = AutonomousRunState(
        run_id=config.identity.run_id,
        phase="failed",
        identity=config.identity.canonical_payload,
        site="grenoble",
        job_id=99,
        facts={
            "continuation_count": 1,
            "max_continuations": 2,
            "error": "job ended without a complete checkpoint",
        },
    )
    controller = AutonomousRunController(
        config,
        state_root=tmp_path / "runs",
        probe_sites=lambda **_: (probe,),
        remote_factory=lambda _site: remote,
        hub_api=_FakeHub(),
        poll_seconds=0,
    )
    controller.state.create(current)

    result = controller.run()

    assert result.phase == "completed"
    assert remote.submission_count > 0


def test_failed_run_retries_a_transient_checkpoint_probe(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    config = replace(_config(), max_continuations=2)
    remote = _FlakyCheckpointProbeRemote()
    probe = SiteProbe(
        name="grenoble",
        reachable=True,
        resources=(
            GpuResource(
                gpu_memory_mb=16_000,
                cuda_capability=(8, 0),
                jobs_assigned=0,
                production=True,
                exotic=False,
            ),
        ),
        persistent_free_bytes=10 * 1024**3,
        queued_jobs=0,
    )
    sleeps: list[float] = []
    controller = AutonomousRunController(
        config,
        state_root=tmp_path / "runs",
        probe_sites=lambda **_: (probe,),
        remote_factory=lambda _site: remote,
        hub_api=_FakeHub(),
        poll_seconds=30,
        sleeper=sleeps.append,
    )
    current = AutonomousRunState(
        run_id=config.identity.run_id,
        phase="failed",
        identity=config.identity.canonical_payload,
        site="grenoble",
        job_id=99,
        facts={
            "continuation_count": 1,
            "max_continuations": 1,
            "error": "job ended without completion after 1 checkpoint continuations",
        },
    )
    controller.state.create(current)

    result = controller.run()

    assert result.phase == "completed"
    assert remote.checkpoint_probe_calls == 2
    assert sleeps[0] == 5.0


def test_failed_run_extension_refuses_an_active_previous_job(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    config = replace(_config(), max_continuations=2)
    remote = _FailedRunContinuationRemote(old_job_state="Running")
    current = AutonomousRunState(
        run_id=config.identity.run_id,
        phase="failed",
        identity=config.identity.canonical_payload,
        site="grenoble",
        job_id=99,
        facts={
            "continuation_count": 1,
            "max_continuations": 1,
            "error": "job ended without completion after 1 checkpoint continuations",
        },
    )
    controller = AutonomousRunController(
        config,
        state_root=tmp_path / "runs",
        remote_factory=lambda _site: remote,
        poll_seconds=0,
    )
    controller.state.create(current)

    with pytest.raises(AutonomousRunError, match="active Grid'5000 job"):
        controller.run()

    assert remote.submission_count == 0


def test_explicit_worker_checkout_revision_is_used_for_continuations(
    tmp_path: Path,
) -> None:
    checkout_commit = "c" * 40
    controller = AutonomousRunController(
        replace(_config(), worker_source_commit=checkout_commit),
        state_root=tmp_path / "runs",
    )
    probe = SiteProbe(
        name="grenoble",
        reachable=True,
        resources=(
            GpuResource(
                gpu_memory_mb=16_000,
                cuda_capability=(8, 0),
                jobs_assigned=0,
                production=True,
                exotic=False,
            ),
        ),
        persistent_free_bytes=10 * 1024**3,
        queued_jobs=0,
    )

    plan = controller._build_plan(probe, resume_from_checkpoint=True)

    assert plan.checkout_commit == checkout_commit


def test_resume_rejects_changed_container_settings(
    tmp_path: Path,
) -> None:
    image = "registry.example/osm-polygon-sentence-classifier@sha256:" + "a" * 64
    changed_image = (
        "registry.example/osm-polygon-sentence-classifier@sha256:" + "b" * 64
    )
    config = replace(_config(), container_image=image, container_runtime="docker")
    store = AutonomousStateStore(tmp_path / "runs")
    state = AutonomousRunState(
        run_id=config.identity.run_id,
        phase="failed",
        identity=config.identity.canonical_payload,
        site="grenoble",
        job_id=99,
        facts={
            "container_image": image,
            "container_runtime": "docker",
        },
    )
    store.create(state)
    controller = AutonomousRunController(
        replace(config, container_image=changed_image),
        state_root=tmp_path / "runs",
    )

    with pytest.raises(AutonomousRunError, match="differ from the persisted run"):
        controller.run()


def test_completion_verification_rejects_a_model_repository_mismatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    controller = AutonomousRunController(
        _config(),
        state_root=tmp_path / "runs",
        hub_api=_FakeHub(),
    )

    with pytest.raises(AutonomousRunError, match="model publication"):
        controller._verify_completion(
            object(),
            {
                "run_id": controller.config.identity.run_id,
                "source_commit": controller.config.identity.source_commit,
                "dataset_revision": controller.config.identity.dataset_revision,
                "model_name_or_path": controller.config.identity.model_name_or_path,
                "model_revision": controller.config.identity.model_revision,
                "model_publication": {
                    "repository_id": "NoeFlandre/wrong-model",
                    "commit_id": "d" * 40,
                },
                "tracking_space_id": TRACKIO_SPACE_ID,
            },
        )


def test_controller_preserves_a_live_state_after_an_unexpected_error(
    tmp_path: Path,
) -> None:
    config = _config()
    store = AutonomousStateStore(tmp_path / "runs")
    state = AutonomousRunState(
        run_id=config.identity.run_id,
        phase="running",
        identity=config.identity.canonical_payload,
        site="grenoble",
        job_id=99,
    )
    store.create(state)
    controller = AutonomousRunController(config, state_root=tmp_path / "runs")

    controller._record_unexpected_failure(RuntimeError("temporary SSH failure"))

    assert store.load(config.identity.run_id) == state


def test_replacement_reuses_the_remote_that_submitted_the_trial(
    tmp_path: Path,
) -> None:
    config = replace(_config(), sites=("nancy", "grenoble"), cleanup=False)
    fallback = _ReplacementRemote("nancy", "Waiting")
    candidate = _ReplacementRemote("grenoble", "Running")
    remotes = {"nancy": fallback, "grenoble": candidate}
    probed_sites: list[tuple[str, ...]] = []
    probe = SiteProbe(
        name="grenoble",
        reachable=True,
        resources=(
            GpuResource(
                gpu_memory_mb=16_000,
                cuda_capability=(8, 0),
                jobs_assigned=0,
                production=True,
                exotic=False,
            ),
        ),
        persistent_free_bytes=10 * 1024**3,
        queued_jobs=0,
    )
    controller = AutonomousRunController(
        config,
        state_root=tmp_path / "runs",
        probe_sites=lambda **kwargs: (
            probed_sites.append(tuple(kwargs["sites"])) or (probe,)
        ),
        remote_factory=lambda site: remotes[site],
        poll_seconds=0,
    )

    site, job_id, selected_remote = controller._try_replacement(
        fallback_site="nancy",
        fallback_job_id=10,
        fallback_remote=fallback,
    )

    assert (site, job_id) == ("grenoble", 11)
    assert selected_remote is candidate
    assert probed_sites == [("nancy", "grenoble")]


def test_checkpoint_replacement_requires_worker_resume_flag(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    config = replace(_config(), sites=("nancy", "grenoble"), cleanup=False)
    fallback = _ReplacementRemote("nancy", "Waiting")
    candidate = _ReplacementRemote("grenoble", "Running")
    remotes = {"nancy": fallback, "grenoble": candidate}
    probe = SiteProbe(
        name="grenoble",
        reachable=True,
        resources=(
            GpuResource(
                gpu_memory_mb=16_000,
                cuda_capability=(8, 0),
                jobs_assigned=0,
                production=True,
                exotic=False,
            ),
        ),
        persistent_free_bytes=10 * 1024**3,
        queued_jobs=0,
    )
    controller = AutonomousRunController(
        config,
        state_root=tmp_path / "runs",
        probe_sites=lambda **_: (probe,),
        remote_factory=lambda site: remotes[site],
        poll_seconds=0,
    )
    plans: list[Grid5000Plan] = []

    def fake_submit_plan(_remote: object, plan: Grid5000Plan) -> int:
        plans.append(plan)
        return 11

    monkeypatch.setattr(controller, "_submit_plan", fake_submit_plan)

    controller._try_replacement(
        fallback_site="nancy",
        fallback_job_id=10,
        fallback_remote=fallback,
        resume_from_checkpoint=True,
    )

    assert len(plans) == 1
    assert plans[0].resume_from_checkpoint is True
    assert plans[0].allocation.walltime_seconds == config.walltime_seconds
    assert "--require-checkpoint" in plans[0].worker_command


def test_replacement_skips_a_site_without_persistent_headroom(
    tmp_path: Path,
) -> None:
    config = replace(_config(), sites=("nancy", "grenoble"), cleanup=False)
    controller = AutonomousRunController(config, state_root=tmp_path / "runs")
    probe = SiteProbe(
        name="grenoble",
        reachable=True,
        resources=(
            GpuResource(
                gpu_memory_mb=16_000,
                cuda_capability=(8, 0),
                jobs_assigned=0,
                production=True,
                exotic=False,
            ),
        ),
        persistent_free_bytes=1 * 1024**3,
        queued_jobs=0,
    )

    candidates = controller._candidate_list((probe,), fallback_site="nancy")

    assert candidates == ()
