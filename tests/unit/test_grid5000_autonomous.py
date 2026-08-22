import json
import sys
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta, tzinfo
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn, cast

import pytest

import osm_polygon_sentence_classifier.grid5000_autonomous as autonomous
from osm_polygon_sentence_classifier.checkpoint_hub import (
    HubCheckpointError,
    PublishedCheckpoint,
)
from osm_polygon_sentence_classifier.grid5000 import (
    CommandResult,
    Grid5000Allocation,
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
from osm_polygon_sentence_classifier.grid5000_checkpointing import (
    CheckpointProbeError,
)
from osm_polygon_sentence_classifier.grid5000_oar import JobState, JobStatus
from osm_polygon_sentence_classifier.grid5000_policy import (
    QueuedReplacementDecision,
)
from osm_polygon_sentence_classifier.grid5000_remote import RemotePreparationResult
from osm_polygon_sentence_classifier.grid5000_replacement import ReplacementContext
from osm_polygon_sentence_classifier.grid5000_sites import (
    GpuResource,
    SiteProbe,
    SiteRequirements,
)
from osm_polygon_sentence_classifier.grid5000_state import (
    AutonomousRunState,
    AutonomousStateStore,
    LegacyAmbiguousStateError,
    RunPhase,
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


@pytest.mark.parametrize(
    "resume_from_checkpoint",
    [False, True],
    ids=["fresh-run", "checkpoint-continuation"],
)
def test_preflight_runs_each_guard_and_accepts_exact_headroom_boundary(
    tmp_path: Path,
    resume_from_checkpoint: bool,
) -> None:
    requirements = SiteRequirements(
        gpu_memory_mb=8_000,
        resume_persistent_free_bytes=2 * 1024**3,
    )
    config = replace(_config(), requirements=requirements)
    plan = Grid5000Plan(
        identity=config.identity,
        allocation=Grid5000Allocation(site="grenoble", walltime_seconds=1_800),
        resume_from_checkpoint=resume_from_checkpoint,
    )

    minimum_headroom = (
        requirements.resume_persistent_free_bytes
        if resume_from_checkpoint
        else autonomous.MINIMUM_HOME_HEADROOM_BYTES
    )

    class Remote:
        def __init__(self) -> None:
            self.commands: list[str] = []

        def run(
            self,
            command: str,
            *,
            input_text: str | None = None,
        ) -> CommandResult:
            del input_text
            self.commands.append(command)
            if command == plan.quota_command[-1]:
                return CommandResult(
                    returncode=0,
                    stdout=f"0 {minimum_headroom // 1024} "
                    f"{minimum_headroom // 1024 + 1}\n",
                )
            return CommandResult(returncode=0, stdout="ok\n")

    remote = Remote()
    emitted: list[str] = []
    controller = AutonomousRunController(
        config,
        state_root=tmp_path / "runs",
        emit=emitted.append,
    )

    controller._preflight(remote, plan)

    assert remote.commands == [
        plan.remote_checkout_command[-1],
        plan.policy_site_command[-1],
        plan.policy_total_command[-1],
        plan.quota_command[-1],
    ]
    assert emitted == [
        "grenoble: running checkout, policy, and quota preflight",
    ]


def test_preflight_rejects_headroom_below_the_required_boundary(
    tmp_path: Path,
) -> None:
    config = _config()
    plan = Grid5000Plan(
        identity=config.identity,
        allocation=Grid5000Allocation(site="grenoble", walltime_seconds=1_800),
    )
    minimum_headroom = autonomous.MINIMUM_HOME_HEADROOM_BYTES

    class Remote:
        def run(
            self,
            command: str,
            *,
            input_text: str | None = None,
        ) -> CommandResult:
            del input_text
            if command == plan.quota_command[-1]:
                return CommandResult(
                    returncode=0,
                    stdout=f"0 {(minimum_headroom - 1024) // 1024} "
                    f"{minimum_headroom // 1024 + 1}\n",
                )
            return CommandResult(returncode=0, stdout="ok\n")

    controller = AutonomousRunController(
        config,
        state_root=tmp_path / "runs",
    )

    with pytest.raises(AutonomousRunError) as error:
        controller._preflight(Remote(), plan)

    assert str(error.value) == (
        "Grid'5000 home soft quota has insufficient safe headroom"
    )


def test_build_plan_forwards_requirements_and_preserves_plan_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requirements = SiteRequirements(
        gpu_memory_mb=16_000,
        cuda_capability=(8, 0),
    )
    image = "registry.example/classifier@sha256:" + "a" * 64
    config = replace(
        _config(),
        requirements=requirements,
        container_image=image,
        container_runtime="docker",
        policy_type="day",
    )
    captured: dict[str, object] = {}

    def choose(
        resources: object,
        *,
        requirements: SiteRequirements | None,
    ) -> dict[str, str]:
        captured["resources"] = resources
        captured["requirements"] = requirements
        return {
            "queue": "production",
            "resource_type": "standard",
            "resource_property": (
                "gpu_mem>=16000 AND production='YES' AND cpuarch='x86_64' "
                "AND gpu_compute_capability IN ('8.0')"
            ),
        }

    monkeypatch.setattr(autonomous, "choose_allocation", choose)
    controller = AutonomousRunController(
        config,
        state_root=tmp_path / "runs",
    )
    probe = SiteProbe(
        name="grenoble",
        reachable=True,
        resources=(),
        persistent_free_bytes=10 * 1024**3,
        queued_jobs=0,
    )

    plan = controller._build_plan(probe, walltime_seconds=720)

    assert captured == {
        "resources": probe.resources,
        "requirements": requirements,
    }
    assert plan.allocation == Grid5000Allocation(
        site="grenoble",
        walltime_seconds=720,
        queue="production",
        resource_type="standard",
        resource_property=(
            "gpu_mem>=16000 AND production='YES' AND cpuarch='x86_64' "
            "AND gpu_compute_capability IN ('8.0')"
        ),
        policy_type="day",
    )
    assert plan.resume_from_checkpoint is False
    assert plan.container_image == image
    assert plan.container_runtime == "docker"


def test_build_plan_rejects_a_site_without_a_compatible_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(autonomous, "choose_allocation", lambda *_args, **_kwargs: None)
    controller = AutonomousRunController(
        _config(),
        state_root=tmp_path / "runs",
    )
    probe = SiteProbe(
        name="grenoble",
        reachable=True,
        resources=(),
        persistent_free_bytes=10 * 1024**3,
        queued_jobs=0,
    )

    with pytest.raises(AutonomousRunError) as error:
        controller._build_plan(probe)

    assert str(error.value) == "site grenoble has no compatible GPU allocation"


def test_provision_hub_forwards_both_publication_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracking_project = "test-trackio-project"
    training = replace(_config().training_config, tracking_project=tracking_project)
    config = replace(_config(), training_config=training)
    controller = AutonomousRunController(config, state_root=tmp_path / "runs")
    hub = object()
    settings = object()
    model_calls: list[tuple[str, object]] = []
    tracking_calls: list[tuple[object, object]] = []
    settings_calls: list[tuple[object, str | None]] = []
    emitted: list[str] = []

    def ensure_model(repository: str, *, hub_api: object) -> None:
        model_calls.append((repository, hub_api))

    def build_settings(
        project_config: object,
        *,
        project: str | None = None,
    ) -> object:
        settings_calls.append((project_config, project))
        return settings

    def ensure_tracking(actual_settings: object, *, hub_api: object) -> None:
        tracking_calls.append((actual_settings, hub_api))

    monkeypatch.setattr(controller, "_hub", lambda: hub)
    monkeypatch.setattr(controller, "emit", emitted.append)
    monkeypatch.setattr(autonomous, "ensure_model_repository", ensure_model)
    monkeypatch.setattr(autonomous, "settings_for", build_settings)
    monkeypatch.setattr(autonomous, "ensure_trackio_resources", ensure_tracking)

    controller._provision_hub()

    assert emitted == [
        "provisioning the Hugging Face model repository",
        "provisioning the Trackio Space and bucket",
    ]
    assert model_calls == [(autonomous.ProjectConfig().target_model_repository_id, hub)]
    assert settings_calls == [(autonomous.ProjectConfig(), tracking_project)]
    assert tracking_calls == [(settings, hub)]


def test_verify_tracking_space_uses_the_configured_project_and_space(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracking_project = "test-trackio-project"
    training = replace(_config().training_config, tracking_project=tracking_project)
    controller = AutonomousRunController(
        replace(_config(), training_config=training),
        state_root=tmp_path / "runs",
    )
    hub = _FakeHub()
    settings = SimpleNamespace(static_space_id="test/trackio-space")
    settings_calls: list[tuple[object, str | None]] = []

    def build_settings(
        project_config: object,
        *,
        project: str | None = None,
    ) -> object:
        settings_calls.append((project_config, project))
        return settings

    monkeypatch.setattr(controller, "_hub", lambda: hub)
    monkeypatch.setattr(autonomous, "settings_for", build_settings)

    controller._verify_tracking_space({"tracking_space_id": "test/trackio-space"})

    assert settings_calls == [(autonomous.ProjectConfig(), tracking_project)]
    assert hub.calls == [("space_info", {"repo_id": "test/trackio-space"})]


def test_verify_tracking_space_rejects_the_wrong_public_space(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = AutonomousRunController(
        _config(),
        state_root=tmp_path / "runs",
    )
    monkeypatch.setattr(
        autonomous,
        "settings_for",
        lambda *_args, **_kwargs: SimpleNamespace(static_space_id="expected/space"),
    )

    with pytest.raises(AutonomousRunError) as error:
        controller._verify_tracking_space({"tracking_space_id": "other/space"})

    assert str(error.value) == "completed worker reported the wrong Trackio Space"


def test_verify_model_publication_forwards_repository_and_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = AutonomousRunController(
        _config(),
        state_root=tmp_path / "runs",
    )
    hub = _FakeHub()
    repository = "NoeFlandre/osm-polygon-sentence-classifier"
    commit_id = "d" * 40
    monkeypatch.setattr(controller, "_hub", lambda: hub)

    def validated_model_publication(value: object) -> tuple[str, str]:
        if value != "facts":
            raise AssertionError("unexpected model publication value")
        return repository, commit_id

    monkeypatch.setattr(
        controller, "_validated_model_publication", validated_model_publication
    )

    controller._verify_model_publication({"model_publication": "facts"})

    assert hub.calls == [
        (
            "model_info",
            {"repo_id": repository, "revision": commit_id},
        )
    ]


@pytest.mark.parametrize(
    "value",
    [
        {"repository_id": 1, "commit_id": "d" * 40},
        {"repository_id": "repo", "commit_id": 1},
        {"repository_id": "repo", "commit_id": "d" * 39},
    ],
)
def test_publication_values_rejects_each_invalid_field(
    value: dict[str, object],
) -> None:
    with pytest.raises(AutonomousRunError) as error:
        AutonomousRunController._publication_values(value)

    assert str(error.value) == "model publication facts are invalid"


def test_fresh_run_persists_all_reproducibility_facts_before_submission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = "registry.example/model@sha256:" + "a" * 64
    requirements = SiteRequirements(
        gpu_memory_mb=12_000,
        cuda_capability=(8, 0),
        persistent_free_bytes=9 * 1024**3,
        resume_persistent_free_bytes=2 * 1024**3,
    )
    config = replace(
        _config(),
        sites=("grenoble", "lille"),
        requirements=requirements,
        walltime_seconds=900,
        policy_type="night",
        max_workers=2,
        max_continuations=7,
        worker_source_commit="e" * 40,
        container_image=image,
        container_runtime="docker",
        cleanup=False,
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
    remote = _FakeRemote()
    probe_calls: list[tuple[tuple[str, ...], SiteRequirements, int]] = []
    remote_sites: list[object] = []
    prepare_calls: list[tuple[str, str]] = []
    transitions: list[tuple[object, dict[str, object]]] = []
    incompatible_probe = SiteProbe(
        name="aaa",
        reachable=True,
        resources=(
            GpuResource(
                gpu_memory_mb=10_000,
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
        probe_sites=lambda *, sites, requirements, max_workers: (
            probe_calls.append((tuple(sites), requirements, max_workers))
            or (probe, incompatible_probe)
        ),
        remote_factory=lambda site: remote_sites.append(site) or remote,
        emit=lambda _message: None,
    )
    original_prepare = remote.prepare

    def prepare(*, run_id: str, source_commit: str, allow_failed_run: bool = False):
        prepare_calls.append((run_id, source_commit))
        return original_prepare(
            run_id=run_id,
            source_commit=source_commit,
            allow_failed_run=allow_failed_run,
        )

    monkeypatch.setattr(remote, "prepare", prepare)
    original_transition = controller._transition

    def transition(current, phase, **kwargs):
        transitions.append((phase, dict(kwargs)))
        return original_transition(current, phase, **kwargs)

    monkeypatch.setattr(controller, "_transition", transition)
    monkeypatch.setattr(controller, "_local_token_for_publication", lambda: "")
    monkeypatch.setattr(controller, "_provision_hub", lambda: None)
    submitted_plans: list[Grid5000Plan] = []
    monkeypatch.setattr(
        controller,
        "_submit_plan",
        lambda _remote, plan: submitted_plans.append(plan) or 99,
    )
    monkeypatch.setattr(
        controller,
        "_monitor",
        lambda state, *, remote: state,
    )

    result = controller._fresh_run()

    assert result.phase == "submitted"
    assert probe_calls == [(config.sites, requirements, 2)]
    assert remote_sites == ["grenoble"]
    assert prepare_calls == [(config.identity.run_id, "e" * 40)]
    assert controller._active_remote is remote
    assert len(submitted_plans) == 1
    assert submitted_plans[0].allocation.site == "grenoble"
    persisted = controller.state.load(config.identity.run_id)
    assert persisted is not None
    facts = dict(persisted.facts or {})
    assert facts["probes"] == [
        probe.to_dict(requirements),
        incompatible_probe.to_dict(requirements),
    ]
    assert facts["selected_site"] == "grenoble"
    assert facts["cleanup"] is False
    assert facts["max_continuations"] == 7
    assert facts["worker_source_commit"] == "e" * 40
    assert facts["container_image"] == image
    assert facts["container_runtime"] == "docker"
    assert facts["requested_policy_type"] == "night"
    assert facts["sites"] == ["grenoble", "lille"]
    assert facts["requirements"] == {
        "gpu_memory_mb": 12_000,
        "cuda_capability": [8, 0],
        "persistent_free_bytes": 9 * 1024**3,
        "resume_persistent_free_bytes": 2 * 1024**3,
    }
    probing = [
        kwargs
        for phase, kwargs in transitions
        if phase == autonomous.RunPhase.PROBING
        and isinstance(kwargs.get("facts"), dict)
        and "selected_site" in cast(dict[str, object], kwargs["facts"])
    ]
    prepared = [
        kwargs for phase, kwargs in transitions if phase == autonomous.RunPhase.PREPARED
    ]
    assert probing
    assert probing[0]["site"] == "grenoble"
    assert prepared
    assert prepared[0]["site"] == "grenoble"
    prepared_facts = cast(dict[str, object], prepared[0]["facts"])
    assert prepared_facts["allocation"] == submitted_plans[0].to_dict()["allocation"]
    assert prepared_facts["scheduler_command"]
    submitted = [
        kwargs
        for phase, kwargs in transitions
        if phase == autonomous.RunPhase.SUBMITTED
    ]
    assert submitted
    assert submitted[0]["site"] == "grenoble"


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


def test_fail_terminal_persists_location_marks_run_failed_and_raises_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    controller = AutonomousRunController(config, state_root=tmp_path / "runs")
    current = AutonomousRunState(
        run_id=config.identity.run_id,
        phase=RunPhase.RUNNING,
        identity=config.identity.canonical_payload,
    )
    transition: dict[str, object] = {}
    status_calls: list[tuple[str, str]] = []
    remote = SimpleNamespace(
        mark_status=lambda run_id, status: status_calls.append((run_id, status))
    )

    def persist_transition(
        current_arg: AutonomousRunState,
        phase: RunPhase,
        *,
        site: str | None,
        job_id: int | None,
        facts: dict[str, object] | None,
    ) -> AutonomousRunState:
        transition.update(phase=phase, site=site, job_id=job_id, facts=facts)
        return controller._transition_state(current_arg, phase, site, job_id, facts)

    monkeypatch.setattr(controller, "_transition", persist_transition)

    with pytest.raises(AutonomousRunError) as error:
        controller._fail_terminal(
            current,
            site="grenoble",
            job_id=99,
            remote=remote,
            message="checkpoint missing",
        )

    assert str(error.value) == "checkpoint missing"
    assert transition == {
        "phase": RunPhase.FAILED,
        "site": "grenoble",
        "job_id": 99,
        "facts": {"error": "checkpoint missing"},
    }
    assert status_calls == [(config.identity.run_id, "failed")]


def test_fail_terminal_uses_the_stable_fallback_error_message(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    controller = AutonomousRunController(config, state_root=tmp_path / "runs")
    current = AutonomousRunState(
        run_id=config.identity.run_id,
        phase=RunPhase.RUNNING,
        identity=config.identity.canonical_payload,
    )
    remote = SimpleNamespace(mark_status=lambda _run_id, _status: None)

    def persist_without_facts(
        current_arg: AutonomousRunState,
        phase: RunPhase,
        *,
        site: str | None,
        job_id: int | None,
        facts: dict[str, object] | None,
    ) -> AutonomousRunState:
        del facts
        return controller._transition_state(current_arg, phase, site, job_id, None)

    monkeypatch.setattr(controller, "_transition", persist_without_facts)

    with pytest.raises(AutonomousRunError) as error:
        controller._fail_terminal(
            current,
            site="grenoble",
            job_id=99,
            remote=remote,
            message="ignored by stub",
        )

    assert str(error.value) == "job failed"


def test_continuation_limit_failure_preserves_terminal_job_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    controller = AutonomousRunController(config, state_root=tmp_path / "runs")
    current = AutonomousRunState(
        run_id=config.identity.run_id,
        phase="running",
        identity=config.identity.canonical_payload,
        site="grenoble",
        job_id=99,
        facts={"continuation_count": config.max_continuations},
    )
    remote = object()
    captured: dict[str, object] = {}
    sentinel = object()

    def fail_terminal(*_args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(controller, "_fail_terminal", fail_terminal)

    result = controller._continue_after_incomplete(
        current,
        site="grenoble",
        job_id=99,
        remote=remote,
        reason="walltime",
    )

    assert result is sentinel
    assert captured == {
        "site": "grenoble",
        "job_id": 99,
        "remote": remote,
        "message": ("job ended without completion after 3 checkpoint continuations"),
    }


def test_continuation_preserves_checkpoint_and_submission_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    controller = AutonomousRunController(config, state_root=tmp_path / "runs")
    current = AutonomousRunState(
        run_id=config.identity.run_id,
        phase="running",
        identity=config.identity.canonical_payload,
        site="grenoble",
        job_id=99,
        facts={"continuation_count": 1},
    )
    remote = object()
    successor_remote = object()
    probe = cast(SiteProbe, object())
    plan = object()
    emitted: list[str] = []
    checkpoint_calls: list[tuple[object, dict[str, object]]] = []
    preparation_calls: list[tuple[object, dict[str, object]]] = []
    submission_calls: list[tuple[object, dict[str, object]]] = []
    sentinel = object()

    def require_checkpoint(current_arg: object, **kwargs: object) -> None:
        checkpoint_calls.append((current_arg, kwargs))

    def prepare(current_arg: object, **kwargs: object) -> tuple[object, object, object]:
        preparation_calls.append((current_arg, kwargs))
        return probe, successor_remote, plan

    def submit(current_arg: object, **kwargs: object) -> object:
        submission_calls.append((current_arg, kwargs))
        return sentinel

    monkeypatch.setattr(controller, "emit", emitted.append)
    monkeypatch.setattr(
        controller,
        "_require_continuation_checkpoint",
        require_checkpoint,
    )
    monkeypatch.setattr(controller, "_prepare_continuation", prepare)
    monkeypatch.setattr(controller, "_submit_continuation", submit)

    result = controller._continue_after_incomplete(
        current,
        site="grenoble",
        job_id=99,
        remote=remote,
        reason="walltime",
        failure_message="custom failure",
    )

    assert result is sentinel
    assert emitted == [
        "grenoble job 99: checking checkpoint evidence before continuation 2/3",
        "grenoble job 99: complete checkpoint found",
    ]
    assert checkpoint_calls == [
        (
            current,
            {
                "site": "grenoble",
                "job_id": 99,
                "remote": remote,
                "failure_message": "custom failure",
            },
        )
    ]
    assert preparation_calls == [
        (
            current,
            {"site": "grenoble", "remote": remote},
        )
    ]
    assert submission_calls == [
        (
            current,
            {
                "probe": probe,
                "remote": successor_remote,
                "plan": plan,
                "raw_count": 1,
                "reason": "walltime",
                "last_terminal_job_id": 99,
            },
        )
    ]


@pytest.mark.parametrize("phase", [RunPhase.SUBMITTED, RunPhase.QUEUED])
@pytest.mark.parametrize("node", [None, "gpu-node-1"])
def test_running_state_persists_scheduler_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: RunPhase,
    node: str | None,
) -> None:
    config = _config()
    controller = AutonomousRunController(config, state_root=tmp_path / "runs")
    current = AutonomousRunState(
        run_id=config.identity.run_id,
        phase=phase,
        identity=config.identity.canonical_payload,
        site="grenoble",
        job_id=99,
    )
    status = JobStatus(job_id=123, state=JobState.RUNNING, node=node)
    transitions: list[tuple[object, RunPhase, dict[str, object]]] = []
    sentinel = object()

    def transition(
        state: object,
        next_phase: RunPhase,
        **kwargs: object,
    ) -> object:
        transitions.append((state, next_phase, kwargs))
        return sentinel

    monkeypatch.setattr(controller, "_transition", transition)

    result = controller._running_state(
        current,
        status,
        site="nantes",
        job_id=123,
    )

    assert result is sentinel
    assert transitions == [
        (
            current,
            RunPhase.RUNNING,
            {
                "site": "nantes",
                "job_id": 123,
                "facts": {"node": node or ""},
            },
        )
    ]


@pytest.mark.parametrize("phase", ["submitted", "queued", "running"])
def test_resume_loaded_phase_monitors_existing_scheduler_states(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
) -> None:
    config = _config()
    remote = object()
    remotes: list[str] = []
    monitor_calls: list[tuple[object, object]] = []
    controller = AutonomousRunController(
        config,
        state_root=tmp_path / "runs",
        remote_factory=lambda site: remotes.append(site) or remote,
    )
    current = AutonomousRunState(
        run_id=config.identity.run_id,
        phase=phase,
        identity=config.identity.canonical_payload,
        site="grenoble",
        job_id=99,
    )
    sentinel = object()
    monkeypatch.setattr(
        controller,
        "_monitor",
        lambda state, *, remote: monitor_calls.append((state, remote)) or sentinel,
    )

    assert controller._resume_loaded_phase(current) is sentinel
    assert remotes == ["grenoble"]
    assert controller._active_remote is remote
    assert monitor_calls == [(current, remote)]


def test_resume_loaded_phase_preserves_the_ambiguous_submission_error(
    tmp_path: Path,
) -> None:
    config = _config()
    controller = AutonomousRunController(config, state_root=tmp_path / "runs")
    current = AutonomousRunState(
        run_id=config.identity.run_id,
        phase=RunPhase.SUBMITTING,
        identity=config.identity.canonical_payload,
    )

    with pytest.raises(AutonomousRunError) as error:
        controller._resume_loaded_phase(current)

    assert str(error.value) == (
        "submission is ambiguous; inspect scheduler state before retrying"
    )


def test_resume_loaded_phase_delegates_failed_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    controller = AutonomousRunController(config, state_root=tmp_path / "runs")
    current = AutonomousRunState(
        run_id=config.identity.run_id,
        phase="failed",
        identity=config.identity.canonical_payload,
    )
    sentinel = object()
    monkeypatch.setattr(controller, "_resume_failed_run", lambda state: sentinel)

    assert controller._resume_loaded_phase(current) is sentinel


@pytest.mark.parametrize(
    ("phase", "message"),
    [("submitting", "ambiguous"), ("created", "not resumable")],
)
def test_resume_loaded_phase_rejects_unsafe_states(
    tmp_path: Path,
    phase: str,
    message: str,
) -> None:
    config = _config()
    controller = AutonomousRunController(config, state_root=tmp_path / "runs")
    current = AutonomousRunState(
        run_id=config.identity.run_id,
        phase=phase,
        identity=config.identity.canonical_payload,
    )

    with pytest.raises(AutonomousRunError, match=message):
        controller._resume_loaded_phase(current)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sites", (), "at least one Grid'5000 site is required"),
        ("walltime_seconds", 0, "walltime_seconds must be positive"),
        ("policy_type", "invalid", "policy_type must be auto, day, or night"),
        ("max_workers", 0, "max_workers must be positive"),
        ("max_continuations", 0, "max_continuations must be positive"),
    ],
)
def test_autonomous_config_rejects_invalid_lifecycle_limits(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(AutonomousRunError) as error:
        replace(_config(), **{field: value})
    assert str(error.value) == message


@pytest.mark.parametrize("field", ["walltime_seconds", "max_workers"])
def test_autonomous_config_accepts_the_smallest_positive_boundary(
    field: str,
) -> None:
    config = replace(_config(), **{field: 1})
    assert getattr(config, field) == 1


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("model_name_or_path", "training model does not match run identity"),
        ("model_revision", "training revision does not match run identity"),
    ],
)
def test_autonomous_identity_validation_reports_each_mismatch(
    field: str,
    message: str,
) -> None:
    config = _config()
    training = replace(
        config.training_config,
        **{field: "different-model" if field == "model_name_or_path" else "e" * 40},
    )

    with pytest.raises(AutonomousRunError) as error:
        autonomous._validate_autonomous_identity(training, config.identity)

    assert str(error.value) == message


def test_validated_model_publication_rejects_non_mapping_and_wrong_repository(
    tmp_path: Path,
) -> None:
    controller = AutonomousRunController(_config(), state_root=tmp_path / "runs")

    with pytest.raises(AutonomousRunError) as non_mapping_error:
        controller._validated_model_publication(None)
    assert str(non_mapping_error.value) == (
        "completed worker did not report a model publication"
    )

    with pytest.raises(AutonomousRunError) as repository_error:
        controller._validated_model_publication(
            {"repository_id": "other/repository", "commit_id": "d" * 40}
        )
    assert str(repository_error.value) == "model publication repository is invalid"


def test_autonomous_validators_preserve_exact_boundary_contracts() -> None:
    autonomous._validate_max_continuations(1)
    autonomous._validate_worker_source_commit("a" * 40)

    for value in (True, 0, "1"):
        with pytest.raises(AutonomousRunError) as error:
            autonomous._validate_max_continuations(value)
        assert str(error.value) == "max_continuations must be positive"

    for value in ("", "a" * 39, "g" * 40):
        with pytest.raises(AutonomousRunError) as error:
            autonomous._validate_worker_source_commit(value)
        assert str(error.value) == "worker_source_commit must be a pinned revision"


@pytest.mark.parametrize("runtime", ["docker", "podman", "auto"])
def test_persisted_container_runtime_accepts_each_supported_value(runtime: str) -> None:
    autonomous._validate_persisted_container_runtime(runtime)


@pytest.mark.parametrize("runtime", [None, "invalid", "PODMAN", 1])
def test_persisted_container_runtime_rejects_each_unsupported_value(
    runtime: object,
) -> None:
    with pytest.raises(AutonomousRunError) as error:
        autonomous._validate_persisted_container_runtime(runtime)

    assert str(error.value) == "persisted container runtime is invalid"


def test_autonomous_container_validation_preserves_configuration_errors() -> None:
    with pytest.raises(AutonomousRunError) as error:
        autonomous._validate_autonomous_container(None, "docker")

    assert str(error.value) == "container_runtime requires an explicit container_image"

    with pytest.raises(AutonomousRunError) as error:
        autonomous._validate_autonomous_container(None, cast(Any, "invalid"))

    assert str(error.value) == "container_runtime must be auto, docker, or podman"


def test_controller_constructor_preserves_runtime_dependencies_and_defaults(
    tmp_path: Path,
) -> None:
    runner: Any = object()
    environ = {"HF_TOKEN": "test-token"}
    controller = AutonomousRunController(
        _config(),
        state_root=tmp_path / "runs",
        runner=runner,
        environ=environ,
    )

    assert controller.runner is runner
    assert controller.environ is environ
    assert controller.poll_seconds == 30.0
    assert controller._active_remote is None
    assert controller.emit("ignored") is None


def test_controller_constructor_rejects_a_negative_poll_interval(
    tmp_path: Path,
) -> None:
    with pytest.raises(AutonomousRunError) as error:
        AutonomousRunController(
            _config(),
            state_root=tmp_path / "runs",
            poll_seconds=-0.1,
        )

    assert str(error.value) == "poll_seconds must be non-negative"


def test_run_wraps_unexpected_controller_errors_and_preserves_the_cause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = AutonomousRunController(_config(), state_root=tmp_path / "runs")
    unexpected = RuntimeError("unexpected failure")
    recorded: list[Exception] = []

    def run_loaded(_current: object) -> AutonomousRunState:
        raise unexpected

    monkeypatch.setattr(controller, "_run_loaded", run_loaded)
    monkeypatch.setattr(
        controller,
        "_record_unexpected_failure",
        lambda error: recorded.append(error),
    )

    with pytest.raises(AutonomousRunError) as error:
        controller.run()

    assert str(error.value) == "autonomous Grid'5000 run failed"
    assert error.value.__cause__ is unexpected
    assert recorded == [unexpected]


def test_hub_returns_an_injected_api_without_importing_dependencies(
    tmp_path: Path,
) -> None:
    config = _config()
    hub = object()
    controller = AutonomousRunController(
        config,
        state_root=tmp_path / "runs",
        hub_api=hub,
    )

    assert controller._hub() is hub


def test_load_or_reconcile_returns_a_persisted_run_state(
    tmp_path: Path,
) -> None:
    config = _config()
    state_root = tmp_path / "runs"
    state = AutonomousRunState(
        run_id=config.identity.run_id,
        phase="created",
        identity=config.identity.canonical_payload,
    )
    AutonomousStateStore(state_root).create(state)
    controller = AutonomousRunController(config, state_root=state_root)

    assert controller._load_or_reconcile() == state


def test_load_or_reconcile_archives_legacy_state_after_checking_all_sites(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(_config(), sites=("nancy", "lille"))
    requested_sites: list[str] = []
    reconciled: list[tuple[str, tuple[int, ...]]] = []
    emitted: list[str] = []

    class UserJobsRemote:
        def raw(self, command: str) -> CommandResult:
            assert command == "oarstat -u -J"
            return CommandResult(returncode=0, stdout="{}")

    controller = AutonomousRunController(
        config,
        state_root=tmp_path / "runs",
        remote_factory=lambda site: requested_sites.append(site) or UserJobsRemote(),
        emit=emitted.append,
    )

    def raise_legacy(_run_id: str) -> AutonomousRunState | None:
        raise LegacyAmbiguousStateError("legacy state is ambiguous")

    monkeypatch.setattr(controller.state, "load", raise_legacy)
    monkeypatch.setattr(
        controller.state,
        "reconcile_legacy",
        lambda run_id, *, active_job_ids: reconciled.append(
            (run_id, tuple(active_job_ids))
        )
        or tmp_path / "archive",
    )

    assert controller._load_or_reconcile() is None
    assert requested_sites == ["nancy", "lille"]
    assert reconciled == [(config.identity.run_id, ())]
    assert emitted == [
        "archived legacy ambiguous state at " + str(tmp_path / "archive")
    ]


def test_hub_wraps_dependency_initialization_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    controller = AutonomousRunController(config, state_root=tmp_path / "runs")
    monkeypatch.setattr(
        autonomous,
        "configure_huggingface_http",
        lambda: (_ for _ in ()).throw(RuntimeError("missing dependency")),
    )

    with pytest.raises(AutonomousRunError) as error:
        controller._hub()

    assert str(error.value) == "Hugging Face dependencies are required for publication"


def test_hub_caches_the_constructed_api(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    controller = AutonomousRunController(config, state_root=tmp_path / "runs")
    api = object()
    module = SimpleNamespace(HfApi=lambda: api)
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    monkeypatch.setattr(autonomous, "configure_huggingface_http", lambda: None)

    assert controller._hub() is api
    assert controller._hub() is api


def test_local_publication_token_uses_the_injected_environment(
    tmp_path: Path,
) -> None:
    controller = AutonomousRunController(
        _config(),
        state_root=tmp_path / "runs",
        environ={"HF_TOKEN": "injected-token"},
    )

    assert controller._local_token_for_publication() == "injected-token"


@pytest.mark.parametrize(
    ("publish_to_hub", "sync_trackio"),
    [(True, False), (False, True)],
)
def test_local_publication_token_requires_auth_for_each_publication_target(
    tmp_path: Path,
    publish_to_hub: bool,
    sync_trackio: bool,
) -> None:
    training = replace(
        _config().training_config,
        publish_to_hub=publish_to_hub,
        sync_trackio=sync_trackio,
    )
    controller = AutonomousRunController(
        replace(_config(), training_config=training),
        state_root=tmp_path / "runs",
        environ={"HF_HOME": str(tmp_path / "empty-cache")},
    )

    with pytest.raises(AutonomousRunError) as error:
        controller._local_token_for_publication()

    assert str(error.value) == "HF authentication is required for requested publication"


def test_local_hugging_face_token_prefers_environment_and_reads_hf_home(
    tmp_path: Path,
) -> None:
    token_path = tmp_path / "token"
    token_path.write_text(" file-token \n", encoding="utf-8")

    assert (
        autonomous._local_hugging_face_token(
            {"HF_TOKEN": " env-token ", "HF_HOME": str(tmp_path)}
        )
        == "env-token"
    )
    assert autonomous._local_hugging_face_token({"HF_HOME": str(tmp_path)}) == (
        "file-token"
    )
    assert (
        autonomous._local_hugging_face_token({"HF_HOME": str(tmp_path / "missing")})
        == ""
    )


def test_local_hugging_face_token_uses_default_home_and_utf8(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    token_path = tmp_path / ".cache" / "huggingface" / "token"
    token_path.parent.mkdir(parents=True)
    token_path.write_text(" default-token \n", encoding="utf-8")
    monkeypatch.setattr(autonomous.Path, "home", lambda: tmp_path)

    encodings: list[object] = []
    read_paths: list[Path] = []
    original_read_text = autonomous.Path.read_text

    def read_text(
        path: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        read_paths.append(path)
        encodings.append(encoding)
        return original_read_text(path, encoding=encoding, errors=errors)

    monkeypatch.setattr(autonomous.Path, "read_text", read_text)

    assert autonomous._local_hugging_face_token({}) == "default-token"
    assert read_paths == [token_path]
    assert encodings == ["utf-8"]


@pytest.mark.parametrize("raw_timestamp", [1, False, object()])
def test_last_replacement_attempt_rejects_non_string_timestamps(
    raw_timestamp: object,
) -> None:
    with pytest.raises(AutonomousRunError) as error:
        autonomous._last_replacement_attempt(
            {"replacement_last_attempt_at": raw_timestamp}
        )

    assert str(error.value) == "durable replacement timestamp is invalid"


def test_timezone_validation_requires_a_known_offset() -> None:
    class _UnknownOffset(tzinfo):
        def utcoffset(self, _value: datetime | None):  # type: ignore[override]
            return None

    autonomous._validate_timezone(datetime(2026, 8, 5, tzinfo=UTC))

    for value in (
        datetime(2026, 8, 5),
        datetime(2026, 8, 5, tzinfo=_UnknownOffset()),
    ):
        with pytest.raises(ValueError, match=r"^now must be timezone-aware$") as error:
            autonomous._validate_timezone(value)
        assert str(error.value) == "now must be timezone-aware"


def test_replacement_timestamp_requires_an_aware_iso_timestamp() -> None:
    timestamp = autonomous._parse_replacement_timestamp("2026-08-05T19:00:00Z")

    assert timestamp == datetime(2026, 8, 5, 19, 0, tzinfo=UTC)

    for raw_timestamp in ("not-a-timestamp", "2026-08-05T19:00:00"):
        with pytest.raises(AutonomousRunError) as error:
            autonomous._parse_replacement_timestamp(raw_timestamp)
        assert str(error.value) == "durable replacement timestamp is invalid"


@pytest.mark.parametrize(
    ("validator", "valid", "invalid", "message"),
    [
        (
            autonomous._validated_replacement_attempt_count,
            0,
            (True, -1, "0"),
            "durable replacement attempt count is invalid",
        ),
        (
            autonomous._validated_failed_run_count,
            0,
            (True, -1, "0"),
            "failed run continuation evidence is invalid",
        ),
        (
            autonomous._validated_failed_run_limit,
            1,
            (True, 0, "1"),
            "failed run continuation evidence is invalid",
        ),
        (
            autonomous._validated_continuation_count,
            0,
            (True, -1, "0"),
            "durable continuation count is invalid",
        ),
    ],
)
def test_durable_integer_validators_enforce_their_boundaries(
    validator,
    valid: int,
    invalid: tuple[object, ...],
    message: str,
) -> None:
    assert validator(valid) == valid

    for value in invalid:
        with pytest.raises(AutonomousRunError) as error:
            validator(value)
        assert str(error.value) == message


def test_replacement_retry_due_obeys_the_ten_minute_boundary() -> None:
    now = datetime(2026, 8, 5, 19, 0, tzinfo=UTC)
    facts = {
        "replacement_attempted": True,
        "replacement_attempted_job_id": 99,
        "replacement_attempt_count": 1,
        "replacement_last_attempt_at": (now - REPLACEMENT_RETRY_INTERVAL).isoformat(),
    }

    assert autonomous._replacement_retry_due_for_job(facts, job_id=99, now=now)
    assert not autonomous._replacement_retry_due_for_job(
        {
            **facts,
            "replacement_last_attempt_at": (
                now - REPLACEMENT_RETRY_INTERVAL + timedelta(seconds=1)
            ).isoformat(),
        },
        job_id=99,
        now=now,
    )


def test_replacement_retry_is_due_without_a_prior_attempt() -> None:
    now = datetime(2026, 8, 5, 19, 0, tzinfo=UTC)

    assert autonomous._replacement_retry_due_for_job(
        {
            "replacement_attempted": True,
            "replacement_attempted_job_id": 99,
            "replacement_attempt_count": 1,
        },
        job_id=99,
        now=now,
    )
    assert autonomous._replacement_retry_due_for_job(
        {
            "replacement_attempted": True,
            "replacement_attempted_job_id": 99,
            "replacement_attempt_count": 0,
            "replacement_last_attempt_at": "not-a-timestamp",
        },
        job_id=99,
        now=now,
    )


def test_replacement_retry_rejects_a_naive_clock() -> None:
    with pytest.raises(ValueError, match="now must be timezone-aware"):
        autonomous._replacement_retry_due_for_job(
            {}, job_id=99, now=datetime(2026, 8, 5, 19, 0)
        )


def test_record_unexpected_failure_does_not_create_missing_state(
    tmp_path: Path,
) -> None:
    controller = AutonomousRunController(_config(), state_root=tmp_path / "runs")

    controller._record_unexpected_failure(RuntimeError("no state"))

    assert not (tmp_path / "runs").exists()


def test_record_unexpected_failure_marks_a_terminal_state(
    tmp_path: Path,
) -> None:
    config = _config()
    store = AutonomousStateStore(tmp_path / "runs")
    state = AutonomousRunState(
        run_id=config.identity.run_id,
        phase="failed",
        identity=config.identity.canonical_payload,
    )
    store.create(state)

    class Remote:
        def __init__(self) -> None:
            self.marked: list[str] = []

        def mark_status(self, run_id: str, status: str) -> None:
            del run_id
            self.marked.append(status)

    remote = Remote()
    controller = AutonomousRunController(config, state_root=tmp_path / "runs")
    controller._active_remote = remote

    controller._record_unexpected_failure(RuntimeError("terminal error"))

    failed = store.load(config.identity.run_id)
    assert failed is not None
    assert dict(failed.facts or {})["error"] == "terminal error"
    assert remote.marked == ["failed"]


def test_record_unexpected_failure_appends_the_complete_live_error_event(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    events: list[tuple[object, ...]] = []
    monkeypatch.setattr(
        controller.state,
        "append_event",
        lambda *args: events.append(args),
    )

    controller._record_unexpected_failure(RuntimeError("temporary SSH failure"))

    assert events == [
        (
            config.identity.run_id,
            "controller_error",
            {"message": "temporary SSH failure"},
        )
    ]


def test_record_unexpected_failure_suppresses_terminal_status_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    store = AutonomousStateStore(tmp_path / "runs")
    state = AutonomousRunState(
        run_id=config.identity.run_id,
        phase="failed",
        identity=config.identity.canonical_payload,
    )
    store.create(state)
    controller = AutonomousRunController(config, state_root=tmp_path / "runs")
    transitions: list[tuple[object, RunPhase, dict[str, object]]] = []
    failed = replace(state, facts={"error": "terminal error"})

    def transition(
        current: object,
        phase: RunPhase,
        **kwargs: object,
    ) -> AutonomousRunState:
        transitions.append((current, phase, kwargs))
        return failed

    class Remote:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def mark_status(self, run_id: str, status: str) -> None:
            self.calls.append((run_id, status))
            raise RuntimeError("status endpoint unavailable")

    remote = Remote()
    controller._active_remote = remote
    monkeypatch.setattr(controller, "_transition", transition)

    controller._record_unexpected_failure(RuntimeError("terminal error"))

    assert transitions == [
        (
            state,
            RunPhase.FAILED,
            {"facts": {"error": "terminal error"}},
        )
    ]
    assert remote.calls == [(config.identity.run_id, "failed")]


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


def test_require_continuation_checkpoint_forwards_failure_context_and_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    controller = AutonomousRunController(config, state_root=tmp_path / "runs")
    current = AutonomousRunState(
        run_id=config.identity.run_id,
        phase="running",
        identity=config.identity.canonical_payload,
    )
    remote = object()
    checkpoint_calls: list[dict[str, object]] = []
    failure_calls: list[dict[str, object]] = []

    def probe_checkpoint(actual_remote: object, **kwargs: object) -> bool:
        checkpoint_calls.append({"remote": actual_remote, **kwargs})
        return False

    monkeypatch.setattr(controller, "_has_complete_checkpoint", probe_checkpoint)
    monkeypatch.setattr(
        controller,
        "_fail_terminal",
        lambda _current, **kwargs: failure_calls.append(kwargs),
    )

    controller._require_continuation_checkpoint(
        current,
        site="grenoble",
        job_id=99,
        remote=remote,
        failure_message=None,
    )

    assert checkpoint_calls == [
        {
            "remote": remote,
            "site": "grenoble",
            "job_id": 99,
            "allow_failed_status": False,
        }
    ]
    assert failure_calls == [
        {
            "site": "grenoble",
            "job_id": 99,
            "remote": remote,
            "message": "job ended without a complete checkpoint",
        }
    ]


def test_require_continuation_checkpoint_preserves_probe_error_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    controller = AutonomousRunController(config, state_root=tmp_path / "runs")
    current = AutonomousRunState(
        run_id=config.identity.run_id,
        phase="failed",
        identity=config.identity.canonical_payload,
    )
    remote = object()
    failure_calls: list[dict[str, object]] = []

    class _TerminalFailure(Exception):
        pass

    def probe_checkpoint(*_args: object, **_kwargs: object) -> bool:
        raise AutonomousRunError("checkpoint probe failed")

    monkeypatch.setattr(controller, "_has_complete_checkpoint", probe_checkpoint)

    def fail_terminal(_current: object, **kwargs: object) -> NoReturn:
        failure_calls.append(kwargs)
        raise _TerminalFailure

    monkeypatch.setattr(controller, "_fail_terminal", fail_terminal)

    with pytest.raises(_TerminalFailure):
        controller._require_continuation_checkpoint(
            current,
            site="nancy",
            job_id=123,
            remote=remote,
            failure_message="custom checkpoint failure",
        )

    assert failure_calls == [
        {
            "site": "nancy",
            "job_id": 123,
            "remote": remote,
            "message": "checkpoint probe failed",
        }
    ]


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


def test_continuation_probe_forwards_resume_requirements_and_worker_limit(
    tmp_path: Path,
) -> None:
    config = replace(_config(), sites=("nancy", "nantes"), max_workers=7)
    compatible_resource = GpuResource(
        gpu_memory_mb=16_000,
        cuda_capability=(8, 0),
        jobs_assigned=0,
        production=True,
        exotic=False,
    )
    captured: dict[str, object] = {}

    def probe_sites(**kwargs: object) -> tuple[SiteProbe, ...]:
        captured.update(kwargs)
        return (
            SiteProbe(
                name="nantes",
                reachable=True,
                resources=(compatible_resource,),
                persistent_free_bytes=10 * 1024**3,
                queued_jobs=0,
            ),
        )

    controller = AutonomousRunController(
        config,
        state_root=tmp_path / "runs",
        probe_sites=probe_sites,
    )

    selected = controller._continuation_probe("nancy")

    assert selected.name == "nantes"
    assert captured == {
        "sites": config.sites,
        "requirements": config.requirements.for_checkpoint_continuation(),
        "max_workers": 7,
    }


def test_continuation_probe_reports_the_stable_no_site_error(
    tmp_path: Path,
) -> None:
    controller = AutonomousRunController(
        _config(),
        state_root=tmp_path / "runs",
        probe_sites=lambda **_kwargs: (),
    )

    with pytest.raises(AutonomousRunError) as error:
        controller._continuation_probe("grenoble")

    assert str(error.value) == (
        "no compatible Grid'5000 site is available for checkpoint continuation"
    )


def test_complete_checkpoint_probe_forwards_local_and_published_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = AutonomousRunController(_config(), state_root=tmp_path / "runs")
    local_calls: list[dict[str, object]] = []
    published_calls: list[dict[str, object]] = []

    def local_probe(remote: object, **kwargs: object) -> bool:
        local_calls.append({"remote": remote, **kwargs})
        return False

    def published_probe(**kwargs: object) -> bool:
        published_calls.append(kwargs)
        return True

    monkeypatch.setattr(controller, "_probe_local_checkpoint", local_probe)
    monkeypatch.setattr(controller, "_probe_published_checkpoint", published_probe)
    remote = object()

    assert controller._has_complete_checkpoint(
        remote,
        site="grenoble",
        job_id=99,
        allow_failed_status=True,
    )
    assert local_calls == [
        {
            "remote": remote,
            "site": "grenoble",
            "job_id": 99,
            "allow_failed_status": True,
        }
    ]
    assert published_calls == [{"site": "grenoble", "job_id": 99}]


def test_complete_checkpoint_probe_does_not_query_publication_when_local_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = AutonomousRunController(_config(), state_root=tmp_path / "runs")
    monkeypatch.setattr(
        controller, "_probe_local_checkpoint", lambda *_args, **_kwargs: True
    )
    monkeypatch.setattr(
        controller,
        "_probe_published_checkpoint",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("published checkpoint should not be queried")
        ),
    )

    assert controller._has_complete_checkpoint(
        object(), site="grenoble", job_id=99, allow_failed_status=False
    )


@pytest.mark.parametrize(
    ("site", "job_id"),
    [(None, 99), ("grenoble", None), (None, None)],
)
def test_failed_run_location_requires_both_scheduler_coordinates(
    site: str | None,
    job_id: int | None,
) -> None:
    current = AutonomousRunState(
        run_id="a" * 20,
        phase=RunPhase.FAILED,
        identity={},
        site=site,
        job_id=job_id,
    )

    with pytest.raises(AutonomousRunError) as error:
        AutonomousRunController._failed_run_location(current)

    assert str(error.value) == "failed run lacks its last Grid'5000 job"


def test_failed_run_location_returns_the_complete_scheduler_coordinates() -> None:
    current = AutonomousRunState(
        run_id="a" * 20,
        phase=RunPhase.FAILED,
        identity={},
        site="grenoble",
        job_id=99,
    )

    assert AutonomousRunController._failed_run_location(current) == ("grenoble", 99)


def test_failed_run_resume_validation_preserves_non_resumable_and_exhausted_errors(
    tmp_path: Path,
) -> None:
    controller = AutonomousRunController(
        replace(_config(), max_continuations=1),
        state_root=tmp_path / "runs",
    )

    with pytest.raises(AutonomousRunError) as not_resumable:
        controller._validate_failed_run_resume({}, raw_count=0, raw_limit=1)
    assert str(not_resumable.value) == "run is not resumable from phase failed"

    exhausted_facts = {
        "error": "job ended without completion after 1 checkpoint continuations"
    }
    with pytest.raises(AutonomousRunError) as exhausted:
        controller._validate_failed_run_resume(
            exhausted_facts,
            raw_count=1,
            raw_limit=1,
        )
    assert str(exhausted.value) == (
        "failed run exhausted 1 checkpoint continuations; "
        "resume with --max-continuations greater than 1"
    )


def test_failed_run_resumability_helpers_preserve_count_and_error_boundaries(
    tmp_path: Path,
) -> None:
    controller = AutonomousRunController(_config(), state_root=tmp_path / "runs")
    recoverable = {"error": "job ended without a complete checkpoint"}
    stale = {
        "error": (
            "grenoble job 99 remained queued with no start-time prediction "
            "after 3 replacement rounds"
        )
    }

    assert controller._recoverable_checkpoint_failure(recoverable, 0)
    assert not controller._recoverable_checkpoint_failure(recoverable, 3)
    assert controller._stale_queued_failure(stale, 0)
    assert not controller._stale_queued_failure(stale, 3)
    assert controller._failed_run_is_exhausted(
        {"error": "job ended without completion after 1 checkpoint continuations"},
        1,
        1,
    )
    assert not controller._failed_run_is_exhausted(
        {"error": "job ended without completion after 1 checkpoint continuations"},
        0,
        1,
    )
    assert not controller._failed_run_is_exhausted(
        {"error": "different error"},
        1,
        1,
    )


def test_resume_loaded_phase_uses_an_empty_site_only_when_site_is_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    controller = AutonomousRunController(config, state_root=tmp_path / "runs")
    current = AutonomousRunState(
        run_id=config.identity.run_id,
        phase=RunPhase.SUBMITTED,
        identity=config.identity.canonical_payload,
        job_id=99,
    )
    requested_sites: list[str] = []
    remote = object()
    monkeypatch.setattr(
        controller,
        "remote_factory",
        lambda site: requested_sites.append(site) or remote,
    )
    monkeypatch.setattr(
        controller,
        "_monitor",
        lambda state, *, remote: state,
    )

    assert controller._resume_loaded_phase(current) is current
    assert requested_sites == [""]


def test_previous_job_status_preserves_success_and_wrapped_failure(
    tmp_path: Path,
) -> None:
    controller = AutonomousRunController(_config(), state_root=tmp_path / "runs")
    remote = _FakeRemote()

    status = controller._previous_job_status(remote, 99)

    assert status == JobStatus(job_id=99, state=JobState.QUEUED)

    class BrokenRemote:
        def raw(self, _command: str) -> CommandResult:
            raise RuntimeError("status transport failed")

    with pytest.raises(AutonomousRunError) as error:
        controller._previous_job_status(BrokenRemote(), 99)
    assert str(error.value) == "previous Grid'5000 job status could not be verified"
    assert isinstance(error.value.__cause__, Exception)


def test_candidate_list_forwards_fallback_and_requirements(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = AutonomousRunController(_config(), state_root=tmp_path / "runs")
    captured: dict[str, object] = {}
    sentinel = (object(),)

    def candidates(probes: object, **kwargs: object) -> tuple[object, ...]:
        captured.update({"probes": probes, **kwargs})
        return sentinel

    monkeypatch.setattr(autonomous, "replacement_candidates", candidates)
    probes = cast(tuple[SiteProbe, ...], (object(),))

    assert controller._candidate_list(probes, fallback_site="nancy") is sentinel
    assert captured == {
        "probes": probes,
        "fallback_site": "nancy",
        "requirements": controller.config.requirements,
    }


def test_fail_queued_job_cancels_and_forwards_the_stable_failure_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    controller = AutonomousRunController(config, state_root=tmp_path / "runs")
    current = AutonomousRunState(
        run_id=config.identity.run_id,
        phase=RunPhase.QUEUED,
        identity=config.identity.canonical_payload,
    )
    decision = QueuedReplacementDecision(
        action="fail",
        message="grenoble job 99 remained queued with no start-time prediction after 3 replacement rounds",
    )
    command_calls: list[str] = []
    continuation_calls: list[dict[str, object]] = []
    emitted: list[str] = []
    remote = SimpleNamespace(
        run=lambda command, **_kwargs: command_calls.append(command)
        or CommandResult(returncode=0, stdout=""),
    )
    sentinel = object()
    monkeypatch.setattr(controller, "emit", emitted.append)
    monkeypatch.setattr(
        controller,
        "_continue_after_incomplete",
        lambda _current, **kwargs: continuation_calls.append(kwargs) or sentinel,
    )

    result = controller._fail_queued_job(
        current,
        site="grenoble",
        job_id=99,
        remote=remote,
        decision=decision,
    )

    assert result is sentinel
    assert emitted == [f"{decision.message}; canceling stale fallback"]
    assert command_calls == ["oardel 99"]
    assert continuation_calls == [
        {
            "site": "grenoble",
            "job_id": 99,
            "remote": remote,
            "reason": decision.message,
            "failure_message": decision.message,
        }
    ]


def test_transition_persists_state_event_and_human_location(
    tmp_path: Path,
) -> None:
    config = _config()
    controller = AutonomousRunController(config, state_root=tmp_path / "runs")
    current = AutonomousRunState(
        run_id=config.identity.run_id,
        phase=RunPhase.CREATED,
        identity=config.identity.canonical_payload,
    )
    saved: list[AutonomousRunState] = []
    events: list[tuple[str, str, Mapping[str, object]]] = []
    emitted: list[str] = []
    controller.state.save = saved.append  # ty: ignore[invalid-assignment]
    controller.state.append_event = (  # ty: ignore[invalid-assignment]
        lambda run_id, phase, facts: events.append((run_id, phase, facts))
    )
    controller.emit = emitted.append

    facts = {"reason": "test"}
    updated = controller._transition(
        current,
        RunPhase.QUEUED,
        site="grenoble",
        job_id=99,
        facts=facts,
    )

    assert saved == [updated]
    assert events == [(config.identity.run_id, "queued", facts)]
    assert emitted == [
        f"run {config.identity.run_id}: phase=queued site=grenoble job=99"
    ]


def test_completion_identity_reports_the_invalid_manifest_field(
    tmp_path: Path,
) -> None:
    config = _config()
    controller = AutonomousRunController(config, state_root=tmp_path / "runs")
    manifest = dict(config.identity.canonical_payload)
    manifest["run_id"] = config.identity.run_id
    manifest["dataset_revision"] = "wrong"

    with pytest.raises(AutonomousRunError) as error:
        controller._verify_completion_identity(manifest)

    assert str(error.value) == "completion manifest dataset_revision is invalid"


@pytest.mark.parametrize(
    ("site", "job_id", "expected"),
    [
        (None, None, ("", "")),
        ("grenoble", None, (" site=grenoble", "")),
        (None, 99, ("", " job=99")),
        ("grenoble", 99, (" site=grenoble", " job=99")),
    ],
)
def test_transition_location_formats_each_optional_coordinate(
    site: str | None,
    job_id: int | None,
    expected: tuple[str, str],
) -> None:
    state = AutonomousRunState(
        run_id="a" * 20,
        phase="created",
        identity={},
        site=site,
        job_id=job_id,
    )

    assert AutonomousRunController._transition_location(state) == expected


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


def test_prepare_continuation_preserves_remote_resume_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_commit = "d" * 40
    config = replace(
        _config(),
        sites=("nancy", "nantes"),
        worker_source_commit=source_commit,
    )
    controller = AutonomousRunController(config, state_root=tmp_path / "runs")
    current = AutonomousRunState(
        run_id=config.identity.run_id,
        phase="failed",
        identity=config.identity.canonical_payload,
        site="nancy",
        job_id=99,
    )
    probe = SiteProbe(
        name="nantes",
        reachable=True,
        resources=(),
        persistent_free_bytes=10 * 1024**3,
        queued_jobs=0,
    )
    initial_remote = object()
    plan = cast(Grid5000Plan, object())
    probe_calls: list[str] = []
    remote_factory_calls: list[str] = []
    prepare_calls: list[dict[str, object]] = []
    token_calls: list[str] = []
    build_calls: list[tuple[object, bool]] = []
    preflight_calls: list[tuple[object, object]] = []
    emitted: list[str] = []

    class Remote:
        def prepare(self, **kwargs: object) -> None:
            prepare_calls.append(kwargs)

        def install_hugging_face_token(self, token: str) -> None:
            token_calls.append(token)

    remote = Remote()
    monkeypatch.setattr(controller, "emit", emitted.append)
    monkeypatch.setattr(
        controller,
        "_continuation_probe",
        lambda requested_site: probe_calls.append(requested_site) or probe,
    )
    monkeypatch.setattr(
        controller,
        "remote_factory",
        lambda site: remote_factory_calls.append(site) or remote,
    )
    monkeypatch.setattr(controller, "_local_token_for_publication", lambda: "hf-token")
    monkeypatch.setattr(
        controller,
        "_build_plan",
        lambda actual_probe, *, resume_from_checkpoint: (
            build_calls.append((actual_probe, resume_from_checkpoint)) or plan
        ),
    )
    monkeypatch.setattr(
        controller,
        "_preflight",
        lambda actual_remote, actual_plan: preflight_calls.append(
            (actual_remote, actual_plan)
        ),
    )

    actual_probe, actual_remote, actual_plan = controller._prepare_continuation(
        current,
        site="nancy",
        remote=initial_remote,
    )

    assert (actual_probe, actual_remote, actual_plan) == (
        probe,
        remote,
        plan,
    )
    assert probe_calls == ["nancy"]
    assert remote_factory_calls == ["nantes"]
    assert controller._active_remote is remote
    assert emitted == [
        f"run {config.identity.run_id}: continuing on nantes "
        "after nancy became unavailable"
    ]
    assert prepare_calls == [
        {
            "run_id": config.identity.run_id,
            "source_commit": source_commit,
            "allow_failed_run": True,
        }
    ]
    assert token_calls == ["hf-token"]
    assert build_calls == [(probe, True)]
    assert preflight_calls == [(remote, plan)]


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


def test_unmatched_replacement_job_ignores_a_malformed_attempt_count() -> None:
    assert (
        _replacement_attempt_count_for_job(
            {
                "replacement_attempted": False,
                "replacement_attempted_job_id": 99,
                "replacement_attempt_count": "not-an-integer",
            },
            job_id=99,
        )
        == 0
    )


def test_replacement_state_facts_require_the_same_site_and_job() -> None:
    matching = AutonomousRunController._replacement_state_facts(
        "lille",
        123,
        site="lille",
        job_id=123,
        attempt_count=2,
        attempt_timestamp="2026-08-21T10:00:00+00:00",
    )
    reset = AutonomousRunController._replacement_state_facts(
        "lille",
        124,
        site="lille",
        job_id=123,
        attempt_count=2,
        attempt_timestamp="2026-08-21T10:00:00+00:00",
    )

    assert matching == {
        "replacement_attempted": True,
        "replacement_attempted_job_id": 123,
        "replacement_attempt_count": 2,
        "replacement_last_attempt_at": "2026-08-21T10:00:00+00:00",
    }
    assert reset == {
        "replacement_attempted": False,
        "replacement_attempted_job_id": None,
        "replacement_attempt_count": 0,
        "replacement_last_attempt_at": None,
    }

    for replacement_site, replacement_job in (
        ("nancy", 123),
        ("nancy", 124),
    ):
        assert (
            AutonomousRunController._replacement_state_facts(
                replacement_site,
                replacement_job,
                site="lille",
                job_id=123,
                attempt_count=2,
                attempt_timestamp="2026-08-21T10:00:00+00:00",
            )
            == reset
        )


def test_queued_status_transitions_submitted_jobs_and_forwards_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    controller = AutonomousRunController(config, state_root=tmp_path / "runs")
    current = AutonomousRunState(
        run_id=config.identity.run_id,
        phase="submitted",
        identity=config.identity.canonical_payload,
        site="grenoble",
        job_id=99,
    )
    status = JobStatus(job_id=99, state=JobState.QUEUED)
    transitioned = replace(current, phase=RunPhase.QUEUED)
    transition_calls: list[tuple[object, RunPhase, dict[str, object]]] = []
    decision_calls: list[tuple[object, dict[str, object]]] = []
    remote = object()

    def transition(
        state: object,
        phase: RunPhase,
        **kwargs: object,
    ) -> AutonomousRunState:
        transition_calls.append((state, phase, kwargs))
        return transitioned

    def decision(actual_status: object, **kwargs: object) -> QueuedReplacementDecision:
        decision_calls.append((actual_status, kwargs))
        return QueuedReplacementDecision(action="wait")

    monkeypatch.setattr(controller, "_transition", transition)
    monkeypatch.setattr(controller, "_queued_replacement_decision", decision)

    result = controller._handle_queued_status(
        current,
        status=status,
        site="nantes",
        job_id=99,
        remote=remote,
    )

    assert result == (transitioned, "nantes", 99, remote)
    assert transition_calls == [(current, RunPhase.QUEUED, {})]
    assert len(decision_calls) == 1
    assert decision_calls[0][0] is status
    assert decision_calls[0][1]["site"] == "nantes"
    assert decision_calls[0][1]["job_id"] == 99
    assert decision_calls[0][1]["attempt_count"] == 0


def test_queued_status_failure_preserves_the_fallback_site(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    controller = AutonomousRunController(config, state_root=tmp_path / "runs")
    current = AutonomousRunState(
        run_id=config.identity.run_id,
        phase="queued",
        identity=config.identity.canonical_payload,
        site="grenoble",
        job_id=99,
    )
    status = JobStatus(job_id=99, state=JobState.QUEUED)
    decision = QueuedReplacementDecision(
        action="fail",
        message="nantes job 99 remained queued",
    )
    captured: dict[str, object] = {}
    sentinel = object()

    monkeypatch.setattr(
        controller,
        "_queued_replacement_decision",
        lambda *_args, **_kwargs: decision,
    )

    def fail_queued(*_args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(controller, "_fail_queued_job", fail_queued)

    result = controller._handle_queued_status(
        current,
        status=status,
        site="nantes",
        job_id=99,
        remote=object(),
    )

    assert result is sentinel
    assert captured["site"] == "nantes"
    assert captured["job_id"] == 99
    assert captured["decision"] is decision


@pytest.mark.parametrize(
    ("attempt_count", "seek_replacement", "expected_retry_due"),
    [
        (0, True, True),
        (0, False, False),
        (MAX_REPLACEMENT_ATTEMPTS, True, False),
    ],
)
def test_queued_replacement_decision_preserves_retry_boundaries(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attempt_count: int,
    seek_replacement: bool,
    expected_retry_due: bool,
) -> None:
    config = _config()
    controller = AutonomousRunController(config, state_root=tmp_path / "runs")
    status = JobStatus(job_id=99, state=JobState.QUEUED)
    now = datetime(2026, 8, 5, 19, 0, tzinfo=UTC)
    retry_calls: list[dict[str, object]] = []
    decision_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        autonomous,
        "should_seek_replacement",
        lambda actual_status, *, now: (
            actual_status is status and now == datetime(2026, 8, 5, 19, 0, tzinfo=UTC)
        )
        and seek_replacement,
    )

    def retry_due(_facts: object, **kwargs: object) -> bool:
        kwargs = {"facts": _facts, **kwargs}
        retry_calls.append(kwargs)
        return True

    monkeypatch.setattr(autonomous, "_replacement_retry_due_for_job", retry_due)
    sentinel = QueuedReplacementDecision(action="wait")

    def decide(_status: object, **kwargs: object) -> QueuedReplacementDecision:
        decision_calls.append(kwargs)
        return sentinel

    monkeypatch.setattr(autonomous, "decide_queued_replacement", decide)

    result = controller._queued_replacement_decision(
        status,
        site="grenoble",
        job_id=99,
        now=now,
        facts={"example": True},
        attempt_count=attempt_count,
    )

    assert result is sentinel
    assert decision_calls == [
        {
            "site": "grenoble",
            "job_id": 99,
            "now": now,
            "attempt_count": attempt_count,
            "retry_due": expected_retry_due,
            "max_attempts": MAX_REPLACEMENT_ATTEMPTS,
        }
    ]
    if expected_retry_due:
        assert retry_calls == [{"facts": {"example": True}, "job_id": 99, "now": now}]
    else:
        assert retry_calls == []


def test_terminal_failure_forwards_reason_and_job_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    controller = AutonomousRunController(config, state_root=tmp_path / "runs")
    current = AutonomousRunState(
        run_id=config.identity.run_id,
        phase="running",
        identity=config.identity.canonical_payload,
        site="grenoble",
        job_id=99,
    )
    status = JobStatus(job_id=99, state=JobState.ERROR, exit_code=17)
    captured: dict[str, object] = {}
    sentinel = object()

    monkeypatch.setattr(controller, "_terminal_job_failed", lambda _: True)

    def continue_after_incomplete(*_args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(
        controller,
        "_continue_after_incomplete",
        continue_after_incomplete,
    )

    result = controller._handle_terminal_status(
        current,
        status=status,
        site="nantes",
        job_id=123,
        remote=object(),
    )

    assert result is sentinel
    assert captured["site"] == "nantes"
    assert captured["job_id"] == 123
    assert captured["reason"] == "job ended as error; exit_code=17"


def test_terminal_success_forwards_site_and_job_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    controller = AutonomousRunController(config, state_root=tmp_path / "runs")
    current = AutonomousRunState(
        run_id=config.identity.run_id,
        phase="running",
        identity=config.identity.canonical_payload,
        site="grenoble",
        job_id=99,
    )
    status = JobStatus(job_id=99, state=JobState.TERMINATED, exit_code=0)
    captured: dict[str, object] = {}
    sentinel = object()

    monkeypatch.setattr(controller, "_terminal_job_failed", lambda _: False)

    def complete_terminal_job(*_args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(controller, "_complete_terminal_job", complete_terminal_job)

    result = controller._handle_terminal_status(
        current,
        status=status,
        site="nantes",
        job_id=123,
        remote=object(),
    )

    assert result is sentinel
    assert captured["site"] == "nantes"
    assert captured["job_id"] == 123


def test_monitor_status_forwards_running_context_and_keeps_updated_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    controller = AutonomousRunController(config, state_root=tmp_path / "runs")
    current = AutonomousRunState(
        run_id=config.identity.run_id,
        phase="queued",
        identity=config.identity.canonical_payload,
        site="grenoble",
        job_id=99,
    )
    status = JobStatus(job_id=123, state=JobState.RUNNING)
    remote = object()
    updated = object()
    calls: list[tuple[object, object, str, int]] = []

    def running_state(
        actual_current: object,
        actual_status: object,
        *,
        site: str,
        job_id: int,
    ) -> object:
        calls.append((actual_current, actual_status, site, job_id))
        return updated

    monkeypatch.setattr(controller, "_running_state", running_state)

    result = controller._monitor_status(
        current,
        status=status,
        site="nantes",
        job_id=123,
        remote=remote,
    )

    assert result == (updated, "nantes", 123, remote)
    assert calls == [(current, status, "nantes", 123)]


def test_monitor_status_forwards_terminal_job_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    controller = AutonomousRunController(config, state_root=tmp_path / "runs")
    current = AutonomousRunState(
        run_id=config.identity.run_id,
        phase="running",
        identity=config.identity.canonical_payload,
        site="grenoble",
        job_id=99,
    )
    status = JobStatus(job_id=123, state=JobState.TERMINATED, exit_code=0)
    remote = object()
    captured: dict[str, object] = {}
    sentinel = object()

    def terminal_status(*_args: object, **kwargs: object) -> object:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(controller, "_handle_terminal_status", terminal_status)

    result = controller._monitor_status(
        current,
        status=status,
        site="nantes",
        job_id=123,
        remote=remote,
    )

    assert result is sentinel
    assert captured["site"] == "nantes"
    assert captured["job_id"] == 123
    assert captured["remote"] is remote


def test_monitor_updates_remote_after_replacement_and_sleeps_between_polls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    remote = object()
    replacement_remote = object()
    current = AutonomousRunState(
        run_id=config.identity.run_id,
        phase="running",
        identity=config.identity.canonical_payload,
        site="grenoble",
        job_id=99,
    )
    replacement_state = replace(current, phase=RunPhase.SUBMITTED)
    completed_state = replace(current, phase=RunPhase.COMPLETED)
    status_calls: list[tuple[object, int]] = []
    client_remotes: list[object] = []
    sleeps: list[float] = []
    monitor_calls: list[tuple[object, object, str, int, object]] = []

    class Oar:
        def __init__(self, actual_remote: object) -> None:
            client_remotes.append(actual_remote)

        def status(self, job_id: int) -> JobStatus:
            status_calls.append((client_remotes[-1], job_id))
            return JobStatus(job_id=job_id, state=JobState.RUNNING)

    def monitor_status(
        actual_current: object,
        *,
        status: object,
        site: str,
        job_id: int,
        remote: object,
    ) -> object:
        monitor_calls.append((actual_current, status, site, job_id, remote))
        if len(monitor_calls) == 1:
            return replacement_state, "nantes", 123, replacement_remote
        return completed_state

    controller = AutonomousRunController(
        config,
        state_root=tmp_path / "runs",
        poll_seconds=2,
        sleeper=sleeps.append,
        emit=lambda _message: None,
    )
    monkeypatch.setattr(autonomous, "OarClient", Oar)
    monkeypatch.setattr(controller, "_monitor_status", monitor_status)

    result = controller._monitor(current, remote=remote)

    assert result is completed_state
    assert client_remotes == [remote, replacement_remote]
    assert status_calls == [(remote, 99), (replacement_remote, 123)]
    assert sleeps == [2]
    assert [call[2:] for call in monitor_calls] == [
        ("grenoble", 99, remote),
        ("nantes", 123, replacement_remote),
    ]


@pytest.mark.parametrize(
    ("site", "job_id"),
    [(None, 99), ("grenoble", None)],
    ids=["missing-site", "missing-job"],
)
def test_monitor_location_rejects_each_incomplete_coordinate(
    site: str | None,
    job_id: int | None,
) -> None:
    state = AutonomousRunState(
        run_id="a" * 20,
        phase="submitted",
        identity={},
        site=site,
        job_id=job_id,
    )

    with pytest.raises(AutonomousRunError) as error:
        AutonomousRunController._monitor_location(state)

    assert str(error.value) == "submitted state lacks site or job ID"


def test_replace_queued_job_records_attempt_and_replacement_location(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    emitted: list[str] = []
    controller = AutonomousRunController(
        config,
        state_root=tmp_path / "runs",
        emit=emitted.append,
    )
    current = AutonomousRunState(
        run_id=config.identity.run_id,
        phase="queued",
        identity=config.identity.canonical_payload,
        site="grenoble",
        job_id=99,
    )
    fallback_remote = object()
    replacement_remote = object()
    attempt_timestamp = "2026-08-21T10:00:00+00:00"
    replacement_calls: list[dict[str, object]] = []
    transitions: list[tuple[object, object, dict[str, object]]] = []

    def try_replacement(
        *,
        fallback_site: str,
        fallback_job_id: int,
        fallback_remote: object,
        resume_from_checkpoint: bool,
    ) -> tuple[str, int, object]:
        replacement_calls.append(
            {
                "fallback_site": fallback_site,
                "fallback_job_id": fallback_job_id,
                "fallback_remote": fallback_remote,
                "resume_from_checkpoint": resume_from_checkpoint,
            }
        )
        return "nantes", 123, replacement_remote

    def transition(
        state: object,
        phase: object,
        **facts: object,
    ) -> AutonomousRunState:
        transitions.append((state, phase, facts))
        return current

    monkeypatch.setattr(controller, "_try_replacement", try_replacement)
    monkeypatch.setattr(controller, "_transition", transition)

    result = controller._replace_queued_job(
        current,
        site="grenoble",
        job_id=99,
        remote=fallback_remote,
        facts={"resume_from_checkpoint": True},
        decision=QueuedReplacementDecision(
            action="replace",
            attempt_count=2,
            attempt_timestamp=attempt_timestamp,
        ),
    )

    assert result == (current, "nantes", 123, replacement_remote)
    assert emitted == ["grenoble job 99: replacement round 2/3"]
    assert replacement_calls == [
        {
            "fallback_site": "grenoble",
            "fallback_job_id": 99,
            "fallback_remote": fallback_remote,
            "resume_from_checkpoint": True,
        }
    ]
    assert transitions == [
        (
            current,
            RunPhase.QUEUED,
            {
                "facts": {
                    "replacement_attempted": True,
                    "replacement_attempted_job_id": 99,
                    "replacement_attempt_count": 2,
                    "replacement_last_attempt_at": attempt_timestamp,
                }
            },
        ),
        (
            current,
            RunPhase.QUEUED,
            {
                "site": "nantes",
                "job_id": 123,
                "facts": {
                    "replacement_attempted": False,
                    "replacement_attempted_job_id": None,
                    "replacement_attempt_count": 0,
                    "replacement_last_attempt_at": None,
                },
            },
        ),
    ]


@pytest.mark.parametrize(
    ("attempt_count", "attempt_timestamp"),
    [(None, "2026-08-21T10:00:00+00:00"), (2, None)],
    ids=["missing-count", "missing-timestamp"],
)
def test_replace_queued_job_rejects_missing_attempt_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    attempt_count: int | None,
    attempt_timestamp: str | None,
) -> None:
    config = _config()
    controller = AutonomousRunController(
        config,
        state_root=tmp_path / "runs",
    )
    current = AutonomousRunState(
        run_id=config.identity.run_id,
        phase=RunPhase.QUEUED,
        identity=config.identity.canonical_payload,
        site="grenoble",
        job_id=99,
    )

    def replacement_should_not_start(**kwargs: object) -> NoReturn:
        raise AssertionError("replacement must not start")

    def transition_should_not_run(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("state must not transition")

    monkeypatch.setattr(controller, "_try_replacement", replacement_should_not_start)
    monkeypatch.setattr(controller, "_transition", transition_should_not_run)

    with pytest.raises(AutonomousRunError) as error:
        controller._replace_queued_job(
            current,
            site="grenoble",
            job_id=99,
            remote=object(),
            facts={},
            decision=QueuedReplacementDecision(
                action="replace",
                attempt_count=attempt_count,
                attempt_timestamp=attempt_timestamp,
            ),
        )

    assert str(error.value) == "replacement decision is missing attempt metadata"


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


@pytest.mark.parametrize("cleanup", [True, False])
def test_complete_terminal_job_persists_verified_completion_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cleanup: bool,
) -> None:
    config = replace(_config(), cleanup=cleanup)
    controller = AutonomousRunController(config, state_root=tmp_path / "runs")
    remote = _FakeRemote()
    current = AutonomousRunState(
        run_id=config.identity.run_id,
        phase="running",
        identity=config.identity.canonical_payload,
        site="grenoble",
        job_id=99,
    )
    manifest = {"schema_version": 1, "run_id": config.identity.run_id}
    read_calls: list[str] = []
    verification_calls: list[tuple[object, object]] = []
    status_calls: list[tuple[object, object]] = []
    cleanup_calls: list[object] = []
    transitions: list[tuple[object, object, dict[str, object]]] = []

    monkeypatch.setattr(
        remote,
        "read_completion",
        lambda run_id: read_calls.append(run_id) or manifest,
    )
    monkeypatch.setattr(
        controller,
        "_verify_completion",
        lambda actual_remote, actual_manifest: verification_calls.append(
            (actual_remote, actual_manifest)
        ),
    )
    monkeypatch.setattr(
        remote,
        "mark_status",
        lambda run_id, status: status_calls.append((run_id, status)),
    )
    monkeypatch.setattr(
        remote,
        "cleanup",
        lambda run_id: cleanup_calls.append(run_id),
    )

    def transition(
        state: object,
        phase: object,
        **facts: object,
    ) -> AutonomousRunState:
        transitions.append((state, phase, facts))
        return current

    monkeypatch.setattr(controller, "_transition", transition)

    result = controller._complete_terminal_job(
        current,
        site="grenoble",
        job_id=99,
        remote=remote,
    )

    assert result is current
    assert read_calls == [config.identity.run_id]
    assert verification_calls == [(remote, manifest)]
    assert status_calls == [(config.identity.run_id, "complete")]
    assert cleanup_calls == ([config.identity.run_id] if cleanup else [])
    assert transitions == [
        (
            current,
            RunPhase.COMPLETED,
            {
                "site": "grenoble",
                "job_id": 99,
                "facts": {"completion": manifest, "cleanup": cleanup},
            },
        )
    ]


def test_complete_terminal_job_routes_manifest_errors_to_terminal_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    controller = AutonomousRunController(config, state_root=tmp_path / "runs")
    remote = _FakeRemote()
    current = AutonomousRunState(
        run_id=config.identity.run_id,
        phase="running",
        identity=config.identity.canonical_payload,
        site="grenoble",
        job_id=99,
    )
    failure_calls: list[tuple[object, dict[str, object]]] = []

    monkeypatch.setattr(
        controller,
        "_verify_completion",
        lambda _remote, _manifest: (_ for _ in ()).throw(
            AutonomousRunError("manifest is invalid")
        ),
    )

    def fail_terminal(
        state: object,
        *,
        site: str,
        job_id: int,
        remote: object,
        message: str,
    ) -> AutonomousRunState:
        failure_calls.append(
            (
                state,
                {
                    "site": site,
                    "job_id": job_id,
                    "remote": remote,
                    "message": message,
                },
            )
        )
        return current

    monkeypatch.setattr(controller, "_fail_terminal", fail_terminal)

    assert (
        controller._complete_terminal_job(
            current,
            site="grenoble",
            job_id=99,
            remote=remote,
        )
        is current
    )
    assert failure_calls == [
        (
            current,
            {
                "site": "grenoble",
                "job_id": 99,
                "remote": remote,
                "message": "manifest is invalid",
            },
        )
    ]


def test_complete_terminal_job_routes_unexpected_read_errors_to_continuation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    controller = AutonomousRunController(config, state_root=tmp_path / "runs")
    remote = _FakeRemote()
    current = AutonomousRunState(
        run_id=config.identity.run_id,
        phase="running",
        identity=config.identity.canonical_payload,
        site="grenoble",
        job_id=99,
    )
    continuation_calls: list[tuple[object, dict[str, object]]] = []

    monkeypatch.setattr(
        remote,
        "read_completion",
        lambda _run_id: (_ for _ in ()).throw(RuntimeError("manifest unavailable")),
    )

    def continue_after_incomplete(
        state: object,
        *,
        site: str,
        job_id: int,
        remote: object,
        reason: str,
    ) -> AutonomousRunState:
        continuation_calls.append(
            (
                state,
                {
                    "site": site,
                    "job_id": job_id,
                    "remote": remote,
                    "reason": reason,
                },
            )
        )
        return current

    monkeypatch.setattr(
        controller,
        "_continue_after_incomplete",
        continue_after_incomplete,
    )

    assert (
        controller._complete_terminal_job(
            current,
            site="grenoble",
            job_id=99,
            remote=remote,
        )
        is current
    )
    assert continuation_calls == [
        (
            current,
            {
                "site": "grenoble",
                "job_id": 99,
                "remote": remote,
                "reason": "manifest unavailable",
            },
        )
    ]


def test_submit_continuation_records_pending_and_submitted_state_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(
        _config(),
        max_continuations=5,
        policy_type="day",
        worker_source_commit="e" * 40,
    )
    controller = AutonomousRunController(config, state_root=tmp_path / "runs")
    current = AutonomousRunState(
        run_id=config.identity.run_id,
        phase="running",
        identity=config.identity.canonical_payload,
        site="grenoble",
        job_id=77,
    )
    probe = SiteProbe(
        name="lille",
        reachable=True,
        resources=(),
        persistent_free_bytes=0,
        queued_jobs=0,
    )
    remote = _FakeRemote()
    scheduler_command = ("oarsub", "-l", "nodes=1")
    plan = cast(
        Grid5000Plan,
        SimpleNamespace(scheduler_command=scheduler_command),
    )
    submitted_commands: list[tuple[str, ...]] = []
    transitions: list[tuple[object, object, dict[str, object]]] = []
    monitor_calls: list[tuple[object, object]] = []

    def submit(_client: object, command: tuple[str, ...]) -> int:
        submitted_commands.append(command)
        return 123

    monkeypatch.setattr(autonomous.OarClient, "submit", submit)

    def transition(
        state: object,
        phase: object,
        **facts: object,
    ) -> AutonomousRunState:
        transitions.append((state, phase, facts))
        return current

    monkeypatch.setattr(controller, "_transition", transition)

    def monitor(state: object, *, remote: object) -> AutonomousRunState:
        monitor_calls.append((state, remote))
        return current

    monkeypatch.setattr(controller, "_monitor", monitor)

    result = controller._submit_continuation(
        current,
        probe=probe,
        remote=remote,
        plan=plan,
        raw_count=2,
        reason="walltime",
        last_terminal_job_id=77,
    )

    assert result is current
    assert submitted_commands == [scheduler_command]
    assert monitor_calls == [(current, remote)]
    assert transitions == [
        (
            current,
            RunPhase.SUBMITTING,
            {
                "site": "lille",
                "job_id": None,
                "facts": {
                    "continuation_count": 3,
                    "max_continuations": 5,
                    "worker_source_commit": "e" * 40,
                    "continuation_pending": True,
                    "continuation_reason": "walltime",
                    "last_terminal_job_id": 77,
                    "requested_policy_type": "day",
                    "replacement_attempted": False,
                    "replacement_attempted_job_id": None,
                    "replacement_attempt_count": 0,
                    "replacement_last_attempt_at": None,
                },
            },
        ),
        (
            current,
            RunPhase.SUBMITTED,
            {
                "site": "lille",
                "job_id": 123,
                "facts": {
                    "continuation_count": 3,
                    "max_continuations": 5,
                    "worker_source_commit": "e" * 40,
                    "continuation_pending": False,
                    "resume_from_checkpoint": True,
                    "scheduler_command": list(scheduler_command),
                    "requested_policy_type": "day",
                    "replacement_attempted": False,
                    "replacement_attempted_job_id": None,
                    "replacement_attempt_count": 0,
                    "replacement_last_attempt_at": None,
                },
            },
        ),
    ]


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

    with pytest.raises(AutonomousRunError) as error:
        controller.run()

    assert str(error.value) == (
        "failed run still has an active Grid'5000 job; refusing a duplicate"
    )
    assert remote.submission_count == 0


def test_failed_run_extension_preserves_remote_and_continuation_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = replace(_config(), max_continuations=2)
    controller = AutonomousRunController(config, state_root=tmp_path / "runs")
    current = AutonomousRunState(
        run_id=config.identity.run_id,
        phase=RunPhase.FAILED,
        identity=config.identity.canonical_payload,
        site="grenoble",
        job_id=99,
        facts={
            "continuation_count": 1,
            "max_continuations": 1,
            "error": "job ended without completion after 1 checkpoint continuations",
        },
    )
    remote = object()
    remote_sites: list[str] = []
    emitted: list[str] = []
    continuation: dict[str, object] = {}
    sentinel = object()

    def remote_factory(site: str) -> object:
        remote_sites.append(site)
        return remote

    def continue_after_incomplete(
        current_arg: AutonomousRunState,
        *,
        site: str,
        job_id: int,
        remote: object,
        reason: str,
    ) -> object:
        continuation.update(
            current=current_arg,
            site=site,
            job_id=job_id,
            remote=remote,
            reason=reason,
        )
        return sentinel

    monkeypatch.setattr(controller, "remote_factory", remote_factory)
    monkeypatch.setattr(
        controller,
        "_previous_job_status",
        lambda _remote, job_id: JobStatus(
            job_id=job_id,
            state=JobState.TERMINATED,
        ),
    )
    monkeypatch.setattr(controller, "emit", emitted.append)
    monkeypatch.setattr(
        controller,
        "_continue_after_incomplete",
        continue_after_incomplete,
    )

    result = controller._resume_failed_run(current)

    assert result is sentinel
    assert remote_sites == ["grenoble"]
    assert controller._active_remote is remote
    assert emitted == [
        f"run {config.identity.run_id}: extending from the retained checkpoint "
        "after job 99"
    ]
    assert continuation == {
        "current": current,
        "site": "grenoble",
        "job_id": 99,
        "remote": remote,
        "reason": "explicit continuation extension",
    }


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

    with pytest.raises(AutonomousRunError) as error:
        controller.run()

    assert str(error.value) == (
        "container settings differ from the persisted run; start a new run instead"
    )


def test_resume_accepts_matching_persisted_container_settings(
    tmp_path: Path,
) -> None:
    image = "registry.example/osm-polygon-sentence-classifier@sha256:" + "a" * 64
    config = replace(_config(), container_image=image, container_runtime="docker")
    controller = AutonomousRunController(
        config,
        state_root=tmp_path / "runs",
    )
    state = AutonomousRunState(
        run_id=config.identity.run_id,
        phase=RunPhase.FAILED,
        identity=config.identity.canonical_payload,
        facts={
            "container_image": image,
            "container_runtime": "docker",
        },
    )

    controller._validate_persisted_container_settings(state)


def test_resume_rejects_a_non_string_persisted_container_image(
    tmp_path: Path,
) -> None:
    image = "registry.example/osm-polygon-sentence-classifier@sha256:" + "a" * 64
    config = replace(_config(), container_image=image, container_runtime="docker")
    controller = AutonomousRunController(
        config,
        state_root=tmp_path / "runs",
    )
    state = AutonomousRunState(
        run_id=config.identity.run_id,
        phase=RunPhase.FAILED,
        identity=config.identity.canonical_payload,
        facts={
            "container_image": 123,
            "container_runtime": "docker",
        },
    )

    with pytest.raises(AutonomousRunError) as error:
        controller._validate_persisted_container_settings(state)

    assert str(error.value) == "persisted container image is invalid"


def test_published_checkpoint_probe_skips_hub_when_publication_is_disabled(
    tmp_path: Path,
) -> None:
    config = replace(
        _config(),
        training_config=replace(_config().training_config, publish_to_hub=False),
    )
    hub = _FakeHub()
    controller = AutonomousRunController(
        config,
        state_root=tmp_path / "runs",
        hub_api=hub,
    )

    assert controller._probe_published_checkpoint(site="grenoble", job_id=99) is False
    assert hub.calls == []


def test_published_checkpoint_probe_returns_false_when_no_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    hub = _FakeHub()
    calls: list[tuple[object, str, object]] = []

    def latest_checkpoint(
        identity: object,
        *,
        repository_id: str,
        hub_api: object,
    ) -> None:
        calls.append((identity, repository_id, hub_api))
        return None

    monkeypatch.setattr(
        "osm_polygon_sentence_classifier.grid5000_autonomous.latest_published_checkpoint",
        latest_checkpoint,
    )
    controller = AutonomousRunController(
        config,
        state_root=tmp_path / "runs",
        hub_api=hub,
    )

    assert controller._probe_published_checkpoint(site="grenoble", job_id=99) is False
    assert calls == [
        (
            config.identity.canonical_payload,
            "NoeFlandre/osm-polygon-sentence-classifier",
            hub,
        )
    ]


def test_published_checkpoint_probe_reports_a_published_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    hub = _FakeHub()
    checkpoint = PublishedCheckpoint(
        repository_id="NoeFlandre/osm-polygon-sentence-classifier",
        prefix="studies/landuse-v1/run-test",
        step=12,
        files=(),
    )
    calls: list[tuple[object, str, object]] = []

    def latest_checkpoint(
        identity: object,
        *,
        repository_id: str,
        hub_api: object,
    ) -> PublishedCheckpoint:
        calls.append((identity, repository_id, hub_api))
        return checkpoint

    monkeypatch.setattr(
        "osm_polygon_sentence_classifier.grid5000_autonomous.latest_published_checkpoint",
        latest_checkpoint,
    )
    emitted: list[str] = []
    controller = AutonomousRunController(
        config,
        state_root=tmp_path / "runs",
        hub_api=hub,
        emit=emitted.append,
    )

    assert controller._probe_published_checkpoint(site="grenoble", job_id=99) is True
    assert calls == [
        (
            config.identity.canonical_payload,
            "NoeFlandre/osm-polygon-sentence-classifier",
            hub,
        )
    ]
    assert emitted == [
        "grenoble job 99: using published checkpoint at step 12",
    ]


def test_published_checkpoint_probe_wraps_hub_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def latest_checkpoint(*args: object, **kwargs: object) -> None:
        raise HubCheckpointError("hub unavailable")

    monkeypatch.setattr(
        "osm_polygon_sentence_classifier.grid5000_autonomous.latest_published_checkpoint",
        latest_checkpoint,
    )
    controller = AutonomousRunController(
        _config(),
        state_root=tmp_path / "runs",
        hub_api=_FakeHub(),
    )

    with pytest.raises(AutonomousRunError) as error:
        controller._probe_published_checkpoint(site="grenoble", job_id=99)

    assert str(error.value) == "published checkpoint availability could not be verified"
    assert isinstance(error.value.__cause__, HubCheckpointError)


def test_local_checkpoint_probe_passes_the_complete_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    controller = AutonomousRunController(
        config,
        state_root=tmp_path / "runs",
        poll_seconds=17.0,
    )
    remote = object()
    calls: list[tuple[object, dict[str, object]]] = []

    def probe_complete_checkpoint(
        actual_remote: object,
        **kwargs: object,
    ) -> bool:
        calls.append((actual_remote, kwargs))
        return True

    monkeypatch.setattr(
        "osm_polygon_sentence_classifier.grid5000_autonomous.probe_complete_checkpoint",
        probe_complete_checkpoint,
    )

    assert (
        controller._probe_local_checkpoint(
            remote,
            site="grenoble",
            job_id=99,
            allow_failed_status=True,
        )
        is True
    )
    assert calls == [
        (
            remote,
            {
                "run_id": config.identity.run_id,
                "output_subdirectory": config.training_config.output_subdirectory,
                "identity": config.identity.canonical_payload,
                "allow_failed_status": True,
                "site": "grenoble",
                "job_id": 99,
                "poll_seconds": 17.0,
                "emit": controller.emit,
                "sleep": controller.sleeper,
                "attempts": autonomous.CHECKPOINT_PROBE_ATTEMPTS,
                "retry_seconds": autonomous.CHECKPOINT_PROBE_RETRY_SECONDS,
            },
        )
    ]


@pytest.mark.parametrize("with_cause", [False, True], ids=["direct", "chained"])
def test_local_checkpoint_probe_preserves_probe_error_message_and_cause(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    with_cause: bool,
) -> None:
    underlying = ValueError("underlying checkpoint failure")

    def probe_complete_checkpoint(*args: object, **kwargs: object) -> bool:
        if with_cause:
            raise CheckpointProbeError("checkpoint failure") from underlying
        raise CheckpointProbeError("checkpoint failure")

    monkeypatch.setattr(
        "osm_polygon_sentence_classifier.grid5000_autonomous.probe_complete_checkpoint",
        probe_complete_checkpoint,
    )
    controller = AutonomousRunController(
        _config(),
        state_root=tmp_path / "runs",
    )

    with pytest.raises(AutonomousRunError) as error:
        controller._probe_local_checkpoint(
            object(),
            site="grenoble",
            job_id=99,
            allow_failed_status=False,
        )

    assert str(error.value) == "checkpoint failure"
    assert (error.value.__cause__ is underlying) is with_cause


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
    monkeypatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
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


def test_try_replacement_builds_the_complete_context_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requirements = SiteRequirements(
        gpu_memory_mb=16_000,
        persistent_free_bytes=3 * 1024**3,
    )
    source_commit = "d" * 40
    config = replace(
        _config(),
        sites=("nancy", "nantes"),
        requirements=requirements,
        max_workers=5,
        cleanup=False,
        worker_source_commit=source_commit,
    )
    controller = AutonomousRunController(
        config,
        state_root=tmp_path / "runs",
        poll_seconds=0,
    )
    captured_contexts: list[ReplacementContext] = []
    run_kwargs: dict[str, object] = {}
    build_calls: list[tuple[object, int | None, bool]] = []
    plan_marker = cast(Grid5000Plan, object())

    def build_plan(
        probe: object,
        *,
        walltime_seconds: int | None = None,
        resume_from_checkpoint: bool = False,
    ) -> Grid5000Plan:
        build_calls.append((probe, walltime_seconds, resume_from_checkpoint))
        return plan_marker

    class Coordinator:
        def __init__(self, context: ReplacementContext) -> None:
            captured_contexts.append(context)

        def run(self, **kwargs: object) -> tuple[str, int, object]:
            run_kwargs.update(kwargs)
            return "nantes", 123, replacement_remote

    replacement_remote = object()
    monkeypatch.setattr(controller, "_build_plan", build_plan)
    monkeypatch.setattr(autonomous, "ReplacementCoordinator", Coordinator)
    fallback_remote = object()

    result = controller._try_replacement(
        fallback_site="nancy",
        fallback_job_id=99,
        fallback_remote=fallback_remote,
    )

    assert result == ("nantes", 123, replacement_remote)
    assert len(captured_contexts) == 1
    context = captured_contexts[0]
    assert context.run_id == config.identity.run_id
    assert context.source_commit == source_commit
    assert context.sites == ("nancy", "nantes")
    assert context.requirements is requirements
    assert context.max_workers == 5
    assert context.cleanup is False
    assert context.poll_seconds == 1.0
    assert context.walltime_seconds == config.walltime_seconds
    assert context.sleep is controller.sleeper
    probe = cast(SiteProbe, object())
    assert context.build_plan(probe, 900) is plan_marker
    assert build_calls == [(probe, 900, False)]
    assert run_kwargs == {
        "fallback_site": "nancy",
        "fallback_job_id": 99,
        "fallback_remote": fallback_remote,
    }


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
