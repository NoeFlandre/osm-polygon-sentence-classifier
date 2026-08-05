import json
from dataclasses import replace
from pathlib import Path

import pytest

from osm_polygon_sentence_classifier.grid5000 import (
    CommandResult,
    Grid5000RunIdentity,
)
from osm_polygon_sentence_classifier.grid5000_autonomous import (
    AutonomousRunConfig,
    AutonomousRunController,
    AutonomousRunError,
)
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
        self.installed_token: str | None = None
        self.cleaned = False
        self.marked: list[str] = []
        self.status_calls = 0

    def prepare(self, *, run_id: str, source_commit: str) -> RemotePreparationResult:
        del source_commit
        self.prepared = True
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
            "tracking_space_id": ("NoeFlandre/osm-polygon-sentence-classifier-trackio"),
        }

    def mark_status(self, run_id: str, status: str) -> None:
        del run_id
        self.marked.append(status)

    def cleanup(self, run_id: str) -> None:
        del run_id
        self.cleaned = True


class _ReplacementRemote:
    def __init__(self, site: str, state: str) -> None:
        self.site = site
        self.state = state
        self.cancelled: list[int] = []

    def prepare(self, *, run_id: str, source_commit: str) -> RemotePreparationResult:
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


def test_controller_runs_prepare_submit_monitor_publish_verify_and_cleanup(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    remote = _FakeRemote()
    hub = _FakeHub()
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
                "tracking_space_id": (
                    "NoeFlandre/osm-polygon-sentence-classifier-trackio"
                ),
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

    site, job_id, selected_remote = controller._try_replacement(
        fallback_site="nancy",
        fallback_job_id=10,
        fallback_remote=fallback,
    )

    assert (site, job_id) == ("grenoble", 11)
    assert selected_remote is candidate


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
