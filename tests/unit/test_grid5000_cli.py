import json
from types import SimpleNamespace
from typing import cast

from osm_polygon_sentence_classifier import ablation_study, grid5000_cli
from osm_polygon_sentence_classifier.grid5000 import (
    Grid5000Plan,
    Grid5000RunIdentity,
    Grid5000Submission,
)
from osm_polygon_sentence_classifier.grid5000_state import AutonomousRunState

SOURCE_COMMIT = "a" * 40
MODEL_REVISION = "b" * 40


def _arguments(*extra: str) -> list[str]:
    return [
        "--site",
        "nancy",
        "--source-commit",
        SOURCE_COMMIT,
        "--model-revision",
        MODEL_REVISION,
        *extra,
    ]


def test_plan_command_prints_a_reproducible_plan_without_an_operator(
    monkeypatch, capsys
) -> None:
    def unexpected_operator(*args: object, **kwargs: object) -> None:
        raise AssertionError("plan mode must not construct an executing operator")

    monkeypatch.setattr(grid5000_cli, "Grid5000Operator", unexpected_operator)

    exit_code = grid5000_cli.main(["plan", *_arguments()])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"]
    assert "usagepolicycheck" not in payload["submission_command"]
    assert "oarsub" in payload["submission_command"]
    assert payload["identity"]["model_name_or_path"] == "jhu-clsp/mmBERT-small"


def test_submit_without_execute_calls_only_the_plan_path(monkeypatch, capsys) -> None:
    calls: list[bool] = []

    class FakeOperator:
        def __init__(self, plan: Grid5000Plan) -> None:
            self.plan = plan

        def submit(self, *, execute: bool = False) -> Grid5000Submission:
            calls.append(execute)
            return Grid5000Submission(plan=self.plan, executed=execute)

    monkeypatch.setattr(grid5000_cli, "Grid5000Operator", FakeOperator)

    exit_code = grid5000_cli.main(["submit", *_arguments()])

    assert exit_code == 0
    assert calls == [False]
    assert json.loads(capsys.readouterr().out)["executed"] is False


def test_submit_execute_is_the_only_explicit_execution_gate(
    monkeypatch, capsys
) -> None:
    calls: list[bool] = []

    class FakeOperator:
        def __init__(self, plan: Grid5000Plan) -> None:
            self.plan = plan

        def submit(self, *, execute: bool = False) -> Grid5000Submission:
            calls.append(execute)
            return Grid5000Submission(plan=self.plan, executed=execute, job_id=7)

    monkeypatch.setattr(grid5000_cli, "Grid5000Operator", FakeOperator)

    exit_code = grid5000_cli.main(["submit", *_arguments("--execute")])

    assert exit_code == 0
    assert calls == [True]
    assert json.loads(capsys.readouterr().out)["job_id"] == 7


def test_plan_can_explicitly_enable_final_publication_and_trackio_sync(
    capsys,
) -> None:
    exit_code = grid5000_cli.main(["plan", *_arguments("--publish", "--sync-trackio")])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["identity"]["training_config"]["publish_to_hub"] is True
    assert payload["identity"]["training_config"]["sync_trackio"] is True


def test_plan_can_request_a_policy_bounded_day_allocation(capsys) -> None:
    exit_code = grid5000_cli.main(
        [
            "plan",
            *_arguments(
                "--policy-type",
                "day",
                "--walltime-seconds",
                "3600",
            ),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["allocation"]["policy_type"] == "day"
    assert payload["allocation"]["walltime_seconds"] == 3_600


def test_day_policy_defaults_to_a_short_thirty_minute_allocation(capsys) -> None:
    exit_code = grid5000_cli.main(["plan", *_arguments("--policy-type", "day")])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["allocation"]["walltime_seconds"] == 1_800
    assert payload["allocation"]["walltime"] == "00:30:00"


def test_autonomous_run_without_execute_prints_a_side_effect_free_plan(capsys) -> None:
    exit_code = grid5000_cli.main(
        [
            "run",
            "--source-commit",
            SOURCE_COMMIT,
            "--model-revision",
            MODEL_REVISION,
            "--publish",
            "--sync-trackio",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"]
    assert payload["sites"]
    assert payload["walltime_seconds"] == 1_200
    assert payload["publish"] is True
    assert payload["sync_trackio"] is True


def test_autonomous_plan_persists_a_bounded_continuation_limit(capsys) -> None:
    exit_code = grid5000_cli.main(
        [
            "run",
            "--source-commit",
            SOURCE_COMMIT,
            "--model-revision",
            MODEL_REVISION,
            "--max-continuations",
            "2",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["max_continuations"] == 2


def test_autonomous_plan_records_the_explicit_container_runtime(capsys) -> None:
    image = "registry.example/osm-polygon-sentence-classifier@sha256:" + "d" * 64

    exit_code = grid5000_cli.main(
        [
            "run",
            "--source-commit",
            SOURCE_COMMIT,
            "--model-revision",
            MODEL_REVISION,
            "--container-image",
            image,
            "--container-runtime",
            "docker",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["container_image"] == image
    assert payload["container_runtime"] == "docker"


def test_resume_can_explicitly_extend_a_failed_run_continuation_limit(
    monkeypatch,
    capsys,
) -> None:
    identity = Grid5000RunIdentity(
        source_commit=SOURCE_COMMIT,
        dataset_revision="d" * 40,
        model_name_or_path="jhu-clsp/mmBERT-small",
        model_revision=MODEL_REVISION,
        training_config={
            "model_name_or_path": "jhu-clsp/mmBERT-small",
            "model_revision": MODEL_REVISION,
            "publish_to_hub": False,
            "sync_trackio": False,
        },
    )
    state = AutonomousRunState(
        run_id=identity.run_id,
        phase="failed",
        identity=identity.canonical_payload,
        site="nancy",
        job_id=99,
        facts={"max_continuations": 3, "continuation_count": 3},
    )
    captured: list[int] = []
    checkout_revisions: list[str | None] = []

    class FakeController:
        def __init__(self, config, *, emit) -> None:
            del emit
            captured.append(config.max_continuations)
            checkout_revisions.append(config.worker_source_commit)

        def run(self) -> SimpleNamespace:
            return SimpleNamespace(to_dict=lambda: {"phase": "submitted"})

    monkeypatch.setattr(
        grid5000_cli.AutonomousStateStore,
        "load",
        lambda _self, _run_id: state,
    )
    monkeypatch.setattr(grid5000_cli, "_current_source_commit", lambda: "e" * 40)
    monkeypatch.setattr(grid5000_cli, "AutonomousRunController", FakeController)

    exit_code = grid5000_cli.main(
        [
            "resume",
            "--run-id",
            identity.run_id,
            "--max-continuations",
            "6",
            "--execute",
        ]
    )

    assert exit_code == 0
    assert captured == [6]
    assert checkout_revisions == ["e" * 40]
    assert json.loads(capsys.readouterr().out) == {"phase": "submitted"}


def test_resume_execution_uses_current_worker_commit_without_limit_override(
    monkeypatch,
    capsys,
) -> None:
    identity = Grid5000RunIdentity(
        source_commit=SOURCE_COMMIT,
        dataset_revision="d" * 40,
        model_name_or_path="jhu-clsp/mmBERT-small",
        model_revision=MODEL_REVISION,
        training_config={
            "model_name_or_path": "jhu-clsp/mmBERT-small",
            "model_revision": MODEL_REVISION,
            "publish_to_hub": False,
            "sync_trackio": False,
        },
    )
    state = AutonomousRunState(
        run_id=identity.run_id,
        phase="running",
        identity=identity.canonical_payload,
        site="nancy",
        job_id=99,
        facts={
            "max_continuations": 3,
            "continuation_count": 1,
            "worker_source_commit": "c" * 40,
        },
    )
    captured: list[str | None] = []

    class FakeController:
        def __init__(self, config, *, emit) -> None:
            del emit
            captured.append(config.worker_source_commit)

        def run(self) -> SimpleNamespace:
            return SimpleNamespace(to_dict=lambda: {"phase": "running"})

    monkeypatch.setattr(
        grid5000_cli.AutonomousStateStore,
        "load",
        lambda _self, _run_id: state,
    )
    monkeypatch.setattr(grid5000_cli, "_current_source_commit", lambda: "e" * 40)
    monkeypatch.setattr(grid5000_cli, "AutonomousRunController", FakeController)

    exit_code = grid5000_cli.main(["resume", "--run-id", identity.run_id, "--execute"])

    assert exit_code == 0
    assert captured == ["e" * 40]
    assert json.loads(capsys.readouterr().out) == {"phase": "running"}


def test_resume_uses_the_requested_auto_policy_after_a_day_allocation(
    monkeypatch,
) -> None:
    identity = Grid5000RunIdentity(
        source_commit=SOURCE_COMMIT,
        dataset_revision="d" * 40,
        model_name_or_path="jhu-clsp/mmBERT-small",
        model_revision=MODEL_REVISION,
        training_config={
            "model_name_or_path": "jhu-clsp/mmBERT-small",
            "model_revision": MODEL_REVISION,
            "publish_to_hub": False,
            "sync_trackio": False,
        },
    )
    state = AutonomousRunState(
        run_id=identity.run_id,
        phase="queued",
        identity=identity.canonical_payload,
        site="nancy",
        job_id=99,
        facts={
            "allocation": {
                "policy_type": "day",
                "walltime_seconds": 1_200,
            },
            "requested_policy_type": "auto",
            "max_continuations": 3,
            "continuation_count": 1,
        },
    )

    captured: list[str] = []

    class FakeController:
        def __init__(self, config, *, emit) -> None:
            del emit
            captured.append(config.policy_type)

        def run(self) -> SimpleNamespace:
            return SimpleNamespace(to_dict=lambda: {"phase": "queued"})

    monkeypatch.setattr(
        grid5000_cli.AutonomousStateStore,
        "load",
        lambda _self, _run_id: state,
    )
    monkeypatch.setattr(grid5000_cli, "_current_source_commit", lambda: "e" * 40)
    monkeypatch.setattr(grid5000_cli, "AutonomousRunController", FakeController)

    grid5000_cli.main(["resume", "--run-id", identity.run_id, "--execute"])

    assert captured == ["auto"]


def test_resume_can_override_the_policy_for_a_legacy_state(
    monkeypatch,
) -> None:
    identity = Grid5000RunIdentity(
        source_commit=SOURCE_COMMIT,
        dataset_revision="d" * 40,
        model_name_or_path="jhu-clsp/mmBERT-small",
        model_revision=MODEL_REVISION,
        training_config={
            "model_name_or_path": "jhu-clsp/mmBERT-small",
            "model_revision": MODEL_REVISION,
            "publish_to_hub": False,
            "sync_trackio": False,
        },
    )
    state = AutonomousRunState(
        run_id=identity.run_id,
        phase="queued",
        identity=identity.canonical_payload,
        site="nancy",
        job_id=99,
        facts={
            "allocation": {
                "policy_type": "day",
                "walltime_seconds": 1_200,
            },
            "max_continuations": 3,
            "continuation_count": 1,
        },
    )

    captured: list[str] = []

    class FakeController:
        def __init__(self, config, *, emit) -> None:
            del emit
            captured.append(config.policy_type)

        def run(self) -> SimpleNamespace:
            return SimpleNamespace(to_dict=lambda: {"phase": "queued"})

    monkeypatch.setattr(
        grid5000_cli.AutonomousStateStore,
        "load",
        lambda _self, _run_id: state,
    )
    monkeypatch.setattr(grid5000_cli, "_current_source_commit", lambda: "e" * 40)
    monkeypatch.setattr(grid5000_cli, "AutonomousRunController", FakeController)

    grid5000_cli.main(
        [
            "resume",
            "--run-id",
            identity.run_id,
            "--policy-type",
            "auto",
            "--execute",
        ]
    )

    assert captured == ["auto"]


def test_autonomous_execution_prints_progress_to_stderr_and_json_to_stdout(
    monkeypatch, capsys
) -> None:
    class FakeController:
        def __init__(self, config, *, emit) -> None:
            del config
            emit("submitted")

        def run(self) -> SimpleNamespace:
            return SimpleNamespace(to_dict=lambda: {"phase": "completed"})

    monkeypatch.setattr(grid5000_cli, "AutonomousRunController", FakeController)

    exit_code = grid5000_cli.main(
        [
            "run",
            "--source-commit",
            SOURCE_COMMIT,
            "--model-revision",
            MODEL_REVISION,
            "--execute",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert json.loads(captured.out) == {"phase": "completed"}
    assert captured.err == "[grid5000] submitted\n"


def test_ablations_without_execute_prints_the_side_effect_free_study_plan(
    monkeypatch, capsys, tmp_path
) -> None:
    monkeypatch.setattr(ablation_study, "_default_state_root", lambda: tmp_path)
    exit_code = grid5000_cli.main(
        [
            "ablations",
            "--source-commit",
            SOURCE_COMMIT,
            "--model-revision",
            MODEL_REVISION,
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["study_id"] == "landuse-v1"
    assert payload["total_runs"] == 7
    assert payload["model_repository_id"] == (
        "NoeFlandre/osm-polygon-sentence-classifier"
    )


def test_ablations_execute_is_the_explicit_remote_execution_gate(
    monkeypatch,
    capsys,
) -> None:
    calls: list[object] = []

    class FakeController:
        def __init__(self, **kwargs) -> None:
            calls.append(kwargs)

        def plan(self) -> dict[str, object]:
            raise AssertionError("execute mode must run the study")

        def run(self) -> dict[str, object]:
            return {"phase": "completed"}

    monkeypatch.setattr(grid5000_cli, "AblationStudyController", FakeController)

    exit_code = grid5000_cli.main(
        [
            "ablations",
            "--source-commit",
            SOURCE_COMMIT,
            "--model-revision",
            MODEL_REVISION,
            "--allow-source-commit-update",
            "--execute",
        ]
    )

    assert exit_code == 0
    assert calls
    assert isinstance(calls[0], dict)
    call = cast(dict[str, object], calls[0])
    assert call["allow_source_commit_update"] is True
    assert json.loads(capsys.readouterr().out) == {"phase": "completed"}
