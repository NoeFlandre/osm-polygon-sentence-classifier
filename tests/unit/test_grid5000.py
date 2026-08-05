from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pytest

from osm_polygon_sentence_classifier.config import ProjectConfig
from osm_polygon_sentence_classifier.grid5000 import (
    GRID5000_DATASET_REVISION,
    MINIMUM_HOME_HEADROOM_BYTES,
    CommandResult,
    Grid5000Allocation,
    Grid5000ConfigurationError,
    Grid5000ExecutionError,
    Grid5000Operator,
    Grid5000Plan,
    Grid5000RunIdentity,
    Grid5000State,
    Grid5000StateError,
    Grid5000StateStore,
    parse_quota_output,
)
from osm_polygon_sentence_classifier.grid5000_worker import (
    WorkerError,
    run_landuse_training_worker,
    validate_compute_node,
)
from osm_polygon_sentence_classifier.training import TrainingConfig, TrainingResult

SOURCE_COMMIT = "a" * 40
MODEL_REVISION = "b" * 40


def _identity(
    *, training_config: Mapping[str, object] | None = None
) -> Grid5000RunIdentity:
    return Grid5000RunIdentity(
        source_commit=SOURCE_COMMIT,
        dataset_revision=GRID5000_DATASET_REVISION,
        model_name_or_path="test-model",
        model_revision=MODEL_REVISION,
        training_config=training_config
        or {
            "max_steps": 100,
            "output_subdirectory": "models/landuse",
        },
    )


def _plan() -> Grid5000Plan:
    return Grid5000Plan(
        identity=_identity(),
        allocation=Grid5000Allocation(site="nancy", walltime_seconds=3_600),
    )


@pytest.mark.parametrize(
    "field",
    ["source_commit", "dataset_revision", "model_revision"],
)
def test_run_identity_rejects_unpinned_revisions(field: str) -> None:
    values: dict[str, object] = {
        "source_commit": SOURCE_COMMIT,
        "dataset_revision": GRID5000_DATASET_REVISION,
        "model_name_or_path": "test-model",
        "model_revision": MODEL_REVISION,
        "training_config": {"max_steps": 100},
    }
    values[field] = "not-a-revision"

    with pytest.raises(Grid5000ConfigurationError, match="40 lowercase"):
        cast(Any, Grid5000RunIdentity)(**values)


def test_run_identity_is_canonical_and_changes_with_training_settings() -> None:
    first = _identity(training_config={"max_steps": 100, "seed": 42})
    equivalent = _identity(training_config={"seed": 42, "max_steps": 100})
    changed = _identity(training_config={"max_steps": 101, "seed": 42})

    assert first.canonical_json == equivalent.canonical_json
    assert first.run_id == equivalent.run_id
    assert first.fingerprint == equivalent.fingerprint
    assert first.run_id != changed.run_id
    assert len(first.run_id) == 20


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("site", "Nancy"),
        ("queue", "best-effort"),
        ("policy_type", "day"),
        ("resource_type", "default"),
        ("gpu_count", 0),
        ("gpu_count", 2),
        ("walltime_seconds", 0),
        ("walltime_seconds", 12 * 60 * 60 + 1),
    ],
)
def test_allocation_rejects_unsafe_scheduler_values(field: str, value: object) -> None:
    values: dict[str, object] = {"site": "nancy", "walltime_seconds": 3_600}
    values[field] = value

    with pytest.raises(Grid5000ConfigurationError):
        cast(Any, Grid5000Allocation)(**values)


def test_allocation_renders_one_bounded_gpu_request() -> None:
    allocation = Grid5000Allocation(site="nancy", walltime_seconds=3_661)

    command = allocation.scheduler_command("worker command")

    assert command == (
        "oarsub",
        "-q",
        "default",
        "-t",
        "exotic",
        "-t",
        "night",
        "-l",
        "gpu=1,walltime=01:01:01",
        "worker command",
    )


def test_worker_command_uses_allocation_local_locked_uv_environment() -> None:
    command = _plan().worker_command

    assert (
        'export UV_PROJECT_ENVIRONMENT="/tmp/osm-polygon-sentence-classifier-'
        in command
    )
    assert 'export UV_CACHE_DIR="/tmp/osm-polygon-sentence-classifier-' in command
    assert '"$HOME/.local/bin/uv" run --locked python -m ' in command
    assert '"$HOME/osm-polygon-sentence-classifier"' in command


def test_plan_contains_a_read_only_clean_checkout_guard() -> None:
    plan = _plan()

    command = plan.remote_checkout_command

    assert "git -C" in command[-1]
    assert "rev-parse HEAD" in command[-1]
    assert "status --porcelain" in command[-1]
    assert SOURCE_COMMIT in command[-1]
    assert "git clone" not in command[-1]
    assert "rm " not in command[-1]


def test_quota_parser_uses_soft_headroom() -> None:
    quota = parse_quota_output("1000 25000000 100000000\n")

    assert quota.soft_headroom_bytes == (25_000_000 - 1_000) * 1024
    assert not quota.soft_limit_exceeded


def test_quota_parser_rejects_missing_data() -> None:
    with pytest.raises(Grid5000ConfigurationError, match="quota"):
        parse_quota_output("Filesystem blocks quota limit\n")


class _RecordingRunner:
    def __init__(self, *, state_store: Grid5000StateStore | None = None) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.state_store = state_store

    def __call__(self, argv: Sequence[str], *, timeout: float) -> CommandResult:
        del timeout
        command = tuple(argv)
        self.calls.append(command)
        remote_command = command[-1]
        if "quota" in remote_command:
            return CommandResult(
                returncode=0,
                stdout=f"0 {MINIMUM_HOME_HEADROOM_BYTES // 1024 + 1} "
                f"{MINIMUM_HOME_HEADROOM_BYTES // 1024 + 2}\n",
            )
        if "oarsub" in remote_command:
            assert self.state_store is not None
            state = self.state_store.load(_plan().identity.run_id)
            assert state is not None
            assert state.phase == "submitting"
            return CommandResult(returncode=0, stdout="OAR_JOB_ID=12345\n")
        return CommandResult(returncode=0, stdout="ok\n")


class _LowQuotaRunner(_RecordingRunner):
    def __call__(self, argv: Sequence[str], *, timeout: float) -> CommandResult:
        if "quota" in argv[-1]:
            self.calls.append(tuple(argv))
            return CommandResult(returncode=0, stdout="0 1 2\n")
        return super().__call__(argv, timeout=timeout)


def test_plan_only_submit_makes_no_runner_call_or_state_directory(
    tmp_path: Path,
) -> None:
    state_store = Grid5000StateStore(tmp_path / "runs")
    runner = _RecordingRunner(state_store=state_store)
    operator = Grid5000Operator(_plan(), state_store=state_store, runner=runner)

    result = operator.submit()

    assert result.job_id is None
    assert not runner.calls
    assert not (tmp_path / "runs").exists()


def test_execute_checks_policy_and_quota_before_recording_and_submitting(
    tmp_path: Path,
) -> None:
    state_store = Grid5000StateStore(tmp_path / "runs")
    runner = _RecordingRunner(state_store=state_store)
    operator = Grid5000Operator(_plan(), state_store=state_store, runner=runner)

    result = operator.submit(execute=True)

    assert result.job_id == 12345
    assert ["git -C" in call[-1] for call in runner.calls] == [
        True,
        False,
        False,
        False,
        False,
    ]
    assert "usagepolicycheck -l --sites nancy" in runner.calls[1][-1]
    assert "usagepolicycheck -t" in runner.calls[2][-1]
    assert "quota" in runner.calls[3][-1]
    assert "oarsub" in runner.calls[4][-1]
    saved = state_store.load(_plan().identity.run_id)
    assert saved is not None
    assert saved.phase == "submitted"
    assert saved.job_id == 12345
    assert saved.identity.canonical_json == _plan().identity.canonical_json
    assert saved.submission_command == _plan().submission_command


def test_execute_fails_closed_on_insufficient_soft_quota(tmp_path: Path) -> None:
    state_store = Grid5000StateStore(tmp_path / "runs")
    runner = _LowQuotaRunner(state_store=state_store)
    operator = Grid5000Operator(_plan(), state_store=state_store, runner=runner)

    with pytest.raises(Grid5000ExecutionError, match="soft quota"):
        operator.submit(execute=True)

    assert not any("oarsub" in call[-1] for call in runner.calls)
    assert not (tmp_path / "runs" / _plan().identity.run_id).exists()


@pytest.mark.parametrize("phase", ["submitted", "submitting"])
def test_execute_refuses_existing_or_ambiguous_state(
    tmp_path: Path, phase: str
) -> None:
    state_store = Grid5000StateStore(tmp_path / "runs")
    plan = _plan()
    state_store.save(
        Grid5000State(
            identity=plan.identity,
            phase=cast(Any, phase),
            scheduler_command=plan.scheduler_command,
            job_id=1 if phase == "submitted" else None,
        )
    )
    runner = _RecordingRunner(state_store=state_store)
    operator = Grid5000Operator(plan, state_store=state_store, runner=runner)

    with pytest.raises(Grid5000StateError, match="already|ambiguous"):
        operator.submit(execute=True)

    assert not runner.calls


def test_state_store_uses_restrictive_modes(tmp_path: Path) -> None:
    state_store = Grid5000StateStore(tmp_path / "runs")
    plan = _plan()

    state_store.save(
        Grid5000State(
            identity=plan.identity,
            phase="submitting",
            scheduler_command=plan.scheduler_command,
        )
    )

    run_directory = tmp_path / "runs" / plan.identity.run_id
    assert run_directory.stat().st_mode & 0o777 == 0o700
    assert (run_directory / "state.json").stat().st_mode & 0o777 == 0o600


def test_state_store_rejects_a_dangling_run_directory_symlink(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    root.mkdir()
    plan = _plan()
    (root / plan.identity.run_id).symlink_to(tmp_path / "missing-run")

    with pytest.raises(Grid5000StateError, match="symlink"):
        Grid5000StateStore(root).load(plan.identity.run_id)


def test_state_store_rejects_a_symlinked_root_component(tmp_path: Path) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(Grid5000StateError, match="symlink"):
        Grid5000StateStore(linked_parent / "runs")


def _git_runner(
    expected_commit: str, *, dirty: str = ""
) -> Callable[..., CommandResult]:
    def run(argv: Sequence[str], *, timeout: float) -> CommandResult:
        del timeout
        if argv[-1] == "HEAD":
            return CommandResult(returncode=0, stdout=f"{expected_commit}\n")
        if argv[-1] == "--porcelain":
            return CommandResult(returncode=0, stdout=dirty)
        raise AssertionError(f"unexpected git command: {argv!r}")

    return run


def _valid_worker_kwargs(tmp_path: Path) -> dict[str, Any]:
    return {
        "expected_source_commit": SOURCE_COMMIT,
        "checkout": tmp_path / "checkout",
        "environ": {"OAR_JOB_ID": "12345"},
        "platform_name": "linux",
        "git_runner": _git_runner(SOURCE_COMMIT),
        "cuda_probe": lambda: (True, 1, "Test GPU"),
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {"platform_name": "darwin"},
        {"environ": {}},
        {"environ": {"OAR_JOB_ID": "not-numeric"}},
        {"git_runner": _git_runner("c" * 40)},
        {"git_runner": _git_runner(SOURCE_COMMIT, dirty=" M file.py")},
        {"cuda_probe": lambda: (False, 1, "")},
        {"cuda_probe": lambda: (True, 2, "Test GPU")},
    ],
)
def test_worker_preflight_rejects_unsafe_compute_environment(
    tmp_path: Path, overrides: Mapping[str, object]
) -> None:
    values = _valid_worker_kwargs(tmp_path)
    values.update(overrides)

    with pytest.raises(WorkerError):
        validate_compute_node(**values)  # type: ignore[arg-type]


def test_worker_preflight_returns_validated_facts(tmp_path: Path) -> None:
    facts = validate_compute_node(**_valid_worker_kwargs(tmp_path))

    assert facts.job_id == 12345
    assert facts.source_commit == SOURCE_COMMIT
    assert facts.cuda_device_name == "Test GPU"


def test_worker_runs_training_only_after_preflight(tmp_path: Path) -> None:
    training_config = TrainingConfig(
        model_name_or_path="test-model",
        model_revision=MODEL_REVISION,
    )
    identity = Grid5000RunIdentity(
        source_commit=SOURCE_COMMIT,
        dataset_revision=GRID5000_DATASET_REVISION,
        model_name_or_path=training_config.model_name_or_path,
        model_revision=MODEL_REVISION,
        training_config={
            "model_name_or_path": training_config.model_name_or_path,
            "model_revision": MODEL_REVISION,
        },
    )
    received: dict[str, object] = {}
    expected_result = TrainingResult(
        output_directory=Path.home() / "model",
        train_output=object(),
    )

    def fake_train(
        *, config: TrainingConfig, project_config: ProjectConfig
    ) -> TrainingResult:
        received["config"] = config
        received["project_config"] = project_config
        return expected_result

    result = run_landuse_training_worker(
        identity,
        checkout=tmp_path / "checkout",
        training_config=training_config,
        remote_data_root=Path.home() / "training-data",
        environ={"OAR_JOB_ID": "12345"},
        platform_name="linux",
        git_runner=_git_runner(SOURCE_COMMIT),
        cuda_probe=lambda: (True, 1, "Test GPU"),
        train=fake_train,
    )

    assert result is expected_result
    assert received["config"] is training_config
    assert received["project_config"] == ProjectConfig.for_remote_root(
        Path.home() / "training-data"
    )


def test_worker_requires_hugging_face_auth_before_publishing_or_tracking(
    tmp_path: Path,
) -> None:
    training_config = TrainingConfig(
        model_name_or_path="test-model",
        model_revision=MODEL_REVISION,
        publish_to_hub=True,
        sync_trackio=True,
    )
    identity = Grid5000RunIdentity(
        source_commit=SOURCE_COMMIT,
        dataset_revision=GRID5000_DATASET_REVISION,
        model_name_or_path=training_config.model_name_or_path,
        model_revision=MODEL_REVISION,
        training_config={
            "model_name_or_path": training_config.model_name_or_path,
            "model_revision": MODEL_REVISION,
            "publish_to_hub": True,
            "sync_trackio": True,
        },
    )
    train_called = False

    def fake_train(**kwargs: object) -> TrainingResult:
        del kwargs
        nonlocal train_called
        train_called = True
        return TrainingResult(output_directory=tmp_path, train_output=object())

    with pytest.raises(WorkerError, match="authentication"):
        run_landuse_training_worker(
            identity,
            checkout=tmp_path / "checkout",
            training_config=training_config,
            remote_data_root=Path.home() / "training-data",
            environ={
                "OAR_JOB_ID": "12345",
                "HF_HOME": str(tmp_path / "empty-hf"),
            },
            platform_name="linux",
            git_runner=_git_runner(SOURCE_COMMIT),
            cuda_probe=lambda: (True, 1, "Test GPU"),
            train=fake_train,
        )

    assert not train_called
