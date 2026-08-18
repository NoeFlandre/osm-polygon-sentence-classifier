import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import pytest

import osm_polygon_sentence_classifier.grid5000_worker as grid5000_worker
from osm_polygon_sentence_classifier.checkpoint_hub import PublishedCheckpoint
from osm_polygon_sentence_classifier.checkpointing import (
    CheckpointInfo,
    write_checkpoint_manifest,
)
from osm_polygon_sentence_classifier.config import ProjectConfig
from osm_polygon_sentence_classifier.dataset_contract import (
    WORLDWIDE_PLACE_RELEVANCE_V2_CONTRACT,
)
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
    run_place_relevance_training_worker,
    validate_compute_node,
    write_completion_manifest,
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


def test_run_identity_accepts_the_worldwide_v2_task_name() -> None:
    identity = Grid5000RunIdentity(
        source_commit=SOURCE_COMMIT,
        dataset_revision=(
            WORLDWIDE_PLACE_RELEVANCE_V2_CONTRACT.provenance.repository_revision
        ),
        model_name_or_path="test-model",
        model_revision=MODEL_REVISION,
        task_name="place-relevance-v2",
        training_config={"max_steps": 100},
    )

    assert identity.task_name == "place-relevance-v2"
    assert identity.canonical_payload["task_name"] == "place-relevance-v2"


def test_run_identity_reads_legacy_landuse_state_without_task_name() -> None:
    payload = _identity().canonical_payload
    payload.pop("task_name")

    identity = Grid5000RunIdentity.from_payload(payload)

    assert identity.task_name == "landuse"


def test_run_identity_rejects_an_unknown_task_name() -> None:
    with pytest.raises(Grid5000ConfigurationError, match="task_name"):
        Grid5000RunIdentity(
            source_commit=SOURCE_COMMIT,
            dataset_revision="d" * 40,
            model_name_or_path="test-model",
            model_revision=MODEL_REVISION,
            task_name="unknown-task",
            training_config={"max_steps": 100},
        )


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
        ("policy_type", "holiday"),
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


def test_allocation_renders_a_standard_production_request() -> None:
    allocation = Grid5000Allocation(
        site="grenoble",
        walltime_seconds=1_800,
        queue="production",
        resource_type="standard",
        policy_type="day",
    )

    assert allocation.scheduler_command("worker command") == (
        "oarsub",
        "-q",
        "production",
        "-t",
        "day",
        "-l",
        "gpu=1,walltime=00:30:00",
        "worker command",
    )


def test_allocation_accepts_a_generated_cuda_capability_filter() -> None:
    allocation = Grid5000Allocation(
        site="lille",
        walltime_seconds=1_800,
        resource_type="standard",
        resource_property=(
            "gpu_mem>=8000 AND production='NO' AND cpuarch='x86_64' "
            "AND gpu_compute_capability IN ('8.0')"
        ),
    )

    assert allocation.scheduler_command("worker command")[1:5] == (
        "-q",
        "default",
        "-p",
        "gpu_mem>=8000 AND production='NO' AND cpuarch='x86_64' "
        "AND gpu_compute_capability IN ('8.0')",
    )


def test_day_allocation_accepts_only_a_one_hour_window() -> None:
    allocation = Grid5000Allocation(
        site="grenoble",
        policy_type="day",
        walltime_seconds=3_600,
    )

    assert allocation.scheduler_command("worker command")[6:9] == (
        "day",
        "-l",
        "gpu=1,walltime=01:00:00",
    )

    with pytest.raises(Grid5000ConfigurationError, match="one hour"):
        Grid5000Allocation(
            site="grenoble",
            policy_type="day",
            walltime_seconds=3_601,
        )


def test_worker_command_reuses_a_run_scoped_locked_uv_environment() -> None:
    command = _plan().worker_command

    assert (
        'remote_run_root="$HOME/osm-polygon-sentence-classifier-data/grid5000/runs/'
        in command
    )
    assert 'export UV_PROJECT_ENVIRONMENT="$remote_run_root/.venv"' in command
    assert 'export UV_CACHE_DIR="$remote_run_root/.uv-cache"' in command
    assert "/tmp/osm-polygon-sentence-classifier-" not in command
    assert 'cpu_architecture="$(uname -m)"' in command
    assert '[ "$cpu_architecture" = "x86_64" ]' in command
    assert 'uv_bin="$(command -v uv || true)"' in command
    assert '[ -n "$uv_bin" ] || uv_bin="$HOME/.local/bin/uv"' in command
    assert '"$uv_bin" --version >/dev/null 2>&1' in command
    assert 'exec "$uv_bin" run --locked --no-dev --extra training python -m ' in command
    assert '"$HOME/osm-polygon-sentence-classifier"' in command
    assert '"$HOME/osm-polygon-sentence-classifier-data/grid5000/runs/' in command


def test_worker_command_carries_the_immutable_task_name() -> None:
    identity = Grid5000RunIdentity(
        source_commit=SOURCE_COMMIT,
        dataset_revision="d" * 40,
        model_name_or_path="test-model",
        model_revision=MODEL_REVISION,
        task_name="place-relevance-v2",
        training_config={"max_steps": 100},
    )
    plan = Grid5000Plan(
        identity=identity,
        allocation=Grid5000Allocation(site="nancy", walltime_seconds=1_800),
    )

    assert "--task-name place-relevance-v2" in plan.worker_command


def test_worker_command_leaves_remote_home_path_for_shell_expansion() -> None:
    command = _plan().worker_command

    assert '--remote-data-root "$remote_run_root"' in command
    assert (
        'remote_run_root="$HOME/osm-polygon-sentence-classifier-data/grid5000/runs/'
        f'{_plan().identity.run_id}"'
    ) in command


def test_container_worker_command_uses_explicit_mounts_and_fails_closed() -> None:
    image = "registry.example/osm-polygon-sentence-classifier@sha256:" + "c" * 64
    plan = Grid5000Plan(
        identity=_identity(),
        allocation=Grid5000Allocation(site="nancy", walltime_seconds=3_600),
        container_image=image,
        container_runtime="docker",
    )

    command = plan.worker_command

    assert "container_runtime=docker" in command
    assert '"$container_runtime" image inspect ' + image in command
    assert '"$container_runtime" run --rm' in command
    assert 'gpu_args=(--gpus "device=$cuda_visible_devices")' in command
    assert "nvidia.com/gpu=all" not in command
    assert "--env CUDA_VISIBLE_DEVICES=0" in command
    assert "--env HF_HOME=/home/app/data/cache/huggingface" in command
    assert "dst=/home/app/data/cache/huggingface/token,readonly" in command
    assert "HF_HOME=/run/secrets" not in command
    assert '--user "$(id -u):$(id -g)"' in command
    assert (
        '--mount "type=bind,src=$checkout,dst=/home/app/checkout,readonly"' in command
    )
    assert '--mount "type=bind,src=$data_root,dst=/home/app/data"' in command
    assert (
        f'data_root="$HOME/osm-polygon-sentence-classifier-data/grid5000/runs/'
        f'{plan.identity.run_id}"' in command
    )
    assert "--env PYTHONPATH=/home/app/checkout/src" in command
    assert "--remote-data-root /home/app/data" in command
    assert "exit 78" in command
    assert "--privileged" not in command
    assert "oarsub" not in command
    assert 'exec "$uv_bin"' not in command


@pytest.mark.parametrize(
    "image",
    [
        "registry.example/osm-polygon-sentence-classifier:latest",
        "registry.example/osm-polygon-sentence-classifier",
    ],
)
def test_container_worker_rejects_mutable_image_references(image: str) -> None:
    with pytest.raises(Grid5000ConfigurationError, match="immutable sha256 digest"):
        Grid5000Plan(
            identity=_identity(),
            allocation=Grid5000Allocation(site="nancy", walltime_seconds=3_600),
            container_image=image,
        )


def test_resume_plan_requires_a_valid_checkpoint_on_the_worker() -> None:
    plan = Grid5000Plan(
        identity=_identity(),
        allocation=Grid5000Allocation(site="nancy", walltime_seconds=1_800),
        resume_from_checkpoint=True,
    )

    assert "--require-checkpoint" in plan.worker_command


def test_plan_can_use_a_new_worker_checkout_without_changing_run_identity() -> None:
    checkout_commit = "c" * 40
    plan = Grid5000Plan(
        identity=_identity(),
        allocation=Grid5000Allocation(site="nancy", walltime_seconds=1_800),
        resume_from_checkpoint=True,
        checkout_commit=checkout_commit,
    )

    assert plan.identity.source_commit == SOURCE_COMMIT
    assert f"--source-commit {SOURCE_COMMIT}" in plan.worker_command
    assert f"--checkout-commit {checkout_commit}" in plan.worker_command
    assert checkout_commit in plan.remote_checkout_command[-1]
    assert SOURCE_COMMIT not in plan.remote_checkout_command[-1]


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


class _EightGiBQuotaRunner(_RecordingRunner):
    def __call__(self, argv: Sequence[str], *, timeout: float) -> CommandResult:
        if "quota" in argv[-1]:
            self.calls.append(tuple(argv))
            eight_gib_kib = (8 * 1024**3) // 1024
            return CommandResult(
                returncode=0,
                stdout=f"0 {eight_gib_kib} {eight_gib_kib + 1}\n",
            )
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


def test_execute_allows_the_reduced_persistent_training_footprint(
    tmp_path: Path,
) -> None:
    state_store = Grid5000StateStore(tmp_path / "runs")
    runner = _EightGiBQuotaRunner(state_store=state_store)
    operator = Grid5000Operator(_plan(), state_store=state_store, runner=runner)

    result = operator.submit(execute=True)

    assert result.job_id == 12345


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
        "cuda_probe": lambda: (True, 1, "Test GPU", (8, 0)),
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {"platform_name": "darwin"},
        {"environ": {}},
        {"environ": {"OAR_JOB_ID": "not-numeric"}},
        {"git_runner": _git_runner("c" * 40)},
        {"git_runner": _git_runner(SOURCE_COMMIT, dirty=" M file.py")},
        {"cuda_probe": lambda: (False, 1, "", (0, 0))},
        {"cuda_probe": lambda: (True, 2, "Test GPU", (8, 0))},
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


def test_worker_preflight_rejects_a_gpu_below_the_supported_cuda_capability(
    tmp_path: Path,
) -> None:
    values = _valid_worker_kwargs(tmp_path)
    values["cuda_probe"] = lambda: (True, 1, "Tesla P100", (6, 0))

    with pytest.raises(WorkerError, match="compute capability"):
        validate_compute_node(**values)  # type: ignore[arg-type]


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
        *,
        config: TrainingConfig,
        project_config: ProjectConfig,
        resume_from_checkpoint: Path | None,
        checkpoint_identity: Mapping[str, object],
    ) -> TrainingResult:
        received["config"] = config
        received["project_config"] = project_config
        received["resume_from_checkpoint"] = resume_from_checkpoint
        received["checkpoint_identity"] = checkpoint_identity
        return expected_result

    result = run_landuse_training_worker(
        identity,
        checkout=tmp_path / "checkout",
        training_config=training_config,
        remote_data_root=Path.home() / "training-data",
        environ={"OAR_JOB_ID": "12345"},
        platform_name="linux",
        git_runner=_git_runner(SOURCE_COMMIT),
        cuda_probe=lambda: (True, 1, "Test GPU", (8, 0)),
        train=fake_train,
    )

    assert result is expected_result
    assert isinstance(received["config"], TrainingConfig)
    assert received["config"].run_name == (
        f"{training_config.run_name} | run-{identity.run_id[:8]} "
        "| segment-from-0000 | oar-12345"
    )
    assert received["project_config"] == ProjectConfig.for_remote_root(
        Path.home() / "training-data"
    )
    assert received["resume_from_checkpoint"] is None
    assert received["checkpoint_identity"] == identity.canonical_payload


def test_worldwide_worker_keeps_one_trackio_run_name_across_allocations(
    tmp_path: Path,
) -> None:
    training_config = TrainingConfig(
        model_name_or_path="test-model",
        model_revision=MODEL_REVISION,
        output_subdirectory=Path("studies/place-relevance-v2/baseline/models"),
        validation_fraction=0.1,
        test_fraction=0.1,
        eval_strategy="epoch",
        trainable_layers="head",
        run_name="place-relevance-v2|baseline|seed-42",
        tracking_project="place-relevance-v2",
        artifact_namespace="studies/place-relevance-v2/baseline",
    )
    identity = Grid5000RunIdentity(
        source_commit=SOURCE_COMMIT,
        dataset_revision=(
            WORLDWIDE_PLACE_RELEVANCE_V2_CONTRACT.provenance.repository_revision
        ),
        model_name_or_path=training_config.model_name_or_path,
        model_revision=MODEL_REVISION,
        task_name="place-relevance-v2",
        training_config={
            "model_name_or_path": training_config.model_name_or_path,
            "model_revision": MODEL_REVISION,
            "output_subdirectory": str(training_config.output_subdirectory),
            "validation_fraction": 0.1,
            "test_fraction": 0.1,
            "eval_strategy": "epoch",
            "trainable_layers": "head",
            "run_name": training_config.run_name,
            "tracking_project": training_config.tracking_project,
            "artifact_namespace": training_config.artifact_namespace,
        },
    )
    received: dict[str, object] = {}

    def fake_train(**kwargs: object) -> TrainingResult:
        received.update(kwargs)
        return TrainingResult(
            output_directory=tmp_path / "model", train_output=object()
        )

    run_place_relevance_training_worker(
        identity,
        checkout=tmp_path / "checkout",
        training_config=training_config,
        remote_data_root=Path.home() / "training-data",
        environ={"OAR_JOB_ID": "12345"},
        platform_name="linux",
        git_runner=_git_runner(SOURCE_COMMIT),
        cuda_probe=lambda: (True, 1, "Test GPU", (8, 0)),
        train=fake_train,
    )

    assert isinstance(received["config"], TrainingConfig)
    assert received["config"].run_name == training_config.run_name  # type: ignore[union-attr]


def test_worker_requires_a_complete_checkpoint_for_continuation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    training_config = TrainingConfig(
        model_name_or_path="test-model",
        model_revision=MODEL_REVISION,
    )
    identity = _identity(
        training_config={
            "model_name_or_path": "test-model",
            "model_revision": MODEL_REVISION,
        }
    )
    train_called = False

    def fake_train(**kwargs: object) -> TrainingResult:
        del kwargs
        nonlocal train_called
        train_called = True
        return TrainingResult(output_directory=tmp_path, train_output=object())

    with pytest.raises(WorkerError, match="complete checkpoint"):
        run_landuse_training_worker(
            identity,
            checkout=tmp_path / "checkout",
            training_config=training_config,
            remote_data_root=tmp_path / "training-data",
            environ={"OAR_JOB_ID": "12345"},
            platform_name="linux",
            git_runner=_git_runner(SOURCE_COMMIT),
            cuda_probe=lambda: (True, 1, "Test GPU", (8, 0)),
            train=fake_train,
            require_checkpoint=True,
        )

    assert not train_called


def test_worker_restores_a_published_checkpoint_when_site_storage_is_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    training_config = TrainingConfig(
        model_name_or_path="test-model",
        model_revision=MODEL_REVISION,
        publish_to_hub=True,
    )
    identity = _identity(
        training_config={
            "model_name_or_path": "test-model",
            "model_revision": MODEL_REVISION,
            "publish_to_hub": True,
        }
    )
    remote_root = tmp_path / "training-data"
    checkpoint = remote_root / "models/landuse/checkpoint-12"
    received: dict[str, object] = {}

    def fake_restore(
        output_directory: Path,
        *,
        identity: Mapping[str, object],
        repository_id: str,
    ) -> CheckpointInfo:
        received["restore_output_directory"] = output_directory
        received["restore_identity"] = identity
        received["restore_repository_id"] = repository_id
        checkpoint.mkdir(parents=True)
        for filename in (
            "model.safetensors",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
        ):
            (checkpoint / filename).write_bytes(b"checkpoint")
        (checkpoint / "trainer_state.json").write_text(
            '{"global_step": 12}', encoding="utf-8"
        )
        write_checkpoint_manifest(
            checkpoint,
            identity=identity,
            global_step=12,
        )
        return CheckpointInfo(path=checkpoint, global_step=12)

    monkeypatch.setattr(
        grid5000_worker,
        "restore_published_checkpoint",
        fake_restore,
    )

    def fake_train(**kwargs: object) -> TrainingResult:
        received.update(kwargs)
        return TrainingResult(
            output_directory=tmp_path / "model", train_output=object()
        )

    run_landuse_training_worker(
        identity,
        checkout=tmp_path / "checkout",
        training_config=training_config,
        remote_data_root=remote_root,
        environ={"OAR_JOB_ID": "12345", "HF_TOKEN": "hf_test_token"},
        platform_name="linux",
        git_runner=_git_runner(SOURCE_COMMIT),
        cuda_probe=lambda: (True, 1, "Test GPU", (8, 0)),
        train=fake_train,
        require_checkpoint=True,
    )

    assert received["resume_from_checkpoint"] == checkpoint
    assert received["restore_identity"] == identity.canonical_payload


def test_worker_prefers_a_newer_published_checkpoint_than_stale_local_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    training_config = TrainingConfig(
        model_name_or_path="test-model",
        model_revision=MODEL_REVISION,
        publish_to_hub=True,
    )
    identity = _identity(
        training_config={
            "model_name_or_path": "test-model",
            "model_revision": MODEL_REVISION,
            "publish_to_hub": True,
        }
    )
    remote_root = tmp_path / "training-data"
    stale = remote_root / "models/landuse/checkpoint-12"
    stale.mkdir(parents=True)
    for filename in (
        "model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    ):
        (stale / filename).write_bytes(b"stale")
    (stale / "trainer_state.json").write_text('{"global_step": 12}', encoding="utf-8")
    write_checkpoint_manifest(
        stale, identity=identity.canonical_payload, global_step=12
    )
    newer = remote_root / "models/landuse/checkpoint-20"
    received: dict[str, object] = {}

    monkeypatch.setattr(
        grid5000_worker,
        "latest_published_checkpoint",
        lambda *_args, **_kwargs: PublishedCheckpoint(
            repository_id="NoeFlandre/osm-polygon-sentence-classifier",
            prefix="experiments/landuse/run-test",
            step=20,
            files=(),
        ),
    )

    def fake_restore(
        output_directory: Path,
        *,
        identity: Mapping[str, object],
        repository_id: str,
    ) -> CheckpointInfo:
        received["repository_id"] = repository_id
        newer.mkdir(parents=True)
        for filename in (
            "model.safetensors",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
        ):
            (newer / filename).write_bytes(b"newer")
        (newer / "trainer_state.json").write_text(
            '{"global_step": 20}', encoding="utf-8"
        )
        write_checkpoint_manifest(newer, identity=identity, global_step=20)
        return CheckpointInfo(path=newer, global_step=20)

    monkeypatch.setattr(grid5000_worker, "restore_published_checkpoint", fake_restore)

    def fake_train(**kwargs: object) -> TrainingResult:
        received.update(kwargs)
        return TrainingResult(
            output_directory=tmp_path / "model", train_output=object()
        )

    run_landuse_training_worker(
        identity,
        checkout=tmp_path / "checkout",
        training_config=training_config,
        remote_data_root=remote_root,
        environ={"OAR_JOB_ID": "12345", "HF_TOKEN": "hf_test_token"},
        platform_name="linux",
        git_runner=_git_runner(SOURCE_COMMIT),
        cuda_probe=lambda: (True, 1, "Test GPU", (8, 0)),
        train=fake_train,
        require_checkpoint=True,
    )

    assert received["resume_from_checkpoint"] == newer
    assert received["repository_id"] == "NoeFlandre/osm-polygon-sentence-classifier"


def test_worker_passes_the_latest_checkpoint_to_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    training_config = TrainingConfig(
        model_name_or_path="test-model",
        model_revision=MODEL_REVISION,
    )
    identity = _identity(
        training_config={
            "model_name_or_path": "test-model",
            "model_revision": MODEL_REVISION,
        }
    )
    remote_root = tmp_path / "training-data"
    checkpoint = remote_root / "models/landuse/checkpoint-12"
    checkpoint.mkdir(parents=True)
    for filename in (
        "model.safetensors",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    ):
        (checkpoint / filename).write_bytes(b"checkpoint")
    (checkpoint / "trainer_state.json").write_text(
        '{"global_step": 12}', encoding="utf-8"
    )
    write_checkpoint_manifest(
        checkpoint,
        identity=identity.canonical_payload,
        global_step=12,
    )
    received: dict[str, object] = {}

    def fake_train(**kwargs: object) -> TrainingResult:
        received.update(kwargs)
        return TrainingResult(
            output_directory=tmp_path / "model", train_output=object()
        )

    run_landuse_training_worker(
        identity,
        checkout=tmp_path / "checkout",
        training_config=training_config,
        remote_data_root=remote_root,
        environ={"OAR_JOB_ID": "12345"},
        platform_name="linux",
        git_runner=_git_runner(SOURCE_COMMIT),
        cuda_probe=lambda: (True, 1, "Test GPU", (8, 0)),
        train=fake_train,
        require_checkpoint=True,
    )

    assert received["resume_from_checkpoint"] == checkpoint
    assert received["checkpoint_identity"] == identity.canonical_payload
    assert isinstance(received["config"], TrainingConfig)
    assert received["config"].run_name.endswith("| segment-from-0012 | oar-12345")


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
            cuda_probe=lambda: (True, 1, "Test GPU", (8, 0)),
            train=fake_train,
        )

    assert not train_called


def test_worker_completion_manifest_is_credential_free_and_identity_bound(
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "model"
    output_directory.mkdir()
    identity = _identity(
        training_config={
            "model_name_or_path": "test-model",
            "model_revision": MODEL_REVISION,
        }
    )
    result = TrainingResult(
        output_directory=output_directory,
        train_output=object(),
        metrics={"eval_f1": 0.7, "eval_macro_f1": 0.6},
    )

    manifest = write_completion_manifest(
        identity,
        result,
        remote_data_root=tmp_path,
    )

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["run_id"] == identity.run_id
    assert payload["output_directory"] == "model"
    assert payload["metrics"] == {"eval_f1": 0.7, "eval_macro_f1": 0.6}
    assert "token" not in manifest.read_text(encoding="utf-8").casefold()
    assert manifest.stat().st_mode & 0o777 == 0o600


def test_worker_completion_manifest_rejects_output_outside_remote_root(
    tmp_path: Path,
) -> None:
    identity = _identity()
    result = TrainingResult(
        output_directory=tmp_path.parent / "outside-model",
        train_output=object(),
    )

    with pytest.raises(WorkerError, match="outside"):
        write_completion_manifest(identity, result, remote_data_root=tmp_path)


def test_worker_completion_manifest_rejects_a_symlinked_remote_root(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "managed"
    link.symlink_to(target, target_is_directory=True)
    identity = _identity()
    result = TrainingResult(output_directory=target / "model", train_output=object())

    with pytest.raises(WorkerError, match="symlink"):
        write_completion_manifest(identity, result, remote_data_root=link)
