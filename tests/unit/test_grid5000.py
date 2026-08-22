import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

import osm_polygon_sentence_classifier.grid5000 as grid5000
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
    COMMAND_TIMEOUT_SECONDS,
    GRID5000_DATASET_REVISION,
    MINIMUM_HOME_HEADROOM_BYTES,
    REMOTE_DATA_SUBDIRECTORY,
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
    _canonical_json,
    _canonical_training_config,
    _identity_payload_values,
    _require_non_empty,
    _ssh_argv,
    _validate_allocation_modes,
    _validate_container_image,
    _validate_container_runtime,
    _validate_resource_property,
    _validate_submitting_job,
    _validate_walltime_range,
    _validated_identity_payload_values,
    parse_quota_output,
)
from osm_polygon_sentence_classifier.grid5000_worker import (
    WorkerError,
    _model_publication_payload,
    _validate_checkout,
    _write_completion_manifest,
    run_landuse_training_worker,
    run_place_relevance_training_worker,
    validate_compute_node,
    write_completion_manifest,
)
from osm_polygon_sentence_classifier.publication import ModelPublicationResult
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


def test_ssh_argv_builds_the_complete_hardened_argument_vector() -> None:
    assert _ssh_argv("nancy", "printf ok") == (
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=10",
        "-o",
        "ServerAliveCountMax=2",
        "nancy",
        "printf ok",
    )


def test_ssh_argv_rejects_an_invalid_site_with_the_public_error_message() -> None:
    with pytest.raises(Grid5000ConfigurationError) as error:
        _ssh_argv("Nancy", "printf ok")

    assert str(error.value) == "site must be a lowercase Grid'5000 site name"


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


def test_canonical_json_uses_strict_compact_unicode_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    original_dumps = grid5000.json.dumps

    def dumps(value: object, *args: Any, **kwargs: Any) -> str:
        observed.update(kwargs)
        return original_dumps(value, *args, **kwargs)

    monkeypatch.setattr(grid5000.json, "dumps", dumps)

    assert _canonical_json({"z": "é", "a": [2, 1]}) == ('{"a":[2,1],"z":"é"}')
    assert observed == {
        "allow_nan": False,
        "ensure_ascii": False,
        "separators": (",", ":"),
        "sort_keys": True,
    }


@pytest.mark.parametrize("value", [{"value": float("nan")}, {"value": object()}])
def test_canonical_json_rejects_invalid_values_with_the_exact_error(
    value: object,
) -> None:
    with pytest.raises(Grid5000ConfigurationError) as error:
        _canonical_json(value)

    assert str(error.value) == (
        "training_config must contain only JSON-compatible finite values"
    )


def test_canonical_training_config_rejects_non_mappings_with_the_exact_error() -> None:
    with pytest.raises(Grid5000ConfigurationError) as error:
        _canonical_training_config(cast(Any, []))

    assert str(error.value) == "training_config must be a mapping"


def test_canonical_training_config_rejects_non_object_json_with_the_exact_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(grid5000, "_canonical_json", lambda _value: "not-json")

    with pytest.raises(Grid5000ConfigurationError) as error:
        _canonical_training_config({"max_steps": 100})

    assert str(error.value) == "training_config must be a JSON object"


def test_normalized_training_config_requires_a_json_object_with_string_keys() -> None:
    assert grid5000._validate_normalized_training_config({"max_steps": 100}) is None

    for value in ([], "not-an-object", 1, None):
        with pytest.raises(Grid5000ConfigurationError) as error:
            grid5000._validate_normalized_training_config(value)
        assert str(error.value) == "training_config must be a JSON object"

    with pytest.raises(Grid5000ConfigurationError) as error:
        grid5000._validate_normalized_training_config({1: "invalid"})
    assert str(error.value) == "training_config must be a JSON object"


@pytest.mark.parametrize("value", [42, "", " ", "line\nbreak", "line\rbreak"])
def test_require_non_empty_rejects_invalid_single_line_values(value: object) -> None:
    with pytest.raises(Grid5000ConfigurationError) as error:
        _require_non_empty("field", value)

    assert str(error.value) == "field must be a non-empty single-line string"


def test_require_non_empty_returns_a_valid_value_unchanged() -> None:
    value = "valid-value"

    assert _require_non_empty("field", value) is value


def test_run_identity_payload_preserves_a_custom_task_name() -> None:
    payload = _identity().canonical_payload
    payload["task_name"] = "place-relevance-v2"

    values = _identity_payload_values(payload)

    assert values[4] == "place-relevance-v2"


@pytest.mark.parametrize(
    "missing",
    [
        "source_commit",
        "dataset_revision",
        "model_name_or_path",
        "model_revision",
        "training_config",
    ],
)
def test_run_identity_payload_reports_missing_required_fields_exactly(
    missing: str,
) -> None:
    payload = _identity().canonical_payload
    payload.pop(missing)

    with pytest.raises(Grid5000StateError) as error:
        _identity_payload_values(payload)

    assert str(error.value) == "durable state has an incomplete run identity"


def test_validated_identity_payload_rejects_non_string_identity_values() -> None:
    values: tuple[object, object, object, object, object, object] = (
        SOURCE_COMMIT,
        GRID5000_DATASET_REVISION,
        "test-model",
        MODEL_REVISION,
        42,
        {"max_steps": 100},
    )

    with pytest.raises(Grid5000StateError) as error:
        _validated_identity_payload_values(values)

    assert str(error.value) == "durable state has an invalid run identity"


def test_validated_identity_payload_rejects_a_non_mapping_training_config() -> None:
    values: tuple[object, object, object, object, object, object] = (
        SOURCE_COMMIT,
        GRID5000_DATASET_REVISION,
        "test-model",
        MODEL_REVISION,
        "landuse",
        [],
    )

    with pytest.raises(Grid5000StateError) as error:
        _validated_identity_payload_values(values)

    assert str(error.value) == "durable state has an invalid run identity"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("site", "Nancy"),
        ("queue", "best-effort"),
        ("policy_type", "holiday"),
        ("resource_type", "default"),
        ("gpu_count", 0),
        ("gpu_count", 2),
        ("gpu_count", True),
        ("walltime_seconds", 0),
        ("walltime_seconds", 12 * 60 * 60 + 1),
    ],
)
def test_allocation_rejects_unsafe_scheduler_values(field: str, value: object) -> None:
    values: dict[str, object] = {"site": "nancy", "walltime_seconds": 3_600}
    values[field] = value

    with pytest.raises(Grid5000ConfigurationError):
        cast(Any, Grid5000Allocation)(**values)


@pytest.mark.parametrize(
    ("queue", "resource_type", "message"),
    [
        ("best-effort", "exotic", "queue must be 'default' or 'production'"),
        ("default", "unknown", "resource_type must be 'standard' or 'exotic'"),
    ],
)
def test_allocation_modes_report_exact_errors(
    queue: object, resource_type: object, message: str
) -> None:
    with pytest.raises(Grid5000ConfigurationError) as error:
        _validate_allocation_modes(queue, resource_type)

    assert str(error.value) == message


def test_allocation_validation_preserves_the_resource_property_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resource_property = (
        "gpu_mem>=8000 AND production='NO' AND cpuarch='x86_64' "
        "AND gpu_compute_capability IN ('8.0')"
    )
    observed: list[object] = []
    monkeypatch.setattr(
        grid5000,
        "_validate_resource_property",
        lambda value: observed.append(value),
    )

    Grid5000Allocation(
        site="nancy",
        walltime_seconds=1_800,
        resource_property=resource_property,
    )

    assert observed == [resource_property]


@pytest.mark.parametrize(
    ("validator", "value", "message"),
    [
        (
            grid5000._validate_allocation_site,
            "Nancy",
            "site must be a lowercase Grid'5000 site name",
        ),
        (
            grid5000._validate_allocation_policy,
            "holiday",
            "policy_type must be 'day' or 'night'",
        ),
        (
            grid5000._validate_allocation_gpu_count,
            2,
            "gpu_count must be exactly 1",
        ),
        (
            grid5000._validate_day_walltime,
            (3_601, "day"),
            "day policy walltime_seconds must be at most one hour",
        ),
    ],
)
def test_allocation_validator_messages_are_exact(
    validator: Callable[..., Any], value: object, message: str
) -> None:
    if validator is grid5000._validate_day_walltime:
        with pytest.raises(Grid5000ConfigurationError) as error:
            validator(*cast(tuple[Any, ...], value))
    else:
        with pytest.raises(Grid5000ConfigurationError) as error:
            validator(value)

    assert str(error.value) == message


def test_submitting_state_rejects_a_job_id_with_the_exact_error() -> None:
    with pytest.raises(Grid5000StateError) as error:
        _validate_submitting_job("submitting", 123)

    assert str(error.value) == "submitting state cannot contain a job ID"


def test_state_validators_reject_invalid_values_with_exact_messages() -> None:
    cases: tuple[tuple[Callable[..., Any], tuple[Any, ...], str], ...] = (
        (
            grid5000._validate_grid_state_phase,
            ("failed",),
            "durable state has an unsupported phase",
        ),
        (
            grid5000._validate_grid_state_scheduler_command,
            ((),),
            "durable state has no scheduler command",
        ),
        (
            grid5000._normalized_submission_command,
            (("oarsub",), ("oarsub", 1)),
            "durable state submission command is invalid",
        ),
        (
            grid5000._validate_grid_state_job,
            ("submitting", 123),
            "submitting state cannot contain a job ID",
        ),
        (
            grid5000._validate_state_identity_payload,
            (None,),
            "durable state identity is invalid",
        ),
        (
            grid5000._validate_state_phase_payload,
            ("failed",),
            "durable state phase is invalid",
        ),
        (
            grid5000._validate_state_job_payload,
            (True,),
            "durable state job ID is invalid",
        ),
    )

    for validator, args, message in cases:
        with pytest.raises(Grid5000StateError) as error:
            cast(Any, validator)(*args)
        assert str(error.value) == message


def test_state_payload_fields_reports_missing_fields_exactly() -> None:
    payload = _plan().identity.canonical_payload

    with pytest.raises(Grid5000StateError) as error:
        grid5000._state_payload_fields(payload)

    assert str(error.value) == "durable state is incomplete"


def test_state_command_payload_rejects_non_string_members_exactly() -> None:
    with pytest.raises(Grid5000StateError) as error:
        grid5000._validate_state_command_payload(
            ["oarsub", 1], "scheduler command is invalid"
        )

    assert str(error.value) == "scheduler command is invalid"


def test_state_job_validator_rejects_missing_submitted_job_exactly() -> None:
    with pytest.raises(Grid5000StateError) as error:
        grid5000._validate_grid_state_job("submitted", None)

    assert str(error.value) == "submitted state must contain one positive job ID"


def test_managed_mode_rejects_missing_and_wrong_mode_paths_exactly(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(Grid5000StateError) as error:
        grid5000._check_managed_mode(missing, 0o600, "managed path is unsafe")
    assert str(error.value) == "managed path is unsafe"

    wrong_mode = tmp_path / "wrong-mode"
    wrong_mode.write_text("state", encoding="utf-8")
    wrong_mode.chmod(0o644)
    with pytest.raises(Grid5000StateError) as error:
        grid5000._check_managed_mode(wrong_mode, 0o600, "managed path is unsafe")
    assert str(error.value) == "managed path is unsafe"


def test_reject_symlink_components_requires_an_absolute_path_exactly() -> None:
    with pytest.raises(Grid5000StateError) as error:
        grid5000._reject_symlink_components(Path("relative/runs"))

    assert str(error.value) == "state root must be an absolute path"


@pytest.mark.parametrize("value", ["", " ", "invalid property"])
def test_resource_property_rejects_invalid_values_with_the_exact_error(
    value: object,
) -> None:
    with pytest.raises(Grid5000ConfigurationError) as error:
        _validate_resource_property(value)

    assert str(error.value) == (
        "resource_property must be a generated GPU capability filter"
    )


def test_resource_property_accepts_none() -> None:
    assert _validate_resource_property(None) is None


def test_container_image_reports_the_non_empty_error_exactly() -> None:
    with pytest.raises(Grid5000ConfigurationError) as error:
        _validate_container_image("")

    assert str(error.value) == "container_image must be a non-empty single-line string"


def test_container_image_reports_the_digest_error_exactly() -> None:
    with pytest.raises(Grid5000ConfigurationError) as error:
        _validate_container_image("registry.example/project:latest")

    assert str(error.value) == "container_image must include an immutable sha256 digest"


@pytest.mark.parametrize("runtime", [1, "unknown"])
def test_container_runtime_reports_invalid_values_exactly(runtime: object) -> None:
    with pytest.raises(Grid5000ConfigurationError) as error:
        _validate_container_runtime(runtime)

    assert str(error.value) == "container_runtime must be auto, docker, or podman"


@pytest.mark.parametrize("runtime", ["auto", "docker", "podman"])
def test_container_runtime_accepts_supported_values(runtime: str) -> None:
    assert _validate_container_runtime(runtime) is None


@pytest.mark.parametrize("value", [True, 0, -1, 12 * 60 * 60 + 1, "3600"])
def test_walltime_range_rejects_invalid_values_with_the_exact_error(
    value: object,
) -> None:
    with pytest.raises(Grid5000ConfigurationError) as error:
        _validate_walltime_range(value)

    assert str(error.value) == (
        "walltime_seconds must be between 1 second and 12 hours"
    )


def test_walltime_range_accepts_the_two_bounds() -> None:
    assert _validate_walltime_range(1) is None
    assert _validate_walltime_range(12 * 60 * 60) is None


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


def test_worker_command_builds_the_runtime_on_node_local_scratch() -> None:
    command = _plan().worker_command

    assert (
        'remote_run_root="$HOME/osm-polygon-sentence-classifier-data/grid5000/runs/'
        in command
    )
    assert (
        'runtime_root="${TMPDIR:-/tmp}/osm-polygon-sentence-classifier-${USER:-unknown}-symlink"'
        in command
    )
    assert 'mkdir -p "$runtime_root"' in command
    assert 'export UV_PROJECT_ENVIRONMENT="$runtime_root/environment"' in command
    assert "grid5000/environment" not in command
    assert (
        'if ! "$UV_PROJECT_ENVIRONMENT/bin/python" -c "import torch" '
        ">/dev/null 2>&1; then " in command
    )
    assert (
        'export UV_CACHE_DIR="$HOME/.cache/osm-polygon-sentence-classifier/uv"'
        in command
    )
    assert "export UV_LINK_MODE=symlink" in command
    assert (
        'if [ -d "$HOME/.cache/osm-polygon-sentence-classifier/wheels" ]; then '
        'torch_wheel="$(find "$HOME/.cache/osm-polygon-sentence-classifier/wheels" '
        '-maxdepth 1 -type f -name "torch-*.whl" -print -quit)"; '
        'if [ -n "$torch_wheel" ]; then '
        'cp "$torch_wheel" "$runtime_root/"; '
        '"$uv_bin" venv "$UV_PROJECT_ENVIRONMENT" --allow-existing '
        "--no-python-downloads >/dev/null; "
        'UV_LINK_MODE=copy UV_NO_CACHE=1 "$uv_bin" pip install --python '
        '"$UV_PROJECT_ENVIRONMENT/bin/python" '
        "--no-index --no-deps --find-links "
        '"$runtime_root" torch; fi; '
        "fi" in command
    )
    assert '"$TMPDIR"' not in command
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

    assert quota.used_bytes == 1_000 * 1024
    assert quota.soft_limit_bytes == 25_000_000 * 1024
    assert quota.hard_limit_bytes == 100_000_000 * 1024
    assert quota.soft_headroom_bytes == (25_000_000 - 1_000) * 1024
    assert not quota.soft_limit_exceeded


def test_quota_parser_accepts_equal_soft_and_hard_limits() -> None:
    quota = parse_quota_output("1000 25000000 25000000\n")

    assert quota.soft_limit_bytes == quota.hard_limit_bytes


@pytest.mark.parametrize(
    "output",
    ["1000 0 1\n", "1000 10 9\n"],
)
def test_quota_parser_rejects_invalid_limits_with_the_exact_error(
    output: str,
) -> None:
    with pytest.raises(Grid5000ConfigurationError) as error:
        parse_quota_output(output)

    assert str(error.value) == "home quota limits are invalid"


def test_quota_parser_rejects_missing_data() -> None:
    with pytest.raises(Grid5000ConfigurationError) as error:
        parse_quota_output("Filesystem blocks quota limit\n")

    assert str(error.value) == "home quota output has no usable data row"


def test_quota_parser_rejects_non_text_output_with_the_exact_error() -> None:
    with pytest.raises(Grid5000ConfigurationError) as error:
        parse_quota_output(cast(Any, None))

    assert str(error.value) == "quota output must be text"


def test_quota_parser_skips_headers_before_the_first_usable_row() -> None:
    quota = parse_quota_output(
        "Filesystem blocks quota limit\n1000 25000000 100000000\n"
    )

    assert quota.used_bytes == 1_000 * 1024


@pytest.mark.parametrize("output", ["", "OAR_JOB_ID=1\nOAR_JOB_ID=2\n"])
def test_parse_job_id_rejects_non_unique_output_with_an_exact_error(
    output: str,
) -> None:
    with pytest.raises(Grid5000ConfigurationError) as error:
        grid5000.parse_job_id(output)

    assert str(error.value) == "submission did not return one job ID"


def test_parse_job_id_accepts_the_pinned_oar_output_shape() -> None:
    assert grid5000.parse_job_id("submitted\nOAR_JOB_ID=12345\n") == 12_345


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
            assert state.submission_command == _plan().submission_command
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


def test_operator_rejects_zero_timeout_and_preserves_the_smallest_valid_timeout(
    tmp_path: Path,
) -> None:
    with pytest.raises(Grid5000ConfigurationError) as error:
        Grid5000Operator(
            _plan(),
            state_store=Grid5000StateStore(tmp_path / "runs"),
            command_timeout=0,
        )
    assert str(error.value) == "command_timeout must be positive"

    operator = Grid5000Operator(
        _plan(),
        state_store=Grid5000StateStore(tmp_path / "valid-runs"),
        command_timeout=1,
    )
    assert operator.command_timeout == 1


def test_operator_plan_only_submission_returns_the_same_plan_and_false_flag(
    tmp_path: Path,
) -> None:
    plan = _plan()
    result = Grid5000Operator(
        plan,
        state_store=Grid5000StateStore(tmp_path / "runs"),
    ).submit()

    assert result.plan is plan
    assert result.executed is False
    assert result.job_id is None


@pytest.mark.parametrize("job_id", [None, True, 0, -1, "1"])
def test_submitted_state_requires_one_positive_integer_job_id(job_id: object) -> None:
    with pytest.raises(Grid5000StateError) as error:
        Grid5000State(
            identity=_plan().identity,
            phase="submitted",
            scheduler_command=_plan().scheduler_command,
            job_id=cast(Any, job_id),
        )

    assert str(error.value) == "submitted state must contain one positive job ID"


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
    assert result.plan is operator.plan
    assert result.executed is True


def test_run_checked_forwards_the_configured_timeout(
    tmp_path: Path,
) -> None:
    observed: list[float] = []

    def runner(_argv: Sequence[str], *, timeout: float) -> CommandResult:
        observed.append(timeout)
        return CommandResult(returncode=0, stdout="ok")

    operator = Grid5000Operator(
        _plan(),
        state_store=Grid5000StateStore(tmp_path / "runs"),
        runner=cast(Any, runner),
        command_timeout=1.25,
    )

    assert operator._run_checked(("command",), "probe").stdout == "ok"
    assert observed == [1.25]


def test_submit_job_preserves_labels_and_wraps_invalid_job_ids(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    operator = Grid5000Operator(
        _plan(),
        state_store=Grid5000StateStore(tmp_path / "runs"),
    )
    observed: list[tuple[tuple[str, ...], str]] = []

    def invalid_check(argv: Sequence[str], label: str) -> CommandResult:
        observed.append((tuple(argv), label))
        return CommandResult(returncode=0, stdout="not-a-job")

    monkeypatch.setattr(operator, "_run_checked", invalid_check)

    with pytest.raises(Grid5000ExecutionError) as error:
        operator._submit_job()
    assert str(error.value) == "OAR submission returned an invalid job ID"
    assert observed == [(operator.plan.submission_command, "OAR submission")]


def test_submission_preflight_preserves_each_checked_command_label(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _plan()
    operator = Grid5000Operator(
        plan,
        state_store=Grid5000StateStore(tmp_path / "runs"),
        runner=_RecordingRunner(),
    )
    calls: list[tuple[tuple[str, ...], str]] = []

    def record_check(command: Sequence[str], label: str) -> CommandResult:
        calls.append((tuple(command), label))
        if tuple(command) == plan.quota_command:
            return CommandResult(
                returncode=0,
                stdout=f"0 {MINIMUM_HOME_HEADROOM_BYTES // 1024 + 1} "
                f"{MINIMUM_HOME_HEADROOM_BYTES // 1024 + 2}\n",
            )
        return CommandResult(returncode=0, stdout="ok\n")

    monkeypatch.setattr(operator, "_run_checked", record_check)

    operator._run_submission_preflight()

    assert calls == [
        (plan.remote_checkout_command, "remote checkout"),
        (plan.policy_site_command, "site policy"),
        (plan.policy_total_command, "total policy"),
        (plan.quota_command, "home quota"),
    ]


def test_submission_preflight_reports_the_exact_soft_quota_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _plan()
    operator = Grid5000Operator(
        plan,
        state_store=Grid5000StateStore(tmp_path / "runs"),
        runner=_RecordingRunner(),
    )

    def low_quota_check(command: Sequence[str], _label: str) -> CommandResult:
        return CommandResult(
            returncode=0,
            stdout="0 1 2\n" if tuple(command) == plan.quota_command else "ok\n",
        )

    monkeypatch.setattr(operator, "_run_checked", low_quota_check)

    with pytest.raises(Grid5000ExecutionError) as error:
        operator._run_submission_preflight()

    assert str(error.value) == (
        "Grid'5000 home soft quota has insufficient safe headroom"
    )


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


def test_state_store_save_rejects_a_regular_submission_lock_exactly(
    tmp_path: Path,
) -> None:
    store = Grid5000StateStore(tmp_path / "runs")
    plan = _plan()
    state = Grid5000State(
        identity=plan.identity,
        phase="submitting",
        scheduler_command=plan.scheduler_command,
    )
    run_directory = store._ensure_run_directory(plan.identity.run_id)
    lock_path = run_directory / ".intent.lock"
    lock_path.write_text("claimed", encoding="utf-8")

    with pytest.raises(Grid5000StateError) as error:
        store.save(state)
    assert str(error.value) == "submission state is ambiguous"


def test_state_store_save_checks_the_exact_lock_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = Grid5000StateStore(tmp_path / "runs")
    plan = _plan()
    state = Grid5000State(
        identity=plan.identity,
        phase="submitting",
        scheduler_command=plan.scheduler_command,
    )
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    (run_directory / ".intent.lock").write_text("claimed", encoding="utf-8")
    monkeypatch.setattr(store, "_ensure_run_directory", lambda _run_id: run_directory)
    observed: list[Path] = []
    original_exists = Path.exists

    def record_exists(path: Path) -> bool:
        if path.parent == run_directory:
            observed.append(path)
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", record_exists)

    with pytest.raises(Grid5000StateError):
        store.save(state)

    assert observed == [run_directory / ".intent.lock"]


def test_state_store_create_submitting_forwards_the_exact_state_and_lock_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = Grid5000StateStore(tmp_path / "runs")
    plan = _plan()
    state = Grid5000State(
        identity=plan.identity,
        phase="submitting",
        scheduler_command=plan.scheduler_command,
    )
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    monkeypatch.setattr(store, "_ensure_run_directory", lambda _run_id: run_directory)
    observed: list[tuple[Path, Path]] = []
    monkeypatch.setattr(
        grid5000,
        "_require_unclaimed_submission",
        lambda state_path, lock_path: observed.append((state_path, lock_path)),
    )
    monkeypatch.setattr(store, "_write_submission_intent", lambda *args: None)

    store.create_submitting(state)

    assert observed == [(run_directory / "state.json", run_directory / ".intent.lock")]


def test_no_existing_submission_guard_distinguishes_both_durable_phases() -> None:
    plan = _plan()
    submitting = Grid5000State(
        identity=plan.identity,
        phase="submitting",
        scheduler_command=plan.scheduler_command,
    )
    submitted = Grid5000State(
        identity=plan.identity,
        phase="submitted",
        scheduler_command=plan.scheduler_command,
        job_id=123,
    )

    with pytest.raises(Grid5000StateError) as error:
        grid5000._require_no_existing_submission(submitted)
    assert str(error.value) == "run already has a recorded submission"

    with pytest.raises(Grid5000StateError) as error:
        grid5000._require_no_existing_submission(submitting)
    assert str(error.value) == "run has an ambiguous submitting state"

    assert grid5000._require_no_existing_submission(None) is None


def test_execute_refuses_a_leftover_submission_lock(tmp_path: Path) -> None:
    state_store = Grid5000StateStore(tmp_path / "runs")
    run_directory = tmp_path / "runs" / _plan().identity.run_id
    run_directory.mkdir(parents=True)
    run_directory.chmod(0o700)
    (run_directory / ".intent.lock").write_text("claimed", encoding="utf-8")
    (run_directory / ".intent.lock").chmod(0o600)

    with pytest.raises(Grid5000StateError, match="ambiguous"):
        state_store.create_submitting(
            Grid5000State(
                identity=_plan().identity,
                phase="submitting",
                scheduler_command=_plan().scheduler_command,
            )
        )


def test_run_checked_wraps_runner_and_nonzero_failures(tmp_path: Path) -> None:
    class FailingRunner:
        def __init__(self, error: Exception | None = None) -> None:
            self.error = error

        def __call__(self, argv: Sequence[str], *, timeout: float) -> CommandResult:
            del argv, timeout
            if self.error is not None:
                raise self.error
            return CommandResult(returncode=7, stderr="denied")

    operator = Grid5000Operator(
        _plan(),
        state_store=Grid5000StateStore(tmp_path / "runs"),
        runner=FailingRunner(RuntimeError("network down")),
    )
    with pytest.raises(Grid5000ExecutionError, match="could not complete"):
        operator._run_checked(("command",), "probe")

    operator.runner = FailingRunner()
    with pytest.raises(Grid5000ExecutionError, match="failed with exit code 7"):
        operator._run_checked(("command",), "probe")


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


def test_submission_intent_uses_a_secure_utf8_lock_and_cleans_it_up(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state_store = Grid5000StateStore(tmp_path / "runs")
    plan = _plan()
    state = Grid5000State(
        identity=plan.identity,
        phase="submitting",
        scheduler_command=plan.scheduler_command,
    )
    run_directory = state_store._ensure_run_directory(plan.identity.run_id)
    lock_path = run_directory / ".intent.lock"
    open_calls: list[tuple[Path, int, int]] = []
    fdopen_calls: list[tuple[str, dict[str, object]]] = []
    atomic_calls: list[tuple[Path, Grid5000State]] = []
    original_open = grid5000.os.open
    original_fdopen = grid5000.os.fdopen

    def recording_open(path: Any, flags: int, mode: int = 0o777) -> int:
        open_calls.append((Path(path), flags, mode))
        return original_open(path, flags, mode)

    def recording_fdopen(descriptor: int, mode: str, **kwargs: object) -> Any:
        fdopen_calls.append((mode, kwargs))
        return cast(Any, original_fdopen)(descriptor, mode, **kwargs)

    monkeypatch.setattr(grid5000.os, "open", recording_open)
    monkeypatch.setattr(grid5000.os, "fdopen", recording_fdopen)
    monkeypatch.setattr(
        state_store,
        "_write_atomic",
        lambda directory, current_state: atomic_calls.append(
            (directory, current_state)
        ),
    )

    state_store._write_submission_intent(run_directory, lock_path, state)

    assert open_calls == [
        (
            lock_path,
            grid5000.os.O_CREAT | grid5000.os.O_EXCL | grid5000.os.O_WRONLY,
            0o600,
        )
    ]
    assert fdopen_calls == [("w", {"encoding": "utf-8"})]
    assert atomic_calls == [(run_directory, state)]
    assert not lock_path.exists()


def test_submission_intent_reports_a_lock_collision_exactly(
    tmp_path: Path,
) -> None:
    state_store = Grid5000StateStore(tmp_path / "runs")
    plan = _plan()
    state = Grid5000State(
        identity=plan.identity,
        phase="submitting",
        scheduler_command=plan.scheduler_command,
    )
    run_directory = state_store._ensure_run_directory(plan.identity.run_id)
    lock_path = run_directory / ".intent.lock"
    lock_path.write_text("claimed", encoding="utf-8")

    with pytest.raises(Grid5000StateError) as error:
        state_store._write_submission_intent(run_directory, lock_path, state)

    assert str(error.value) == "submission state is ambiguous"


def test_submission_intent_wraps_os_errors_with_the_public_message(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state_store = Grid5000StateStore(tmp_path / "runs")
    plan = _plan()
    state = Grid5000State(
        identity=plan.identity,
        phase="submitting",
        scheduler_command=plan.scheduler_command,
    )
    run_directory = state_store._ensure_run_directory(plan.identity.run_id)
    lock_path = run_directory / ".intent.lock"

    def fail_open(*args: object, **kwargs: object) -> int:
        del args, kwargs
        raise OSError("disk full")

    monkeypatch.setattr(grid5000.os, "open", fail_open)

    with pytest.raises(Grid5000StateError) as error:
        state_store._write_submission_intent(run_directory, lock_path, state)

    assert str(error.value) == "submission intent cannot be recorded securely"


def test_state_store_ensure_run_directory_preserves_creation_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "nested" / "runs"
    run_id = _plan().identity.run_id
    mkdir_calls: list[tuple[Path, tuple[object, ...], dict[str, object]]] = []
    chmod_calls: list[tuple[Path, int]] = []
    original_mkdir = Path.mkdir
    original_chmod = grid5000.os.chmod

    def recording_mkdir(path: Path, *args: object, **kwargs: object) -> None:
        mkdir_calls.append((path, args, kwargs))
        cast(Any, original_mkdir)(path, *args, **kwargs)

    def recording_chmod(path: Path, mode: int) -> None:
        chmod_calls.append((Path(path), mode))
        original_chmod(path, mode)

    monkeypatch.setattr(Path, "mkdir", recording_mkdir)
    monkeypatch.setattr(grid5000.os, "chmod", recording_chmod)

    run_directory = Grid5000StateStore(root)._ensure_run_directory(run_id)

    assert run_directory == root / run_id
    assert (root, (), {"parents": True, "exist_ok": True, "mode": 0o700}) in mkdir_calls
    assert (
        run_directory,
        (),
        {"mode": 0o700, "exist_ok": True},
    ) in mkdir_calls
    assert chmod_calls == [(root, 0o700), (run_directory, 0o700)]
    assert root.stat().st_mode & 0o777 == 0o700
    assert run_directory.stat().st_mode & 0o777 == 0o700


def test_state_store_ensure_run_directory_rejects_a_symlinked_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir()
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)
    store = Grid5000StateStore(tmp_path / "runs")
    store.root = linked_root
    monkeypatch.setattr(grid5000, "_reject_symlink_components", lambda path: None)

    with pytest.raises(Grid5000StateError) as error:
        store._ensure_run_directory(_plan().identity.run_id)

    assert str(error.value) == "state root cannot be a symlink"


def test_state_store_ensure_run_directory_rejects_a_symlinked_run_directory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runs"
    root.mkdir()
    run_id = _plan().identity.run_id
    target = tmp_path / "target"
    target.mkdir()
    (root / run_id).symlink_to(target, target_is_directory=True)

    with pytest.raises(Grid5000StateError) as error:
        Grid5000StateStore(root)._ensure_run_directory(run_id)

    assert str(error.value) == "run state directory cannot be a symlink"


def test_state_store_ensure_run_directory_reports_root_hardening_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail_chmod(path: Path, mode: int) -> None:
        del path, mode
        raise OSError("permission denied")

    monkeypatch.setattr(grid5000.os, "chmod", fail_chmod)

    with pytest.raises(Grid5000StateError) as error:
        Grid5000StateStore(tmp_path / "runs")._ensure_run_directory(
            _plan().identity.run_id
        )

    assert str(error.value) == "state root cannot be created securely"


def test_state_store_ensure_run_directory_reports_run_hardening_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "runs"
    run_id = _plan().identity.run_id
    original_chmod = grid5000.os.chmod
    run_directory = root / run_id

    def fail_run_chmod(path: Path, mode: int) -> None:
        if Path(path) == run_directory:
            raise OSError("permission denied")
        original_chmod(path, mode)

    monkeypatch.setattr(grid5000.os, "chmod", fail_run_chmod)

    with pytest.raises(Grid5000StateError) as error:
        Grid5000StateStore(root)._ensure_run_directory(run_id)

    assert str(error.value) == "run state directory cannot be created securely"


def test_state_store_write_atomic_preserves_tempfile_and_encoding_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = Grid5000StateStore(tmp_path / "runs")
    plan = _plan()
    run_directory = store._ensure_run_directory(plan.identity.run_id)
    state = Grid5000State(
        identity=plan.identity,
        phase="submitting",
        scheduler_command=plan.scheduler_command,
    )
    mkstemp_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    fdopen_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    replace_calls: list[tuple[Path, Path]] = []
    original_mkstemp = grid5000.tempfile.mkstemp
    original_fdopen = grid5000.os.fdopen
    original_replace = grid5000.os.replace

    def recording_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        mkstemp_calls.append((args, kwargs))
        return cast(Any, original_mkstemp)(*args, **kwargs)

    def recording_fdopen(descriptor: int, *args: object, **kwargs: object) -> object:
        fdopen_calls.append((args, kwargs))
        return cast(Any, original_fdopen)(descriptor, *args, **kwargs)

    def recording_replace(source: Path, destination: Path) -> None:
        replace_calls.append((Path(source), Path(destination)))
        original_replace(source, destination)

    monkeypatch.setattr(grid5000.tempfile, "mkstemp", recording_mkstemp)
    monkeypatch.setattr(grid5000.os, "fdopen", recording_fdopen)
    monkeypatch.setattr(grid5000.os, "replace", recording_replace)

    store._write_atomic(run_directory, state)

    assert mkstemp_calls == [
        ((), {"dir": run_directory, "prefix": ".state-", "suffix": ".tmp"})
    ]
    assert fdopen_calls == [(("w",), {"encoding": "utf-8"})]
    assert len(replace_calls) == 1
    temporary_path, state_path = replace_calls[0]
    assert temporary_path.parent == run_directory
    assert temporary_path.name.startswith(".state-")
    assert temporary_path.name.endswith(".tmp")
    assert state_path == run_directory / "state.json"
    assert state_path.exists()
    assert not list(run_directory.glob(".state-*.tmp"))


def test_state_store_write_atomic_wraps_initial_tempfile_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = Grid5000StateStore(tmp_path / "runs")
    plan = _plan()
    run_directory = store._ensure_run_directory(plan.identity.run_id)
    state = Grid5000State(
        identity=plan.identity,
        phase="submitting",
        scheduler_command=plan.scheduler_command,
    )

    def fail_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        del args, kwargs
        raise OSError("disk full")

    monkeypatch.setattr(grid5000.tempfile, "mkstemp", fail_mkstemp)

    with pytest.raises(Grid5000StateError) as error:
        store._write_atomic(run_directory, state)

    assert str(error.value) == "run state document cannot be written securely"


def test_state_store_write_atomic_cleans_tempfile_after_replace_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = Grid5000StateStore(tmp_path / "runs")
    plan = _plan()
    run_directory = store._ensure_run_directory(plan.identity.run_id)
    state = Grid5000State(
        identity=plan.identity,
        phase="submitting",
        scheduler_command=plan.scheduler_command,
    )

    def fail_replace(source: Path, destination: Path) -> None:
        del source, destination
        raise OSError("disk full")

    monkeypatch.setattr(grid5000.os, "replace", fail_replace)

    with pytest.raises(Grid5000StateError) as error:
        store._write_atomic(run_directory, state)

    assert str(error.value) == "run state document cannot be written securely"
    assert not list(run_directory.glob(".state-*.tmp"))


def test_state_store_write_atomic_does_not_mask_replace_failure_with_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    store = Grid5000StateStore(tmp_path / "runs")
    plan = _plan()
    run_directory = store._ensure_run_directory(plan.identity.run_id)
    state = Grid5000State(
        identity=plan.identity,
        phase="submitting",
        scheduler_command=plan.scheduler_command,
    )
    original_unlink = Path.unlink

    def fail_unlink(path: Path, *args: object, **kwargs: object) -> None:
        if path.parent == run_directory and path.name.startswith(".state-"):
            raise OSError("cleanup denied")
        cast(Any, original_unlink)(path, *args, **kwargs)

    def fail_replace(source: Path, destination: Path) -> None:
        del source, destination
        raise OSError("disk full")

    monkeypatch.setattr(Path, "unlink", fail_unlink)
    monkeypatch.setattr(grid5000.os, "replace", fail_replace)

    with pytest.raises(Grid5000StateError) as error:
        store._write_atomic(run_directory, state)

    assert str(error.value) == "run state document cannot be written securely"


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

    with pytest.raises(Grid5000StateError) as error:
        Grid5000StateStore(linked_parent / "runs")
    assert str(error.value) == "state root cannot contain symlink components"


def test_state_store_default_root_is_under_the_configured_data_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class ConfiguredProject:
        data_root = tmp_path

    monkeypatch.setattr(grid5000, "ProjectConfig", ConfiguredProject)

    store = Grid5000StateStore()

    assert store.root == tmp_path / "grid5000/runs"


def test_state_store_run_id_validation_has_an_exact_boundary_contract(
    tmp_path: Path,
) -> None:
    store = Grid5000StateStore(tmp_path / "runs")
    store._validate_run_id("a" * 20)

    for value in ("A" * 20, "a" * 19, "a" * 21):
        with pytest.raises(Grid5000StateError) as error:
            store._validate_run_id(value)
        assert str(error.value) == "run ID is invalid"


def test_state_store_read_state_payload_uses_utf8_and_rejects_bad_documents(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state_path = tmp_path / "state.json"
    state_path.write_text('{"é": "ok"}', encoding="utf-8")
    state_path.chmod(0o600)
    original_read_text = Path.read_text
    encodings: list[object] = []

    def read_text(path: Path, *args: object, **kwargs: object) -> str:
        encodings.append(kwargs.get("encoding", args[0] if args else None))
        return cast(Any, original_read_text)(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)

    assert Grid5000StateStore._read_state_payload(state_path) == {"é": "ok"}
    assert encodings == ["utf-8"]

    state_path.write_text("not-json", encoding="utf-8")
    with pytest.raises(Grid5000StateError) as error:
        Grid5000StateStore._read_state_payload(state_path)
    assert str(error.value) == "run state document cannot be read"

    state_path.write_text("[]", encoding="utf-8")
    with pytest.raises(Grid5000StateError) as error:
        Grid5000StateStore._read_state_payload(state_path)
    assert str(error.value) == "run state document must be an object"


def test_state_store_requires_a_readable_state_document_exactly(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.json"
    with pytest.raises(Grid5000StateError) as error:
        Grid5000StateStore._require_readable_state_document(missing)
    assert str(error.value) == "run state document is missing or unsafe"

    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    target.chmod(0o600)
    linked = tmp_path / "linked.json"
    linked.symlink_to(target)
    with pytest.raises(Grid5000StateError) as error:
        Grid5000StateStore._require_readable_state_document(linked)
    assert str(error.value) == "run state document is missing or unsafe"

    target.chmod(0o644)
    with pytest.raises(Grid5000StateError) as error:
        Grid5000StateStore._require_readable_state_document(target)
    assert str(error.value) == "run state document permissions are unsafe"


def test_state_store_read_state_document_checks_the_fixed_lock_and_state_names(
    tmp_path: Path,
) -> None:
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    lock_path = run_directory / ".intent.lock"
    lock_path.write_text("claimed", encoding="utf-8")
    with pytest.raises(Grid5000StateError) as error:
        Grid5000StateStore._read_state_document(run_directory)
    assert str(error.value) == "submission state is ambiguous"

    lock_path.unlink()
    state_path = run_directory / "state.json"
    state_path.write_text("{}", encoding="utf-8")
    state_path.chmod(0o600)
    assert Grid5000StateStore._read_state_document(run_directory) == {}


def test_state_store_read_state_document_forwards_the_exact_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    run_directory = tmp_path / "run"
    observed: list[tuple[str, Path]] = []

    monkeypatch.setattr(
        Grid5000StateStore,
        "_require_no_submission_intent",
        staticmethod(lambda path: observed.append(("lock", path))),
    )
    monkeypatch.setattr(
        Grid5000StateStore,
        "_require_readable_state_document",
        staticmethod(lambda path: observed.append(("state", path))),
    )
    monkeypatch.setattr(
        Grid5000StateStore,
        "_read_state_payload",
        staticmethod(lambda path: {"path": str(path)}),
    )

    assert Grid5000StateStore._read_state_document(run_directory) == {
        "path": str(run_directory / "state.json")
    }
    assert observed == [
        ("lock", run_directory / ".intent.lock"),
        ("state", run_directory / "state.json"),
    ]


def test_state_store_load_rejects_unsafe_run_directory_states_exactly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "runs"
    root.mkdir()
    run_id = _plan().identity.run_id
    run_directory = root / run_id
    run_directory.mkdir()
    run_directory.chmod(0o755)
    with pytest.raises(Grid5000StateError) as error:
        Grid5000StateStore(root).load(run_id)
    assert str(error.value) == "run state directory permissions are unsafe"

    real_root = tmp_path / "real-root"
    real_root.mkdir()
    linked_root = tmp_path / "linked-root"
    linked_root.symlink_to(real_root, target_is_directory=True)
    store = Grid5000StateStore(root)
    store.root = linked_root
    monkeypatch.setattr(grid5000, "_reject_symlink_components", lambda _path: None)
    with pytest.raises(Grid5000StateError) as error:
        store.load(run_id)
    assert str(error.value) == "state root cannot be a symlink"


def test_state_store_load_rejects_a_symlinked_run_directory_exactly(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runs"
    root.mkdir()
    run_id = _plan().identity.run_id
    target = tmp_path / "target-run"
    target.mkdir()
    (root / run_id).symlink_to(target, target_is_directory=True)

    with pytest.raises(Grid5000StateError) as error:
        Grid5000StateStore(root).load(run_id)
    assert str(error.value) == "run state directory cannot be a symlink"


def test_state_store_load_reports_identity_directory_mismatch_exactly(
    tmp_path: Path,
) -> None:
    store = Grid5000StateStore(tmp_path / "runs")
    first = _plan()
    second_identity = replace(first.identity, training_config={"max_steps": 101})
    run_directory = store._ensure_run_directory(second_identity.run_id)
    state = Grid5000State(
        identity=first.identity,
        phase="submitting",
        scheduler_command=first.scheduler_command,
    )
    store._write_atomic(run_directory, state)

    with pytest.raises(Grid5000StateError) as error:
        store.load(second_identity.run_id)
    assert str(error.value) == "run state identity does not match its directory"


def test_state_store_submission_guards_use_fixed_state_and_lock_names(
    tmp_path: Path,
) -> None:
    store = Grid5000StateStore(tmp_path / "runs")
    plan = _plan()
    state = Grid5000State(
        identity=plan.identity,
        phase="submitting",
        scheduler_command=plan.scheduler_command,
    )
    run_directory = store._ensure_run_directory(plan.identity.run_id)
    state_path = run_directory / "state.json"
    state_path.write_text("claimed", encoding="utf-8")
    with pytest.raises(Grid5000StateError) as error:
        store.create_submitting(state)
    assert str(error.value) == "run already has durable submission state"

    state_path.unlink()
    lock_path = run_directory / ".intent.lock"
    lock_path.write_text("claimed", encoding="utf-8")
    with pytest.raises(Grid5000StateError) as error:
        store.create_submitting(state)
    assert str(error.value) == "submission state is ambiguous"


def test_submission_phase_guard_reports_an_exact_error() -> None:
    plan = _plan()
    state = Grid5000State(
        identity=plan.identity,
        phase="submitted",
        scheduler_command=plan.scheduler_command,
        job_id=123,
    )

    with pytest.raises(Grid5000StateError) as error:
        grid5000._require_submitting_state(state)
    assert str(error.value) == "submission intent must be in submitting phase"


def test_unclaimed_submission_guard_rejects_regular_state_and_lock_files(
    tmp_path: Path,
) -> None:
    state_path = tmp_path / "state.json"
    lock_path = tmp_path / ".intent.lock"
    state_path.write_text("state", encoding="utf-8")
    with pytest.raises(Grid5000StateError) as error:
        grid5000._require_unclaimed_submission(state_path, lock_path)
    assert str(error.value) == "run already has durable submission state"

    state_path.unlink()
    lock_path.write_text("lock", encoding="utf-8")
    with pytest.raises(Grid5000StateError) as error:
        grid5000._require_unclaimed_submission(state_path, lock_path)
    assert str(error.value) == "submission state is ambiguous"


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


def test_validate_checkout_checks_the_pinned_revision_and_clean_status() -> None:
    calls: list[tuple[tuple[str, ...], float]] = []

    def runner(argv: Sequence[str], *, timeout: float) -> CommandResult:
        calls.append((tuple(argv), timeout))
        if argv[-1] == "HEAD":
            return CommandResult(returncode=0, stdout=f"{SOURCE_COMMIT}\n")
        if argv[-1] == "--porcelain":
            return CommandResult(returncode=0, stdout="")
        raise AssertionError(f"unexpected git command: {argv!r}")

    _validate_checkout(Path("/checkout"), SOURCE_COMMIT, runner)

    assert calls == [
        (
            ("git", "-C", "/checkout", "rev-parse", "HEAD"),
            COMMAND_TIMEOUT_SECONDS,
        ),
        (
            ("git", "-C", "/checkout", "status", "--porcelain"),
            COMMAND_TIMEOUT_SECONDS,
        ),
    ]


def test_validate_checkout_reports_revision_command_failure() -> None:
    def runner(argv: Sequence[str], *, timeout: float) -> CommandResult:
        del timeout
        if argv[-1] == "HEAD":
            return CommandResult(returncode=7, stdout="")
        raise AssertionError(f"unexpected git command: {argv!r}")

    with pytest.raises(
        WorkerError, match="git revision command failed with exit code 7"
    ):
        _validate_checkout(Path("/checkout"), SOURCE_COMMIT, cast(Any, runner))


def test_validate_checkout_reports_an_unexpected_revision() -> None:
    def runner(argv: Sequence[str], *, timeout: float) -> CommandResult:
        del timeout
        if argv[-1] == "HEAD":
            return CommandResult(returncode=0, stdout=f"{'c' * 40}\n")
        raise AssertionError(f"unexpected git command: {argv!r}")

    with pytest.raises(WorkerError) as error:
        _validate_checkout(Path("/checkout"), SOURCE_COMMIT, cast(Any, runner))
    assert str(error.value) == "worker checkout is not at the expected source commit"


def test_validate_checkout_reports_status_command_failure() -> None:
    def runner(argv: Sequence[str], *, timeout: float) -> CommandResult:
        del timeout
        if argv[-1] == "HEAD":
            return CommandResult(returncode=0, stdout=f"{SOURCE_COMMIT}\n")
        if argv[-1] == "--porcelain":
            return CommandResult(returncode=8, stdout="")
        raise AssertionError(f"unexpected git command: {argv!r}")

    with pytest.raises(WorkerError, match="git status command failed with exit code 8"):
        _validate_checkout(Path("/checkout"), SOURCE_COMMIT, cast(Any, runner))


def test_validate_checkout_rejects_a_dirty_checkout() -> None:
    def runner(argv: Sequence[str], *, timeout: float) -> CommandResult:
        del timeout
        if argv[-1] == "HEAD":
            return CommandResult(returncode=0, stdout=f"{SOURCE_COMMIT}\n")
        if argv[-1] == "--porcelain":
            return CommandResult(returncode=0, stdout=" M tracked.py\n")
        raise AssertionError(f"unexpected git command: {argv!r}")

    with pytest.raises(WorkerError) as error:
        _validate_checkout(Path("/checkout"), SOURCE_COMMIT, runner)
    assert str(error.value) == "worker checkout is not clean"


def test_validate_checkout_wraps_unexpected_runner_errors() -> None:
    def runner(_argv: Sequence[str], *, timeout: float) -> CommandResult:
        del timeout
        raise RuntimeError("runner failed")

    with pytest.raises(WorkerError) as error:
        _validate_checkout(Path("/checkout"), SOURCE_COMMIT, cast(Any, runner))
    assert str(error.value) == "worker checkout validation could not complete"


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
        validate_compute_node(**values)


def test_worker_preflight_returns_validated_facts(tmp_path: Path) -> None:
    facts = validate_compute_node(**_valid_worker_kwargs(tmp_path))

    assert facts.job_id == 12345
    assert facts.source_commit == SOURCE_COMMIT
    assert facts.cuda_device_name == "Test GPU"


def test_worker_preflight_passes_the_validated_checkout_to_git(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    calls: list[tuple[str, ...]] = []

    def runner(argv: Sequence[str], *, timeout: float) -> CommandResult:
        del timeout
        calls.append(tuple(argv))
        assert argv[2] == str(checkout)
        if argv[-1] == "HEAD":
            return CommandResult(returncode=0, stdout=f"{SOURCE_COMMIT}\n")
        return CommandResult(returncode=0, stdout="")

    values = _valid_worker_kwargs(tmp_path)
    values["checkout"] = checkout
    values["git_runner"] = runner

    validate_compute_node(**values)

    assert len(calls) == 2


def test_worker_preflight_rejects_a_gpu_below_the_supported_cuda_capability(
    tmp_path: Path,
) -> None:
    values = _valid_worker_kwargs(tmp_path)
    values["cuda_probe"] = lambda: (True, 1, "Tesla P100", (6, 0))

    with pytest.raises(WorkerError, match="compute capability"):
        validate_compute_node(**values)


def test_default_cuda_probe_reads_one_supported_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def device_count() -> int:
            return 1

        @staticmethod
        def get_device_name(index: int) -> str:
            assert index == 0
            return "Test GPU"

        @staticmethod
        def get_device_capability(index: int) -> tuple[int, int]:
            assert index == 0
            return (8, 0)

    monkeypatch.setattr(
        grid5000_worker,
        "import_module",
        lambda name: type("FakeTorch", (), {"cuda": FakeCuda})
        if name == "torch"
        else None,
    )

    assert grid5000_worker._default_cuda_probe() == (
        True,
        1,
        "Test GPU",
        (8, 0),
    )


@pytest.mark.parametrize(("available", "count"), [(False, 1), (True, 2)])
def test_default_cuda_probe_does_not_read_device_details_unless_one_gpu_is_available(
    monkeypatch: pytest.MonkeyPatch,
    available: bool,
    count: int,
) -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return available

        @staticmethod
        def device_count() -> int:
            return count

        @staticmethod
        def get_device_name(_index: int) -> str:
            raise AssertionError("device details must not be read")

        @staticmethod
        def get_device_capability(_index: int) -> tuple[int, int]:
            raise AssertionError("device details must not be read")

    monkeypatch.setattr(
        grid5000_worker,
        "import_module",
        lambda name: type("FakeTorch", (), {"cuda": FakeCuda})
        if name == "torch"
        else None,
    )

    assert grid5000_worker._default_cuda_probe() == (
        available,
        count,
        "",
        (0, 0),
    )


def test_default_cuda_probe_reports_a_missing_torch_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing_torch(name: str) -> object:
        assert name == "torch"
        raise ModuleNotFoundError(name=name)

    monkeypatch.setattr(grid5000_worker, "import_module", missing_torch)

    with pytest.raises(WorkerError) as error:
        grid5000_worker._default_cuda_probe()
    assert str(error.value) == "Grid'5000 worker requires the torch training dependency"


def test_cuda_probe_values_wraps_unexpected_probe_errors() -> None:
    cause = RuntimeError("probe failed")

    def failing_probe() -> tuple[bool, int, str, tuple[int, int]]:
        raise cause

    with pytest.raises(WorkerError) as error:
        grid5000_worker._cuda_probe_values(failing_probe)

    assert str(error.value) == "CUDA preflight could not complete"
    assert error.value.__cause__ is cause


def test_worker_cuda_guards_preserve_exact_boundaries_and_errors() -> None:
    assert grid5000_worker._require_cuda_available(True) is None
    with pytest.raises(WorkerError) as unavailable:
        grid5000_worker._require_cuda_available(False)
    assert str(unavailable.value) == "CUDA is not available on the worker"

    assert grid5000_worker._require_one_cuda_device(1) is None
    for count in (0, 2, True):
        with pytest.raises(WorkerError) as device_count:
            grid5000_worker._require_one_cuda_device(count)
        assert str(device_count.value) == "worker must expose exactly one CUDA GPU"

    assert grid5000_worker._require_cuda_device_name("Test GPU") is None
    for name in ("", "   "):
        with pytest.raises(WorkerError) as device_name:
            grid5000_worker._require_cuda_device_name(name)
        assert str(device_name.value) == "CUDA device name is missing"


def test_worker_cuda_capability_guard_preserves_the_minimum_and_error_details() -> None:
    minimum = grid5000_worker.MINIMUM_CUDA_CAPABILITY
    assert grid5000_worker._require_cuda_capability(minimum) is None
    assert grid5000_worker._require_cuda_capability((minimum[0] + 1, 0)) is None

    below = (minimum[0] - 1, 0)
    with pytest.raises(WorkerError) as error:
        grid5000_worker._require_cuda_capability(below)
    assert str(error.value) == (
        f"CUDA compute capability {below[0]}.{below[1]} "
        f"is below the required {minimum[0]}.{minimum[1]}"
    )


@pytest.mark.parametrize("platform", ["darwin", "win32", ""])
def test_worker_platform_guard_rejects_non_linux_platforms(platform: str) -> None:
    with pytest.raises(WorkerError) as error:
        grid5000_worker._validate_worker_platform(platform)
    assert str(error.value) == "Grid'5000 worker requires a Linux compute node"


def test_worker_platform_guard_accepts_linux_and_default_platform() -> None:
    assert grid5000_worker._validate_worker_platform("linux") is None


@pytest.mark.parametrize("source_commit", ["a" * 39, "A" * 40, "not-a-commit"])
def test_worker_source_commit_guard_requires_a_lowercase_sha(
    source_commit: str,
) -> None:
    with pytest.raises(WorkerError) as error:
        grid5000_worker._validate_worker_source_commit(source_commit)
    assert str(error.value) == "expected source commit is not a pinned revision"


def test_worker_job_id_guard_accepts_positive_decimal_ids_only() -> None:
    assert grid5000_worker._worker_job_id({"OAR_JOB_ID": "123"}) == 123
    for value in ("", "0", "-1", "1.5", " 123"):
        with pytest.raises(WorkerError) as error:
            grid5000_worker._worker_job_id({"OAR_JOB_ID": value})
        assert str(error.value) == "OAR_JOB_ID must be one positive integer"
    with pytest.raises(WorkerError) as missing:
        grid5000_worker._worker_job_id({})
    assert str(missing.value) == "OAR_JOB_ID must be one positive integer"


def test_worker_checkout_path_guard_requires_absolute_non_symlink_paths(
    tmp_path: Path,
) -> None:
    absolute = tmp_path / "checkout"
    assert grid5000_worker._worker_checkout_path(absolute) == absolute
    with pytest.raises(WorkerError) as relative:
        grid5000_worker._worker_checkout_path(Path("checkout"))
    assert str(relative.value) == (
        "worker checkout must be an absolute non-symlink path"
    )

    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(WorkerError) as symlink:
        grid5000_worker._worker_checkout_path(link)
    assert str(symlink.value) == "worker checkout must be an absolute non-symlink path"


def test_worker_identity_parser_accepts_matching_immutable_inputs() -> None:
    training_config = {
        "model_name_or_path": "test-model",
        "model_revision": MODEL_REVISION,
        "max_steps": 100,
    }
    identity = Grid5000RunIdentity(
        source_commit=SOURCE_COMMIT,
        dataset_revision=GRID5000_DATASET_REVISION,
        model_name_or_path="test-model",
        model_revision=MODEL_REVISION,
        training_config=training_config,
    )

    parsed = grid5000_worker._identity_from_arguments(
        task_name="landuse",
        run_id=identity.run_id,
        source_commit=SOURCE_COMMIT,
        dataset_revision=GRID5000_DATASET_REVISION,
        model_revision=MODEL_REVISION,
        training_config_json=json.dumps(training_config),
    )

    assert parsed == identity


def test_worker_identity_parser_preserves_the_explicit_task_name() -> None:
    training_config = {
        "model_name_or_path": "test-model",
        "model_revision": MODEL_REVISION,
    }
    expected_run_id = Grid5000RunIdentity(
        source_commit=SOURCE_COMMIT,
        dataset_revision=(
            WORLDWIDE_PLACE_RELEVANCE_V2_CONTRACT.provenance.repository_revision
        ),
        model_name_or_path="test-model",
        model_revision=MODEL_REVISION,
        task_name="place-relevance-v2",
        training_config=training_config,
    ).run_id
    parsed = grid5000_worker._identity_from_arguments(
        task_name="place-relevance-v2",
        run_id=expected_run_id,
        source_commit=SOURCE_COMMIT,
        dataset_revision=(
            WORLDWIDE_PLACE_RELEVANCE_V2_CONTRACT.provenance.repository_revision
        ),
        model_revision=MODEL_REVISION,
        training_config_json=json.dumps(training_config),
    )

    assert parsed.task_name == "place-relevance-v2"


@pytest.mark.parametrize(
    ("training_config_json", "message"),
    [
        (
            "not-json",
            "worker training configuration is not valid JSON",
        ),
        ("[]", "worker training configuration must be an object"),
        (json.dumps({"max_steps": 100}), "worker run identity is invalid"),
    ],
)
def test_worker_identity_parser_rejects_invalid_training_configuration(
    training_config_json: str,
    message: str,
) -> None:
    with pytest.raises(WorkerError) as error:
        grid5000_worker._identity_from_arguments(
            task_name="landuse",
            run_id="a" * 20,
            source_commit=SOURCE_COMMIT,
            dataset_revision=GRID5000_DATASET_REVISION,
            model_revision=MODEL_REVISION,
            training_config_json=training_config_json,
        )
    assert str(error.value) == message


def test_worker_identity_parser_rejects_a_run_id_mismatch() -> None:
    training_config = {
        "model_name_or_path": "test-model",
        "model_revision": MODEL_REVISION,
    }

    with pytest.raises(WorkerError) as error:
        grid5000_worker._identity_from_arguments(
            task_name="landuse",
            run_id="a" * 20,
            source_commit=SOURCE_COMMIT,
            dataset_revision=GRID5000_DATASET_REVISION,
            model_revision=MODEL_REVISION,
            training_config_json=json.dumps(training_config),
        )
    assert str(error.value) == "worker run ID does not match its immutable inputs"


def test_worker_identity_training_config_reports_invalid_payloads_exactly() -> None:
    invalid_identity = _identity(training_config={"unknown_field": 1})
    with pytest.raises(WorkerError) as error:
        grid5000_worker._identity_training_config(invalid_identity)
    assert str(error.value) == "worker training configuration is invalid"


def test_validated_worker_config_preserves_identity_and_mismatch_guards() -> None:
    identity = _identity(
        training_config={
            "model_name_or_path": "test-model",
            "model_revision": MODEL_REVISION,
            "output_subdirectory": "models/landuse",
        }
    )
    identity_config = grid5000_worker._identity_training_config(identity)

    assert (
        grid5000_worker._validated_worker_config(
            identity,
            expected_task_name="landuse",
            contract=grid5000_worker.LANDUSE_DATASET_CONTRACT,
            training_config=None,
        )
        == identity_config
    )
    assert (
        grid5000_worker._validated_worker_config(
            identity,
            expected_task_name="landuse",
            contract=grid5000_worker.LANDUSE_DATASET_CONTRACT,
            training_config=identity_config,
        )
        == identity_config
    )

    mismatched = replace(identity_config, max_steps=identity_config.max_steps + 1)
    with pytest.raises(WorkerError) as error:
        grid5000_worker._validated_worker_config(
            identity,
            expected_task_name="landuse",
            contract=grid5000_worker.LANDUSE_DATASET_CONTRACT,
            training_config=mismatched,
        )
    assert str(error.value) == ("worker training configuration does not match identity")


def test_worker_model_identity_guard_checks_model_and_revision_exactly() -> None:
    identity = _identity(
        training_config={
            "model_name_or_path": "test-model",
            "model_revision": MODEL_REVISION,
        }
    )
    matching = TrainingConfig(
        model_name_or_path="test-model",
        model_revision=MODEL_REVISION,
    )
    assert grid5000_worker._validate_model_identity(matching, identity) is None

    with pytest.raises(WorkerError) as model_error:
        grid5000_worker._validate_model_identity(
            replace(matching, model_name_or_path="other-model"),
            identity,
        )
    assert str(model_error.value) == (
        "worker model identity does not match training configuration"
    )

    with pytest.raises(WorkerError) as revision_error:
        grid5000_worker._validate_model_identity(
            replace(matching, model_revision="c" * 40),
            identity,
        )
    assert str(revision_error.value) == (
        "worker model revision does not match training configuration"
    )


def test_worker_boundary_guard_checks_task_and_dataset_revision_exactly() -> None:
    identity = _identity()
    contract = grid5000_worker.LANDUSE_DATASET_CONTRACT
    assert (
        grid5000_worker._validate_worker_boundary(
            identity,
            "landuse",
            contract,
        )
        is None
    )

    wrong_task = replace(identity, task_name="place-relevance-v2")
    with pytest.raises(WorkerError) as task_error:
        grid5000_worker._validate_worker_boundary(wrong_task, "landuse", contract)
    assert str(task_error.value) == (
        "worker task name does not match its training boundary"
    )

    wrong_dataset = replace(identity, dataset_revision="c" * 40)
    with pytest.raises(WorkerError) as dataset_error:
        grid5000_worker._validate_worker_boundary(wrong_dataset, "landuse", contract)
    assert str(dataset_error.value) == (
        "worker dataset revision does not match the pinned contract"
    )


@pytest.mark.parametrize("published_step", [None, 10])
def test_worker_keeps_a_checkpoint_when_the_hub_is_not_newer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    published_step: int | None,
) -> None:
    checkpoint = CheckpointInfo(path=tmp_path / "checkpoint-10", global_step=10)
    identity = _identity()
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    project_config = ProjectConfig.for_remote_root(tmp_path / "data")
    monkeypatch.setattr(
        grid5000_worker,
        "latest_published_checkpoint",
        lambda *_args, **_kwargs: (
            None
            if published_step is None
            else PublishedCheckpoint("owner/model", "models/run", published_step, ())
        ),
    )

    assert (
        grid5000_worker._prefer_published_checkpoint(
            checkpoint,
            output_directory=tmp_path / "output",
            identity=identity,
            project_config=project_config,
        )
        is checkpoint
    )


def test_worker_keeps_a_checkpoint_when_hub_lookup_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = CheckpointInfo(path=tmp_path / "checkpoint-10", global_step=10)
    identity = _identity()
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    project_config = ProjectConfig.for_remote_root(tmp_path / "data")

    def fail_lookup(*_args: object, **_kwargs: object) -> PublishedCheckpoint:
        raise grid5000_worker.HubCheckpointError("temporary hub failure")

    monkeypatch.setattr(grid5000_worker, "latest_published_checkpoint", fail_lookup)

    assert (
        grid5000_worker._prefer_published_checkpoint(
            checkpoint,
            output_directory=tmp_path / "output",
            identity=identity,
            project_config=project_config,
        )
        is checkpoint
    )


def test_worker_rejects_a_newer_checkpoint_that_cannot_be_restored(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = CheckpointInfo(path=tmp_path / "checkpoint-10", global_step=10)
    identity = _identity()
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    project_config = ProjectConfig.for_remote_root(tmp_path / "data")
    monkeypatch.setattr(
        grid5000_worker,
        "latest_published_checkpoint",
        lambda *_args, **_kwargs: PublishedCheckpoint(
            "owner/model", "models/run", 20, ()
        ),
    )

    def fail_restore(*_args: object, **_kwargs: object) -> CheckpointInfo:
        raise grid5000_worker.HubCheckpointError("invalid checkpoint")

    monkeypatch.setattr(grid5000_worker, "restore_published_checkpoint", fail_restore)

    with pytest.raises(WorkerError) as error:
        grid5000_worker._prefer_published_checkpoint(
            checkpoint,
            output_directory=tmp_path / "output",
            identity=identity,
            project_config=project_config,
        )
    assert str(error.value) == "newer published checkpoint could not be restored"


def test_worker_latest_checkpoint_guard_preserves_success_and_failure_contracts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = CheckpointInfo(path=tmp_path / "checkpoint-10", global_step=10)
    identity = _identity()
    monkeypatch.setattr(
        grid5000_worker,
        "find_latest_complete_checkpoint",
        lambda *_args, **_kwargs: checkpoint,
    )
    assert (
        grid5000_worker._latest_worker_checkpoint(
            tmp_path,
            identity=identity,
            require_checkpoint=False,
        )
        is checkpoint
    )

    monkeypatch.setattr(
        grid5000_worker,
        "find_latest_complete_checkpoint",
        lambda *_args, **_kwargs: None,
    )
    with pytest.raises(WorkerError) as missing:
        grid5000_worker._latest_worker_checkpoint(
            tmp_path,
            identity=identity,
            require_checkpoint=True,
        )
    assert str(missing.value) == "no complete checkpoint is available for continuation"

    def fail_lookup(*_args: object, **_kwargs: object) -> None:
        raise grid5000_worker.CheckpointError("invalid checkpoint")

    monkeypatch.setattr(grid5000_worker, "find_latest_complete_checkpoint", fail_lookup)
    with pytest.raises(WorkerError) as invalid:
        grid5000_worker._latest_worker_checkpoint(
            tmp_path,
            identity=identity,
            require_checkpoint=False,
        )
    assert str(invalid.value) == "checkpoint evidence is invalid"


def test_worker_published_checkpoint_preference_forwards_identity_and_repository(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = CheckpointInfo(path=tmp_path / "checkpoint-10", global_step=10)
    restored = CheckpointInfo(path=tmp_path / "checkpoint-20", global_step=20)
    identity = _identity()
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    project_config = ProjectConfig.for_remote_root(tmp_path / "data")
    calls: dict[str, object] = {}
    monkeypatch.setattr(
        grid5000_worker,
        "latest_published_checkpoint",
        lambda received_identity, *, repository_id: calls.update(
            latest_identity=received_identity,
            latest_repository_id=repository_id,
        )
        or PublishedCheckpoint(repository_id, "models/run", 20, ()),
    )
    monkeypatch.setattr(
        grid5000_worker,
        "restore_published_checkpoint",
        lambda output_directory, *, identity, repository_id: calls.update(
            restore_output_directory=output_directory,
            restore_identity=identity,
            restore_repository_id=repository_id,
        )
        or restored,
    )

    assert (
        grid5000_worker._prefer_published_checkpoint(
            checkpoint,
            output_directory=tmp_path / "output",
            identity=identity,
            project_config=project_config,
        )
        is restored
    )
    assert calls == {
        "latest_identity": identity.canonical_payload,
        "latest_repository_id": project_config.target_model_repository_id,
        "restore_output_directory": tmp_path / "output",
        "restore_identity": identity.canonical_payload,
        "restore_repository_id": project_config.target_model_repository_id,
    }


def test_worker_checkpoint_preference_helpers_preserve_short_circuiting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    identity = _identity()
    checkpoint = CheckpointInfo(path=tmp_path / "checkpoint-10", global_step=10)
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    project_config = ProjectConfig.for_remote_root(tmp_path / "data")
    unpublished = TrainingConfig(model_name_or_path="test-model")
    published = replace(unpublished, publish_to_hub=True)

    def unexpected_preference(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("preference lookup should be skipped")

    monkeypatch.setattr(
        grid5000_worker,
        "_prefer_published_checkpoint",
        unexpected_preference,
    )
    assert (
        grid5000_worker._maybe_prefer_published_checkpoint(
            None,
            output_directory=tmp_path / "output",
            identity=identity,
            project_config=project_config,
            training_config=published,
        )
        is None
    )
    assert (
        grid5000_worker._maybe_prefer_published_checkpoint(
            checkpoint,
            output_directory=tmp_path / "output",
            identity=identity,
            project_config=project_config,
            training_config=unpublished,
        )
        is checkpoint
    )

    observed: dict[str, object] = {}
    monkeypatch.setattr(
        grid5000_worker,
        "_prefer_published_checkpoint",
        lambda received_checkpoint, **kwargs: observed.update(
            checkpoint=received_checkpoint,
            **kwargs,
        )
        or checkpoint,
    )
    assert (
        grid5000_worker._maybe_prefer_published_checkpoint(
            checkpoint,
            output_directory=tmp_path / "output",
            identity=identity,
            project_config=project_config,
            training_config=published,
        )
        is checkpoint
    )
    assert observed == {
        "checkpoint": checkpoint,
        "output_directory": tmp_path / "output",
        "identity": identity,
        "project_config": project_config,
    }


def test_worker_required_checkpoint_restoration_preserves_publish_and_error_contracts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    identity = _identity()
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    project_config = ProjectConfig.for_remote_root(tmp_path / "data")
    unpublished = TrainingConfig(model_name_or_path="test-model")
    with pytest.raises(WorkerError) as unpublished_error:
        grid5000_worker._restore_required_checkpoint(
            tmp_path / "output",
            identity=identity,
            project_config=project_config,
            training_config=unpublished,
        )
    assert str(unpublished_error.value) == (
        "no complete checkpoint is available for continuation"
    )

    published = replace(unpublished, publish_to_hub=True)
    restored = CheckpointInfo(path=tmp_path / "checkpoint-20", global_step=20)
    observed: dict[str, object] = {}
    monkeypatch.setattr(
        grid5000_worker,
        "restore_published_checkpoint",
        lambda output_directory, *, identity, repository_id: observed.update(
            output_directory=output_directory,
            identity=identity,
            repository_id=repository_id,
        )
        or restored,
    )
    assert (
        grid5000_worker._restore_required_checkpoint(
            tmp_path / "output",
            identity=identity,
            project_config=project_config,
            training_config=published,
        )
        is restored
    )
    assert observed == {
        "output_directory": tmp_path / "output",
        "identity": identity.canonical_payload,
        "repository_id": project_config.target_model_repository_id,
    }

    monkeypatch.setattr(
        grid5000_worker,
        "restore_published_checkpoint",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            grid5000_worker.HubCheckpointError("missing")
        ),
    )
    with pytest.raises(WorkerError) as restore_error:
        grid5000_worker._restore_required_checkpoint(
            tmp_path / "output",
            identity=identity,
            project_config=project_config,
            training_config=published,
        )
    assert str(restore_error.value) == (
        "no complete local or published checkpoint is available for continuation"
    )


def test_worker_resume_checkpoint_helper_forwards_the_output_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    identity = _identity()
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    project_config = ProjectConfig.for_remote_root(tmp_path / "data")
    training_config = TrainingConfig(model_name_or_path="test-model")
    checkpoint = CheckpointInfo(path=tmp_path / "checkpoint-10", global_step=10)
    restored = CheckpointInfo(path=tmp_path / "checkpoint-20", global_step=20)
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        grid5000_worker,
        "_maybe_prefer_published_checkpoint",
        lambda received_checkpoint, **kwargs: observed.update(
            maybe_checkpoint=received_checkpoint,
            maybe_kwargs=kwargs,
        )
        or checkpoint,
    )
    assert (
        grid5000_worker._resume_worker_checkpoint(
            checkpoint,
            identity=identity,
            output_directory=tmp_path / "output",
            project_config=project_config,
            training_config=training_config,
            require_checkpoint=True,
        )
        is checkpoint
    )
    assert observed == {
        "maybe_checkpoint": checkpoint,
        "maybe_kwargs": {
            "output_directory": tmp_path / "output",
            "identity": identity,
            "project_config": project_config,
            "training_config": training_config,
        },
    }

    observed.clear()
    monkeypatch.setattr(
        grid5000_worker,
        "_maybe_prefer_published_checkpoint",
        lambda received_checkpoint, **kwargs: observed.update(
            maybe_checkpoint=received_checkpoint,
            maybe_kwargs=kwargs,
        )
        or None,
    )
    monkeypatch.setattr(
        grid5000_worker,
        "_restore_required_checkpoint",
        lambda received_output, **kwargs: observed.update(
            restore_output=received_output,
            restore_kwargs=kwargs,
        )
        or restored,
    )
    assert (
        grid5000_worker._resume_worker_checkpoint(
            None,
            identity=identity,
            output_directory=tmp_path / "output",
            project_config=project_config,
            training_config=training_config,
            require_checkpoint=True,
        )
        is restored
    )
    assert observed["restore_output"] == tmp_path / "output"


def test_worker_remote_project_config_and_auth_guards_preserve_exact_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    assert grid5000_worker._remote_project_config(tmp_path / "data").data_root == (
        tmp_path / "data"
    )
    with pytest.raises(WorkerError) as root_error:
        grid5000_worker._remote_project_config(Path("relative-data"))
    assert str(root_error.value) == "remote worker data root is unsafe"

    no_publish = TrainingConfig(model_name_or_path="test-model")
    assert grid5000_worker._require_worker_hugging_face_auth(no_publish, {}) is None
    for config in (
        replace(no_publish, publish_to_hub=True),
        replace(no_publish, sync_trackio=True),
    ):
        with pytest.raises(WorkerError) as auth_error:
            grid5000_worker._require_worker_hugging_face_auth(config, {})
        assert str(auth_error.value) == (
            "worker Hugging Face authentication is unavailable for publication"
        )


def test_worker_main_runs_the_selected_training_boundary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    identity = _identity()
    calls: dict[str, object] = {}

    def parse_identity(**kwargs: object) -> Grid5000RunIdentity:
        calls["identity_arguments"] = kwargs
        return identity

    monkeypatch.setattr(
        grid5000_worker,
        "_identity_from_arguments",
        parse_identity,
    )

    def fake_worker(
        received_identity: Grid5000RunIdentity, **kwargs: object
    ) -> TrainingResult:
        calls["identity"] = received_identity
        calls["kwargs"] = kwargs
        return TrainingResult(
            output_directory=tmp_path / "model", train_output=object()
        )

    monkeypatch.setattr(grid5000_worker, "run_landuse_training_worker", fake_worker)
    monkeypatch.setattr(
        grid5000_worker,
        "write_completion_manifest",
        lambda received_identity, result, *, remote_data_root: calls.update(
            {
                "manifest_identity": received_identity,
                "result": result,
                "remote_data_root": remote_data_root,
            }
        ),
    )

    assert (
        grid5000_worker.main(
            [
                "--run-id",
                identity.run_id,
                "--source-commit",
                SOURCE_COMMIT,
                "--dataset-revision",
                GRID5000_DATASET_REVISION,
                "--model-revision",
                MODEL_REVISION,
                "--training-config-json",
                "{}",
                "--checkout-commit",
                "c" * 40,
                "--checkout",
                str(tmp_path / "checkout"),
                "--remote-data-root",
                str(tmp_path / "data"),
            ]
        )
        == 0
    )
    assert calls["identity_arguments"] == {
        "task_name": "landuse",
        "run_id": identity.run_id,
        "source_commit": SOURCE_COMMIT,
        "dataset_revision": GRID5000_DATASET_REVISION,
        "model_revision": MODEL_REVISION,
        "training_config_json": "{}",
    }
    assert calls["identity"] is identity
    assert calls["kwargs"] == {
        "checkout": tmp_path / "checkout",
        "checkout_source_commit": "c" * 40,
        "remote_data_root": tmp_path / "data",
        "require_checkpoint": False,
    }
    assert calls["manifest_identity"] is identity
    assert calls["result"].__class__ is TrainingResult
    assert calls["remote_data_root"] == tmp_path / "data"


def test_worker_main_uses_stable_defaults_for_checkout_and_remote_data_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity()
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        grid5000_worker,
        "_identity_from_arguments",
        lambda **_kwargs: identity,
    )
    monkeypatch.setattr(
        grid5000_worker,
        "run_landuse_training_worker",
        lambda _identity, **kwargs: calls.update(kwargs)
        or TrainingResult(output_directory=Path.cwd(), train_output=object()),
    )
    monkeypatch.setattr(
        grid5000_worker,
        "write_completion_manifest",
        lambda _identity, _result, *, remote_data_root: calls.update(
            {"manifest_remote_data_root": remote_data_root}
        ),
    )

    assert (
        grid5000_worker.main(
            [
                "--run-id",
                identity.run_id,
                "--source-commit",
                SOURCE_COMMIT,
                "--dataset-revision",
                GRID5000_DATASET_REVISION,
                "--model-revision",
                MODEL_REVISION,
                "--training-config-json",
                "{}",
            ]
        )
        == 0
    )

    assert calls["checkout"] == Path.cwd()
    assert calls["remote_data_root"] == Path.home() / REMOTE_DATA_SUBDIRECTORY
    assert calls["manifest_remote_data_root"] == (
        Path.home() / REMOTE_DATA_SUBDIRECTORY
    )


@pytest.mark.parametrize(
    "missing_argument",
    [
        "--run-id",
        "--source-commit",
        "--dataset-revision",
        "--model-revision",
        "--training-config-json",
    ],
)
def test_worker_main_keeps_identity_arguments_required(
    missing_argument: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    identity = _identity()
    arguments = [
        "--run-id",
        identity.run_id,
        "--source-commit",
        SOURCE_COMMIT,
        "--dataset-revision",
        GRID5000_DATASET_REVISION,
        "--model-revision",
        MODEL_REVISION,
        "--training-config-json",
        "{}",
    ]
    index = arguments.index(missing_argument)
    del arguments[index : index + 2]

    with pytest.raises(SystemExit) as error:
        grid5000_worker.main(arguments)

    assert error.value.code == 2
    assert (
        f"the following arguments are required: {missing_argument}"
        in capsys.readouterr().err
    )


def test_worker_main_help_describes_the_public_worker_contract(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        grid5000_worker.main(["--help"])

    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "Strict compute-node preflight and training boundary" in output
    assert "{landuse,place-relevance-v2}" in output
    assert (
        "                        task-specific training boundary (default: landuse)\n"
        in output
    )
    assert "--checkout-commit CHECKOUT_COMMIT" in output
    assert (
        "                        optional code checkout revision for an identity-\n"
        "                        preserving resume\n" in output
    )
    assert (
        "  --require-checkpoint  fail instead of restarting if no complete checkpoint\n"
        "                        is found\n" in output
    )


def test_worker_main_reports_worker_errors_without_losing_the_message(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        grid5000_worker,
        "_identity_from_arguments",
        lambda **_kwargs: (_ for _ in ()).throw(WorkerError("identity broke")),
    )

    with pytest.raises(SystemExit) as error:
        grid5000_worker.main(
            [
                "--run-id",
                "a" * 20,
                "--source-commit",
                SOURCE_COMMIT,
                "--dataset-revision",
                GRID5000_DATASET_REVISION,
                "--model-revision",
                MODEL_REVISION,
                "--training-config-json",
                "{}",
            ]
        )

    assert error.value.code == 2
    assert "identity broke" in capsys.readouterr().err


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
    assert received["config"].run_name == training_config.run_name
    assert received["project_config"] == ProjectConfig.for_remote_root(
        Path.home() / "training-data"
    )
    assert received["resume_from_checkpoint"] is None
    assert received["checkpoint_identity"] == identity.canonical_payload


def test_training_worker_preserves_all_orchestration_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    training_config = TrainingConfig(
        model_name_or_path="test-model",
        model_revision=MODEL_REVISION,
        output_subdirectory=Path("models/landuse"),
    )
    identity = _identity(
        training_config={
            "model_name_or_path": "test-model",
            "model_revision": MODEL_REVISION,
            "output_subdirectory": "models/landuse",
        }
    )
    observed: dict[str, object] = {}
    git_runner = object()
    cuda_probe = object()
    expected_project_config = ProjectConfig.for_remote_root(
        tmp_path / REMOTE_DATA_SUBDIRECTORY
    )
    local_checkpoint = CheckpointInfo(
        path=tmp_path / "data/models/landuse/checkpoint-10",
        global_step=10,
    )

    def validated_config(*args: object, **kwargs: object) -> TrainingConfig:
        observed["validated_args"] = args
        observed["validated_kwargs"] = kwargs
        return training_config

    def validate_node(**kwargs: object) -> object:
        observed["node_kwargs"] = kwargs
        return object()

    def latest_checkpoint(
        output_directory: Path,
        *,
        identity: Grid5000RunIdentity,
        require_checkpoint: bool,
    ) -> CheckpointInfo | None:
        observed["latest"] = {
            "output_directory": output_directory,
            "identity": identity,
            "require_checkpoint": require_checkpoint,
        }
        return local_checkpoint

    def resume_checkpoint(
        checkpoint: CheckpointInfo | None,
        **kwargs: object,
    ) -> CheckpointInfo:
        observed["resume"] = {"checkpoint": checkpoint, **kwargs}
        return checkpoint  # ty: ignore[invalid-return-type]

    def train(**kwargs: object) -> TrainingResult:
        observed["train"] = kwargs
        return TrainingResult(
            output_directory=tmp_path / "model", train_output=object()
        )

    monkeypatch.setattr(grid5000_worker, "_validated_worker_config", validated_config)
    monkeypatch.setattr(grid5000_worker, "validate_compute_node", validate_node)
    monkeypatch.setattr(grid5000_worker, "_latest_worker_checkpoint", latest_checkpoint)
    monkeypatch.setattr(grid5000_worker, "_resume_worker_checkpoint", resume_checkpoint)

    result = grid5000_worker._run_training_worker(
        identity,
        expected_task_name="landuse",
        contract=grid5000_worker.LANDUSE_DATASET_CONTRACT,
        checkout=tmp_path / "checkout",
        checkout_source_commit="c" * 40,
        training_config=training_config,
        remote_data_root=None,
        environ={"OAR_JOB_ID": "12345"},
        platform_name="linux",
        git_runner=git_runner,  # ty: ignore[invalid-argument-type]
        cuda_probe=cuda_probe,  # ty: ignore[invalid-argument-type]
        train=train,
        require_checkpoint=True,
    )

    assert result.output_directory == tmp_path / "model"
    assert observed["validated_kwargs"] == {
        "expected_task_name": "landuse",
        "contract": grid5000_worker.LANDUSE_DATASET_CONTRACT,
        "training_config": training_config,
    }
    assert observed["node_kwargs"] == {
        "expected_source_commit": "c" * 40,
        "checkout": tmp_path / "checkout",
        "environ": {"OAR_JOB_ID": "12345"},
        "platform_name": "linux",
        "git_runner": git_runner,
        "cuda_probe": cuda_probe,
    }
    assert observed["latest"] == {
        "output_directory": (tmp_path / REMOTE_DATA_SUBDIRECTORY / "models/landuse"),
        "identity": identity,
        "require_checkpoint": False,
    }
    assert observed["resume"] == {
        "checkpoint": local_checkpoint,
        "identity": identity,
        "output_directory": (tmp_path / REMOTE_DATA_SUBDIRECTORY / "models/landuse"),
        "project_config": expected_project_config,
        "training_config": training_config,
        "require_checkpoint": True,
    }
    assert observed["train"] == {
        "config": training_config,
        "project_config": expected_project_config,
        "resume_from_checkpoint": local_checkpoint.path,
        "checkpoint_identity": identity.canonical_payload,
    }

    observed.clear()
    grid5000_worker._run_training_worker(
        identity,
        expected_task_name="landuse",
        contract=grid5000_worker.LANDUSE_DATASET_CONTRACT,
        checkout=tmp_path / "checkout",
        training_config=training_config,
        remote_data_root=None,
        environ={"OAR_JOB_ID": "12345"},
        platform_name="linux",
        git_runner=git_runner,  # ty: ignore[invalid-argument-type]
        cuda_probe=cuda_probe,  # ty: ignore[invalid-argument-type]
        train=train,
    )
    latest_observation = cast(dict[str, object], observed["latest"])
    resume_observation = cast(dict[str, object], observed["resume"])
    assert latest_observation["require_checkpoint"] is False
    assert resume_observation["require_checkpoint"] is False


def test_landuse_worker_forwards_its_complete_boundary_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    identity = _identity()
    training_config = TrainingConfig(model_name_or_path="test-model")
    observed: dict[str, object] = {}
    git_runner = object()
    cuda_probe = object()
    train = object()

    def fake_run(received_identity: Grid5000RunIdentity, **kwargs: object) -> object:
        observed["identity"] = received_identity
        observed.update(kwargs)
        return "result"

    monkeypatch.setattr(grid5000_worker, "_run_training_worker", fake_run)

    result = run_landuse_training_worker(
        identity,
        checkout=tmp_path / "checkout",
        checkout_source_commit="c" * 40,
        training_config=training_config,
        remote_data_root=tmp_path / "remote-data",
        environ={"OAR_JOB_ID": "12345"},
        platform_name="linux",
        git_runner=git_runner,  # ty: ignore[invalid-argument-type]
        cuda_probe=cuda_probe,  # ty: ignore[invalid-argument-type]
        train=train,  # ty: ignore[invalid-argument-type]
        require_checkpoint=True,
    )

    assert result == "result"
    assert observed == {
        "identity": identity,
        "expected_task_name": "landuse",
        "contract": grid5000_worker.LANDUSE_DATASET_CONTRACT,
        "checkout": tmp_path / "checkout",
        "checkout_source_commit": "c" * 40,
        "training_config": training_config,
        "remote_data_root": tmp_path / "remote-data",
        "environ": {"OAR_JOB_ID": "12345"},
        "platform_name": "linux",
        "git_runner": git_runner,
        "cuda_probe": cuda_probe,
        "train": train,
        "require_checkpoint": True,
    }


def test_worldwide_worker_keeps_one_logical_trackio_run_name(
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


def test_worldwide_worker_forwards_its_complete_training_worker_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = Grid5000RunIdentity(
        source_commit=SOURCE_COMMIT,
        dataset_revision=(
            WORLDWIDE_PLACE_RELEVANCE_V2_CONTRACT.provenance.repository_revision
        ),
        model_name_or_path="test-model",
        model_revision=MODEL_REVISION,
        task_name="place-relevance-v2",
        training_config={"model_name_or_path": "test-model"},
    )
    training_config = TrainingConfig(model_name_or_path="test-model")
    observed: dict[str, object] = {}
    expected_result = TrainingResult(
        output_directory=tmp_path / "model",
        train_output=object(),
    )

    def checkout_runner(argv: Sequence[str], *, timeout: float) -> CommandResult:
        del argv, timeout
        return CommandResult(returncode=0)

    def cuda_probe() -> tuple[bool, int, str, tuple[int, int]]:
        return True, 1, "Test GPU", (8, 0)

    def train(**kwargs: object) -> TrainingResult:
        del kwargs
        return expected_result

    def fake_run(
        identity_arg: Grid5000RunIdentity,
        **kwargs: object,
    ) -> TrainingResult:
        observed.update(identity=identity_arg, **kwargs)
        return expected_result

    monkeypatch.setattr(grid5000_worker, "_run_training_worker", fake_run)

    result = run_place_relevance_training_worker(
        identity,
        checkout=tmp_path / "checkout",
        checkout_source_commit="c" * 40,
        training_config=training_config,
        remote_data_root=tmp_path / "remote-data",
        environ={"OAR_JOB_ID": "12345"},
        platform_name="linux",
        git_runner=checkout_runner,
        cuda_probe=cuda_probe,
        train=train,
        require_checkpoint=True,
    )

    assert result is expected_result
    assert observed == {
        "identity": identity,
        "expected_task_name": "place-relevance-v2",
        "contract": WORLDWIDE_PLACE_RELEVANCE_V2_CONTRACT,
        "checkout": tmp_path / "checkout",
        "checkout_source_commit": "c" * 40,
        "training_config": training_config,
        "remote_data_root": tmp_path / "remote-data",
        "environ": {"OAR_JOB_ID": "12345"},
        "platform_name": "linux",
        "git_runner": checkout_runner,
        "cuda_probe": cuda_probe,
        "train": train,
        "require_checkpoint": True,
    }


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
    assert received["config"].run_name == training_config.run_name


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


def test_worker_hf_auth_reads_the_explicit_utf8_token_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    hf_home = tmp_path / "hf"
    token_path = hf_home / "token"
    token_path.parent.mkdir()
    token_path.write_text(" hf_test_token \n", encoding="utf-8")
    calls: list[tuple[Path, tuple[object, ...], dict[str, object]]] = []
    original_read_text = Path.read_text

    def recording_read_text(path: Path, *args: object, **kwargs: object) -> str:
        calls.append((path, args, kwargs))
        return cast(Any, original_read_text)(path, *args, **kwargs)

    monkeypatch.setattr(grid5000_worker.Path, "read_text", recording_read_text)

    assert grid5000_worker._has_hugging_face_auth({"HF_HOME": str(hf_home)})
    assert calls == [(token_path, (), {"encoding": "utf-8"})]


def test_worker_hf_auth_uses_the_default_cache_token_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    token_path = tmp_path / ".cache" / "huggingface" / "token"
    token_path.parent.mkdir(parents=True)
    token_path.write_text("hf_test_token", encoding="utf-8")
    calls: list[Path] = []
    original_read_text = Path.read_text

    def recording_read_text(path: Path, *args: object, **kwargs: object) -> str:
        calls.append(path)
        return cast(Any, original_read_text)(path, *args, **kwargs)

    monkeypatch.setattr(
        grid5000_worker.Path,
        "home",
        classmethod(lambda _cls: tmp_path),
    )
    monkeypatch.setattr(grid5000_worker.Path, "read_text", recording_read_text)

    assert grid5000_worker._has_hugging_face_auth({})
    assert calls == [token_path]


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
    assert manifest.name == "completion.json"
    assert payload["run_id"] == identity.run_id
    assert payload["output_directory"] == "model"
    assert payload["metrics"] == {"eval_f1": 0.7, "eval_macro_f1": 0.6}
    assert "token" not in manifest.read_text(encoding="utf-8").casefold()
    assert manifest.stat().st_mode & 0o777 == 0o600


def test_worker_manifest_metric_filters_are_case_insensitive_and_finite() -> None:
    assert grid5000_worker._safe_metric_key("eval_accuracy") is True
    for key in ("token", "SECRET_VALUE", "user_password", 1):
        assert grid5000_worker._safe_metric_key(key) is False

    for value in (True, 1, "0.5", 0.5):
        assert grid5000_worker._safe_metric_value(value) is True
    for value in ([], float("nan"), float("inf"), float("-inf")):
        assert grid5000_worker._safe_metric_value(value) is False

    assert grid5000_worker._safe_metric_entry("eval_f1", 0.7) == (
        "eval_f1",
        0.7,
    )
    assert grid5000_worker._safe_metric_entry("token", 0.7) is None
    assert grid5000_worker._safe_metric_entry("eval_f1", object()) is None


def test_worker_manifest_metric_mapping_rejects_invalid_mappings_and_drops_unsafe_entries() -> (
    None
):
    assert grid5000_worker._safe_manifest_metrics(None) == {}
    assert grid5000_worker._safe_manifest_metrics(
        {"eval_f1": 0.7, "token": "secret", "bad": float("nan")}
    ) == {"eval_f1": 0.7}

    with pytest.raises(WorkerError) as error:
        grid5000_worker._safe_manifest_metrics(cast(Any, []))
    assert str(error.value) == "training metrics are invalid"


def test_worker_completion_payload_preserves_the_complete_manifest_schema(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "remote-data"
    output_directory = data_root / "models" / "landuse"
    identity = _identity()
    publication = ModelPublicationResult(
        repository_id="NoeFlandre/osm-polygon-sentence-classifier",
        commit_id="c" * 40,
        commit_url="https://huggingface.co/commit/" + "c" * 40,
        files=("README.md", "config.json"),
    )
    result = TrainingResult(
        output_directory=output_directory,
        train_output=object(),
        model_publication=publication,
        tracking_space_id="trackio-space",
        metrics={"eval_accuracy": 0.75},
    )

    assert grid5000_worker._completion_payload(
        identity, result, data_root, output_directory
    ) == {
        "schema_version": 1,
        "run_id": identity.run_id,
        "source_commit": SOURCE_COMMIT,
        "dataset_revision": GRID5000_DATASET_REVISION,
        "model_name_or_path": "test-model",
        "model_revision": MODEL_REVISION,
        "output_directory": "models/landuse",
        "model_publication": {
            "repository_id": "NoeFlandre/osm-polygon-sentence-classifier",
            "commit_id": "c" * 40,
            "commit_url": "https://huggingface.co/commit/" + "c" * 40,
            "files": ["README.md", "config.json"],
        },
        "tracking_space_id": "trackio-space",
        "metrics": {"eval_accuracy": 0.75},
    }


def test_worker_model_publication_payload_preserves_the_commit_schema() -> None:
    publication = ModelPublicationResult(
        repository_id="owner/model",
        commit_id="c" * 40,
        commit_url="https://huggingface.co/commit/" + "c" * 40,
        files=("README.md", "config.json"),
    )

    assert _model_publication_payload(publication) == {
        "repository_id": "owner/model",
        "commit_id": "c" * 40,
        "commit_url": "https://huggingface.co/commit/" + "c" * 40,
        "files": ["README.md", "config.json"],
    }


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


def test_worker_completion_path_guards_validate_each_path_and_error_exactly(
    tmp_path: Path,
) -> None:
    absolute_root = tmp_path / "root"
    absolute_output = absolute_root / "model"
    assert (
        grid5000_worker._require_absolute_completion_paths(
            absolute_root,
            absolute_output,
        )
        is None
    )
    with pytest.raises(WorkerError) as relative_root:
        grid5000_worker._require_absolute_completion_paths(
            Path("root"),
            absolute_output,
        )
    assert str(relative_root.value) == "completion paths must be absolute"
    with pytest.raises(WorkerError) as relative_output:
        grid5000_worker._require_absolute_completion_paths(
            absolute_root,
            Path("model"),
        )
    assert str(relative_output.value) == "completion paths must be absolute"

    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(WorkerError) as symlink:
        grid5000_worker._require_non_symlink_completion_paths(link, absolute_output)
    assert str(symlink.value) == "completion paths must not be symlinks"
    assert (
        grid5000_worker._require_non_symlink_completion_paths(
            absolute_root,
            absolute_output,
        )
        is None
    )

    with pytest.raises(WorkerError) as outside:
        grid5000_worker._require_output_inside_data_root(
            tmp_path / "outside",
            absolute_root,
        )
    assert str(outside.value) == (
        "training output is outside the managed remote data root"
    )
    assert (
        grid5000_worker._require_output_inside_data_root(
            absolute_output,
            absolute_root,
        )
        is None
    )


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


def test_worker_completion_manifest_creates_nested_root_with_secure_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    remote_root = tmp_path / "nested" / "remote-data"
    output_directory = remote_root / "models" / "landuse"
    mkdir_calls: list[tuple[Path, tuple[object, ...], dict[str, object]]] = []
    chmod_calls: list[tuple[object, object]] = []
    original_mkdir = Path.mkdir

    def recording_mkdir(
        path: Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        mkdir_calls.append((path, args, kwargs))
        cast(Any, original_mkdir)(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", recording_mkdir)
    monkeypatch.setattr(
        grid5000_worker.os,
        "chmod",
        lambda path, mode: chmod_calls.append((path, mode)),
    )

    manifest = write_completion_manifest(
        _identity(),
        TrainingResult(output_directory=output_directory, train_output=object()),
        remote_data_root=remote_root,
    )

    assert manifest == remote_root / "completion.json"
    assert (
        remote_root,
        (),
        {"parents": True, "exist_ok": True, "mode": 0o700},
    ) in mkdir_calls
    assert (remote_root, 0o700) in chmod_calls


def test_worker_completion_manifest_reports_the_exact_symlink_error(
    tmp_path: Path,
) -> None:
    output_directory = tmp_path / "model"
    output_directory.mkdir()
    manifest = tmp_path / "completion.json"
    manifest.symlink_to(tmp_path / "outside.json")

    with pytest.raises(WorkerError) as error:
        write_completion_manifest(
            _identity(),
            TrainingResult(output_directory=output_directory, train_output=object()),
            remote_data_root=tmp_path,
        )

    assert str(error.value) == "completion manifest cannot be a symlink"


def test_private_completion_writer_uses_a_stable_temp_name_and_utf8(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[Path, str, dict[str, object]]] = []
    original_write_text = Path.write_text

    def recording_write_text(
        path: Path,
        data: str,
        *args: object,
        **kwargs: object,
    ) -> int:
        calls.append((path, data, kwargs))
        return cast(Any, original_write_text)(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", recording_write_text)
    manifest = tmp_path / "completion.json"

    _write_completion_manifest(manifest, {"label": "é"})

    assert calls == [
        (
            tmp_path / ".completion.json.tmp",
            '{"label": "é"}\n',
            {"encoding": "utf-8"},
        )
    ]
    assert manifest.read_text(encoding="utf-8") == '{"label": "é"}\n'


def test_private_completion_writer_wraps_failures_and_tolerates_missing_temp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    underlying = OSError("cannot create temporary manifest")

    def fail_write_text(*_args: object, **_kwargs: object) -> int:
        raise underlying

    monkeypatch.setattr(Path, "write_text", fail_write_text)

    with pytest.raises(WorkerError) as error:
        _write_completion_manifest(tmp_path / "completion.json", {"ok": True})

    assert str(error.value) == "completion manifest cannot be written"
    assert error.value.__cause__ is underlying
