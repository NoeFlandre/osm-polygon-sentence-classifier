import argparse
import json
from types import SimpleNamespace
from typing import Any, cast

import pytest

from osm_polygon_sentence_classifier import ablation_study, grid5000_cli
from osm_polygon_sentence_classifier.grid5000 import (
    Grid5000ConfigurationError,
    Grid5000Plan,
    Grid5000RunIdentity,
    Grid5000StateError,
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


def _command_parser(
    command: str,
    *,
    task_name: str = "landuse",
) -> argparse.ArgumentParser:
    parser = grid5000_cli._parser(cast(Any, task_name))
    subparsers = next(
        action
        for action in cast(Any, parser)._actions
        if isinstance(getattr(action, "choices", None), dict)
    )
    return cast(argparse.ArgumentParser, subparsers.choices[command])


def _option_actions(parser: argparse.ArgumentParser) -> dict[str, Any]:
    return {
        option: action
        for action in cast(Any, parser)._actions
        for option in action.option_strings
    }


def test_parser_exposes_the_stable_command_surface() -> None:
    parser = grid5000_cli._parser("landuse")
    subparsers = next(
        action
        for action in cast(Any, parser)._actions
        if isinstance(getattr(action, "choices", None), dict)
    )

    assert tuple(subparsers.choices) == (
        "plan",
        "submit",
        "run",
        "resume",
        "status",
        "ablations",
    )


def test_parser_preserves_description_command_help_and_required_gate() -> None:
    parser = grid5000_cli._parser("landuse")
    subparsers = next(
        action
        for action in cast(Any, parser)._actions
        if isinstance(getattr(action, "choices", None), dict)
    )
    choice_help = {action.dest: action.help for action in subparsers._choices_actions}

    assert parser.description == (
        "Plan, submit, or autonomously run one guarded landuse Grid'5000 training run"
    )
    assert subparsers.required is True
    assert choice_help == {
        "plan": "print a side-effect-free plan",
        "submit": "plan or explicitly submit",
        "run": "autonomously probe sites, prepare, submit, monitor, and publish",
        "resume": "resume a durable autonomous run by its run ID",
        "status": "print one local autonomous run state",
        "ablations": "plan or autonomously run the reproducible landuse ablation study",
    }


def test_plan_parser_preserves_required_types_defaults_choices_and_help() -> None:
    parser = _command_parser("plan")
    actions = _option_actions(parser)
    parsed = parser.parse_args(
        [
            "--site",
            "nancy",
            "--source-commit",
            SOURCE_COMMIT,
            "--model-revision",
            MODEL_REVISION,
            "--model-name",
            "model-name",
            "--walltime-seconds",
            "120",
            "--max-steps",
            "12",
            "--policy-type",
            "day",
            "--publish",
            "--sync-trackio",
            "--container-image",
            "image",
            "--container-runtime",
            "docker",
        ]
    )

    assert parsed.site == "nancy"
    assert parsed.source_commit == SOURCE_COMMIT
    assert parsed.model_revision == MODEL_REVISION
    assert parsed.model_name == "model-name"
    assert parsed.walltime_seconds == 120
    assert parsed.max_steps == 12
    assert parsed.policy_type == "day"
    assert parsed.publish is True
    assert parsed.sync_trackio is True
    assert parsed.container_image == "image"
    assert parsed.container_runtime == "docker"
    assert actions["--site"].required is True
    assert actions["--source-commit"].required is True
    assert actions["--model-revision"].required is True
    assert actions["--model-name"].default == "jhu-clsp/mmBERT-small"
    assert actions["--walltime-seconds"].default is None
    assert actions["--walltime-seconds"].type is int
    assert actions["--max-steps"].default is None
    assert actions["--max-steps"].type is int
    assert actions["--policy-type"].choices == ("day", "night")
    assert actions["--policy-type"].default == "night"
    assert actions["--policy-type"].help == (
        "Grid'5000 policy window; day allocations are limited to one hour"
    )
    assert actions["--publish"].const is True
    assert actions["--publish"].default is False
    assert actions["--publish"].help == (
        "publish the completed model to the project Hugging Face repository"
    )
    assert actions["--sync-trackio"].const is True
    assert actions["--sync-trackio"].default is False
    assert actions["--sync-trackio"].help == (
        "publish static Trackio metric snapshots after checkpoints"
    )
    assert actions["--container-runtime"].choices == ("auto", "docker", "podman")
    assert actions["--container-runtime"].default == "auto"
    assert actions["--container-image"].default is None
    assert actions["--container-image"].help == (
        "run the worker in this preloaded Docker/Podman image"
    )
    assert actions["--container-runtime"].help == (
        "container runtime to use when --container-image is supplied"
    )


def test_plan_parser_rejects_missing_required_values_invalid_types_and_policies() -> (
    None
):
    parser = _command_parser("plan")

    with pytest.raises(SystemExit):
        parser.parse_args([])
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--site",
                "nancy",
                "--source-commit",
                SOURCE_COMMIT,
                "--model-revision",
                MODEL_REVISION,
                "--max-steps",
                "not-an-integer",
            ]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--site",
                "nancy",
                "--source-commit",
                SOURCE_COMMIT,
                "--model-revision",
                MODEL_REVISION,
                "--policy-type",
                "auto",
            ]
        )


def test_autonomous_parser_preserves_defaults_types_and_choices() -> None:
    parser = _command_parser("run")
    actions = _option_actions(parser)
    parsed = parser.parse_args(
        [
            "--site",
            "nancy",
            "--site",
            "lille",
            "--source-commit",
            SOURCE_COMMIT,
            "--model-revision",
            MODEL_REVISION,
            "--max-steps",
            "12",
            "--walltime-seconds",
            "900",
            "--policy-type",
            "night",
            "--gpu-memory-mb",
            "9000",
            "--max-workers",
            "2",
            "--max-continuations",
            "4",
            "--publish",
            "--sync-trackio",
            "--container-image",
            "image",
            "--container-runtime",
            "podman",
            "--keep-remote",
        ]
    )

    assert parsed.site == ["nancy", "lille"]
    assert parsed.source_commit == SOURCE_COMMIT
    assert parsed.model_revision == MODEL_REVISION
    assert parsed.max_steps == 12
    assert parsed.walltime_seconds == 900
    assert parsed.policy_type == "night"
    assert parsed.gpu_memory_mb == 9000
    assert parsed.max_workers == 2
    assert parsed.max_continuations == 4
    assert parsed.publish is True
    assert parsed.sync_trackio is True
    assert parsed.container_image == "image"
    assert parsed.container_runtime == "podman"
    assert parsed.keep_remote is True
    assert actions["--site"].default is None
    assert actions["--source-commit"].default is None
    assert actions["--model-name"].default == "jhu-clsp/mmBERT-small"
    assert actions["--max-steps"].default is None
    assert actions["--max-steps"].type is int
    assert actions["--model-revision"].required is True
    assert actions["--walltime-seconds"].default == 1_200
    assert actions["--walltime-seconds"].type is int
    assert actions["--policy-type"].choices == ("auto", "day", "night")
    assert actions["--policy-type"].default == "auto"
    assert actions["--gpu-memory-mb"].default == 8_000
    assert actions["--gpu-memory-mb"].type is int
    assert actions["--max-workers"].default == 4
    assert actions["--max-workers"].type is int
    assert actions["--max-continuations"].default == 3
    assert actions["--max-continuations"].type is int
    assert actions["--max-continuations"].help == (
        "maximum bounded same-site checkpoint successors (default: 3)"
    )
    for option in ("--publish", "--sync-trackio", "--keep-remote"):
        assert actions[option].const is True
        assert actions[option].default is False
    assert actions["--container-runtime"].choices == ("auto", "docker", "podman")
    assert actions["--container-runtime"].default == "auto"
    assert actions["--container-image"].default is None
    assert actions["--container-image"].help == (
        "run each worker in this preloaded Docker/Podman image"
    )
    assert actions["--container-runtime"].help == (
        "container runtime to use when --container-image is supplied"
    )
    assert actions["--keep-remote"].help == (
        "retain the managed per-run remote data after successful verification"
    )


def test_autonomous_parser_preserves_public_help_and_task_budget() -> None:
    parser = _command_parser("run", task_name="place-relevance-v2")
    actions = _option_actions(parser)

    assert actions["--site"].help == (
        "Grid'5000 frontend; repeat to restrict discovery (default: all sites)"
    )
    assert actions["--walltime-seconds"].help == (
        "short one-GPU allocation duration (default: 20 minutes)"
    )
    assert actions["--max-continuations"].default == 40
    assert actions["--max-continuations"].help == (
        "maximum bounded same-site checkpoint successors (default: 40)"
    )


def test_ablation_parser_preserves_defaults_and_execution_gate() -> None:
    parser = _command_parser("ablations", task_name="place-relevance-v2")
    actions = _option_actions(parser)
    parsed = parser.parse_args(
        [
            "--source-commit",
            SOURCE_COMMIT,
            "--allow-source-commit-update",
            "--model-revision",
            MODEL_REVISION,
            "--site",
            "nancy",
            "--walltime-seconds",
            "900",
            "--policy-type",
            "day",
            "--gpu-memory-mb",
            "9000",
            "--max-workers",
            "2",
            "--max-continuations",
            "4",
            "--container-image",
            "image",
            "--container-runtime",
            "docker",
            "--keep-remote",
            "--execute",
        ]
    )

    assert parsed.source_commit == SOURCE_COMMIT
    assert parsed.allow_source_commit_update is True
    assert parsed.model_revision == MODEL_REVISION
    assert parsed.site == ["nancy"]
    assert parsed.walltime_seconds == 900
    assert parsed.policy_type == "day"
    assert parsed.gpu_memory_mb == 9000
    assert parsed.max_workers == 2
    assert parsed.max_continuations == 4
    assert parsed.container_image == "image"
    assert parsed.container_runtime == "docker"
    assert parsed.keep_remote is True
    assert parsed.execute is True
    assert actions["--source-commit"].default is None
    assert actions["--model-revision"].default == grid5000_cli.DEFAULT_MODEL_REVISION
    assert actions["--model-name"].default == "jhu-clsp/mmBERT-small"
    assert actions["--site"].default is None
    assert actions["--walltime-seconds"].default == 1_200
    assert actions["--walltime-seconds"].type is int
    assert actions["--policy-type"].choices == ("auto", "day", "night")
    assert actions["--policy-type"].default == "auto"
    assert actions["--gpu-memory-mb"].default == 8_000
    assert actions["--max-workers"].default == 4
    assert actions["--max-continuations"].type is int
    assert actions["--max-continuations"].default == 40
    for option in ("--allow-source-commit-update", "--keep-remote", "--execute"):
        assert actions[option].const is True
        assert actions[option].default is False
    assert actions["--source-commit"].help == (
        "clean source revision (default: current clean checkout)"
    )
    assert actions["--model-revision"].help == "pinned base-model revision"
    assert actions["--allow-source-commit-update"].help == (
        "adopt a new source revision only for an incomplete, idle study"
    )
    assert actions["--model-revision"].help == "pinned base-model revision"
    assert actions["--site"].help == (
        "Grid'5000 frontend; repeat to restrict discovery (default: all sites)"
    )
    assert actions["--container-image"].default is None
    assert actions["--container-image"].help == (
        "run each worker in this preloaded Docker/Podman image"
    )
    assert actions["--container-runtime"].help == (
        "container runtime to use when --container-image is supplied"
    )
    assert actions["--container-runtime"].choices == ("auto", "docker", "podman")
    assert actions["--keep-remote"].help == (
        "retain exact managed remote study run roots after completion"
    )
    assert actions["--execute"].help == (
        "cross the explicit gate and perform Grid'5000 and Hugging Face actions"
    )


def test_resume_submit_run_and_status_parsers_preserve_execution_contracts() -> None:
    submit_actions = _option_actions(_command_parser("submit"))
    assert submit_actions["--execute"].const is True
    assert submit_actions["--execute"].default is False
    assert submit_actions["--execute"].help == (
        "cross the explicit gate and run policy, quota, and OAR checks"
    )

    run_actions = _option_actions(_command_parser("run"))
    assert run_actions["--execute"].const is True
    assert run_actions["--execute"].default is False
    assert run_actions["--execute"].help == (
        "cross the explicit gate and perform remote/Hugging Face actions"
    )

    resume_actions = _option_actions(_command_parser("resume"))
    assert resume_actions["--run-id"].required is True
    assert resume_actions["--execute"].const is True
    assert resume_actions["--execute"].default is False
    assert resume_actions["--execute"].help == (
        "cross the explicit gate and continue remote actions"
    )
    assert resume_actions["--max-continuations"].default is None
    assert resume_actions["--max-continuations"].type is int
    assert resume_actions["--max-continuations"].help == (
        "extend a failed run beyond its persisted continuation limit"
    )
    assert resume_actions["--policy-type"].choices == ("auto", "day", "night")
    assert resume_actions["--policy-type"].default is None
    assert resume_actions["--policy-type"].help == (
        "override the persisted policy window for a legacy run"
    )

    status_actions = _option_actions(_command_parser("status"))
    assert status_actions["--run-id"].required is True

    landuse_ablation_actions = _option_actions(_command_parser("ablations"))
    assert landuse_ablation_actions["--max-continuations"].default == 6
    place_parser = _command_parser("ablations", task_name="place-relevance-v2")
    assert _option_actions(place_parser)["--max-continuations"].default == 40
    place_run_parser = _command_parser("run", task_name="place-relevance-v2")
    assert _option_actions(place_run_parser)["--max-continuations"].default == 40


@pytest.mark.parametrize(
    ("facts", "policy_override", "expected"),
    [
        ({}, None, ("auto", 1_200, True)),
        ({"requested_policy_type": "night"}, None, ("night", 1_200, True)),
        ({"requested_policy_type": "night"}, "day", ("day", 1_200, True)),
        (
            {
                "requested_policy_type": "day",
                "cleanup": False,
                "allocation": {
                    "policy_type": "night",
                    "walltime_seconds": 900,
                },
            },
            None,
            ("day", 900, False),
        ),
        (
            {
                "requested_policy_type": "night",
                "allocation": {"policy_type": "day", "walltime_seconds": 900},
            },
            None,
            ("night", 900, True),
        ),
        (
            {
                "requested_policy_type": "invalid",
                "allocation": {"policy_type": "night", "walltime_seconds": 900},
            },
            None,
            ("night", 900, True),
        ),
        (
            {"requested_policy_type": "invalid", "allocation": {}},
            None,
            ("auto", 1_200, True),
        ),
        (
            {"requested_policy_type": "auto", "allocation": {"policy_type": "night"}},
            None,
            ("auto", 1_200, True),
        ),
        (
            {"requested_policy_type": "day", "allocation": []},
            None,
            ("day", 1_200, True),
        ),
    ],
)
def test_state_allocation_settings_preserve_legacy_defaults_and_overrides(
    facts: dict[str, object],
    policy_override: str | None,
    expected: tuple[object, object, object],
) -> None:
    assert (
        grid5000_cli._state_allocation_settings(
            facts,
            policy_type_override=policy_override,
        )
        == expected
    )


def test_state_container_settings_apply_legacy_defaults_and_validation() -> None:
    assert grid5000_cli._state_container_settings({}) == (None, "auto")
    image = "registry.example/model@sha256:" + "a" * 64
    for runtime in ("auto", "docker", "podman"):
        assert grid5000_cli._state_container_settings(
            {"container_image": image, "container_runtime": runtime}
        ) == (image, runtime)

    with pytest.raises(Grid5000StateError) as error:
        grid5000_cli._state_container_settings({"container_image": 42})
    assert str(error.value) == "autonomous container image is invalid"

    with pytest.raises(Grid5000StateError) as error:
        grid5000_cli._state_container_settings({"container_runtime": "singularity"})
    assert str(error.value) == "autonomous container runtime is invalid"

    with pytest.raises(Grid5000StateError) as error:
        grid5000_cli._state_container_settings(
            {"container_image": "not-an-immutable-image"}
        )
    assert str(error.value) == "autonomous container settings are invalid"


@pytest.mark.parametrize("runtime", ["auto", "docker", "podman"])
def test_normalize_container_runtime_preserves_supported_values(
    runtime: str,
) -> None:
    assert grid5000_cli._normalize_container_runtime(runtime) == runtime


@pytest.mark.parametrize(
    ("facts", "expected"),
    [
        ({}, 8_000),
        ({"requirements": {"gpu_memory_mb": 12_000}}, 12_000),
        ({"requirements": {}}, 8_000),
        ({"requirements": []}, 8_000),
        ({"requirements": {"gpu_memory_mb": True}}, 8_000),
        ({"requirements": {"gpu_memory_mb": "12000"}}, 8_000),
        ({"requirements": {"gpu_memory_mb": 12.0}}, 8_000),
    ],
)
def test_state_gpu_memory_normalizes_invalid_legacy_values(
    facts: dict[str, object], expected: int
) -> None:
    assert grid5000_cli._state_gpu_memory(facts) == expected


@pytest.mark.parametrize("value", ["auto", "day", "night"])
def test_state_policy_preserves_each_supported_policy(value: str) -> None:
    assert grid5000_cli._state_policy(value) == value


@pytest.mark.parametrize("value", [None, "invalid", 1, True])
def test_state_policy_defaults_invalid_values_to_auto(value: object) -> None:
    assert grid5000_cli._state_policy(value) == "auto"


def test_state_sites_filters_invalid_entries_and_preserves_default_sites() -> None:
    assert grid5000_cli._state_sites({}) == grid5000_cli.DEFAULT_SITES
    assert grid5000_cli._state_sites({"sites": "nancy"}) == grid5000_cli.DEFAULT_SITES
    assert grid5000_cli._state_sites({"sites": []}) == grid5000_cli.DEFAULT_SITES
    assert grid5000_cli._state_sites({"sites": ["nancy", 42, "lille"]}) == (
        "nancy",
        "lille",
    )
    assert (
        grid5000_cli._state_sites({"sites": [42, None]}) == grid5000_cli.DEFAULT_SITES
    )


def test_state_facts_and_cleanup_keep_safe_defaults() -> None:
    assert grid5000_cli._state_facts({"facts": {"cleanup": False}}) == {
        "cleanup": False
    }
    assert grid5000_cli._state_facts({"facts": []}) == {}
    assert grid5000_cli._state_facts({}) == {}
    assert grid5000_cli._state_cleanup(True) is True
    assert grid5000_cli._state_cleanup(False) is False
    assert grid5000_cli._state_cleanup("false") is True
    assert grid5000_cli._state_cleanup(0) is True


def test_persisted_continuation_and_worker_commit_settings_validate_state() -> None:
    assert grid5000_cli._persisted_continuation_limit({}) == 3
    assert grid5000_cli._persisted_continuation_limit({"max_continuations": 1}) == 1
    assert grid5000_cli._persisted_continuation_limit({"max_continuations": 4}) == 4
    for value in (0, -1, True, "4"):
        with pytest.raises(Grid5000StateError) as error:
            grid5000_cli._persisted_continuation_limit({"max_continuations": value})
        assert str(error.value) == "autonomous continuation limit is invalid"

    assert grid5000_cli._persisted_worker_source_commit({}) is None
    assert (
        grid5000_cli._persisted_worker_source_commit(
            {"worker_source_commit": SOURCE_COMMIT}
        )
        == SOURCE_COMMIT
    )
    with pytest.raises(Grid5000StateError) as error:
        grid5000_cli._persisted_worker_source_commit({"worker_source_commit": 42})
    assert str(error.value) == "autonomous worker source commit is invalid"


def test_continuation_override_requires_a_failed_run_and_larger_limit() -> None:
    assert grid5000_cli._continuation_limit_with_override({}, 3, None) == 3
    assert (
        grid5000_cli._continuation_limit_with_override({"phase": "failed"}, 3, 4) == 4
    )
    with pytest.raises(Grid5000StateError) as error:
        grid5000_cli._continuation_limit_with_override({"phase": "running"}, 3, 4)
    assert str(error.value) == "--max-continuations can only extend a failed run"
    for value in (3, 2, 0, -1, True):
        with pytest.raises(Grid5000StateError) as error:
            grid5000_cli._continuation_limit_with_override(
                {"phase": "failed"}, 3, value
            )
        assert str(error.value) == (
            "--max-continuations must be greater than the persisted limit"
        )
    with pytest.raises(Grid5000StateError) as error:
        grid5000_cli._validate_continuation_override(cast(Any, "4"), 3)
    assert str(error.value) == (
        "--max-continuations must be greater than the persisted limit"
    )


def test_state_continuation_settings_preserve_persisted_worker_commit() -> None:
    facts = {"max_continuations": 3, "worker_source_commit": SOURCE_COMMIT}

    assert grid5000_cli._state_continuation_settings(
        {},
        facts,
        max_continuations_override=None,
        worker_source_commit_override=None,
    ) == (3, SOURCE_COMMIT)
    assert grid5000_cli._state_continuation_settings(
        {},
        facts,
        max_continuations_override=None,
        worker_source_commit_override="e" * 40,
    ) == (3, "e" * 40)


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
    assert (
        payload["identity"]["dataset_revision"]
        == grid5000_cli.task_contract("landuse").provenance.repository_revision
    )


def test_plan_preserves_explicit_task_and_training_inputs(capsys) -> None:
    image = "registry.example/model@sha256:" + "a" * 64
    exit_code = grid5000_cli.main(
        [
            "plan",
            "--site",
            "lille",
            "--source-commit",
            SOURCE_COMMIT,
            "--model-revision",
            MODEL_REVISION,
            "--model-name",
            "custom-model",
            "--walltime-seconds",
            "900",
            "--max-steps",
            "123",
            "--policy-type",
            "night",
            "--container-image",
            image,
            "--container-runtime",
            "docker",
        ],
        task_name="place-relevance-v2",
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["allocation"]["site"] == "lille"
    assert payload["allocation"]["walltime_seconds"] == 900
    assert payload["allocation"]["policy_type"] == "night"
    assert payload["container_image"] == image
    assert payload["container_runtime"] == "docker"
    assert payload["identity"]["task_name"] == "place-relevance-v2"
    assert payload["identity"]["source_commit"] == SOURCE_COMMIT
    assert payload["identity"]["model_name_or_path"] == "custom-model"
    assert payload["identity"]["model_revision"] == MODEL_REVISION
    assert payload["identity"]["training_config"]["max_steps"] == 123
    assert payload["identity"]["training_config"]["model_revision"] == MODEL_REVISION
    assert payload["identity"]["training_config"]["output_subdirectory"] == (
        "studies/place-relevance-v2/baseline/models"
    )
    assert (
        payload["identity"]["dataset_revision"]
        == grid5000_cli.task_contract(
            "place-relevance-v2"
        ).provenance.repository_revision
    )


def test_place_relevance_v2_autonomous_plan_records_dataset_revision(capsys) -> None:
    exit_code = grid5000_cli.main(
        [
            "run",
            "--source-commit",
            SOURCE_COMMIT,
            "--model-revision",
            MODEL_REVISION,
        ],
        task_name="place-relevance-v2",
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert (
        payload["identity"]["dataset_revision"]
        == grid5000_cli.task_contract(
            "place-relevance-v2"
        ).provenance.repository_revision
    )


def test_landuse_autonomous_plan_records_dataset_revision(capsys) -> None:
    exit_code = grid5000_cli.main(
        [
            "run",
            "--source-commit",
            SOURCE_COMMIT,
            "--model-revision",
            MODEL_REVISION,
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert (
        payload["identity"]["dataset_revision"]
        == grid5000_cli.task_contract("landuse").provenance.repository_revision
    )


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
    payload = json.loads(capsys.readouterr().out)
    assert payload["executed"] is False
    assert payload["job_id"] is None
    assert payload["plan"]["identity"]["source_commit"] == SOURCE_COMMIT


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
    assert payload["policy_type"] == "auto"
    assert payload["gpu_memory_mb"] == 8_000
    assert payload["max_continuations"] == 3
    assert payload["publish"] is True
    assert payload["sync_trackio"] is True
    assert payload["container_image"] is None
    assert payload["container_runtime"] == "auto"
    assert payload["cleanup"] is True
    assert payload["identity"]["source_commit"] == SOURCE_COMMIT
    assert payload["identity"]["model_revision"] == MODEL_REVISION


def test_autonomous_plan_preserves_explicit_runtime_and_budget_inputs(capsys) -> None:
    image = "registry.example/model@sha256:" + "b" * 64
    exit_code = grid5000_cli.main(
        [
            "run",
            "--site",
            "lille",
            "--source-commit",
            SOURCE_COMMIT,
            "--model-revision",
            MODEL_REVISION,
            "--model-name",
            "custom-model",
            "--max-steps",
            "123",
            "--walltime-seconds",
            "900",
            "--policy-type",
            "night",
            "--gpu-memory-mb",
            "9000",
            "--max-workers",
            "2",
            "--max-continuations",
            "4",
            "--publish",
            "--sync-trackio",
            "--container-image",
            image,
            "--container-runtime",
            "podman",
            "--keep-remote",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["sites"] == ["lille"]
    assert payload["walltime_seconds"] == 900
    assert payload["policy_type"] == "night"
    assert payload["gpu_memory_mb"] == 9_000
    assert payload["max_continuations"] == 4
    assert payload["publish"] is True
    assert payload["sync_trackio"] is True
    assert payload["container_image"] == image
    assert payload["container_runtime"] == "podman"
    assert payload["cleanup"] is False
    assert payload["identity"]["model_name_or_path"] == "custom-model"
    assert payload["identity"]["training_config"]["max_steps"] == 123


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
    captured_configs: list[object] = []

    class FakeController:
        def __init__(self, config, *, emit) -> None:
            captured_configs.append(config)
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
            "--max-workers",
            "2",
            "--execute",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured_configs
    assert cast(Any, captured_configs[0]).identity.source_commit == SOURCE_COMMIT
    assert cast(Any, captured_configs[0]).max_workers == 2
    assert json.loads(captured.out) == {"phase": "completed"}
    assert captured.err == "[grid5000] submitted\n"


def test_print_json_is_sorted_and_keeps_unicode(capsys) -> None:
    grid5000_cli._print_json({"z": 1, "a": "été"})

    assert capsys.readouterr().out == '{"a": "été", "z": 1}\n'


def test_print_progress_flushes_to_stderr(monkeypatch) -> None:
    class Stream:
        def __init__(self) -> None:
            self.writes: list[str] = []
            self.flushes = 0

        def write(self, value: str) -> int:
            self.writes.append(value)
            return len(value)

        def flush(self) -> None:
            self.flushes += 1

    stream = Stream()
    monkeypatch.setattr(grid5000_cli.sys, "stderr", stream)

    grid5000_cli._print_progress("submitted")

    assert stream.writes == ["[grid5000] submitted", "\n"]
    assert stream.flushes == 1


def test_dispatch_routes_status_and_rejects_unknown_commands(monkeypatch) -> None:
    monkeypatch.setattr(grid5000_cli, "_handle_status", lambda _args: 17)

    assert (
        grid5000_cli._dispatch(
            argparse.Namespace(command="status"), task_name="landuse"
        )
        == 17
    )
    with pytest.raises(Grid5000ConfigurationError) as error:
        grid5000_cli._dispatch(
            argparse.Namespace(command="unknown"), task_name="landuse"
        )
    assert str(error.value) == "unknown Grid'5000 command"


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


def test_ablation_builder_forwards_all_runtime_settings_and_report_callback(
    monkeypatch,
) -> None:
    captured: list[dict[str, object]] = []
    report_calls: list[tuple[object, object]] = []

    class FakeController:
        def __init__(self, **kwargs: object) -> None:
            captured.append(kwargs)

    def fake_publish_report(state: object, *, protocol: object) -> None:
        report_calls.append((state, protocol))

    monkeypatch.setattr(grid5000_cli, "AblationStudyController", FakeController)
    monkeypatch.setattr(grid5000_cli, "publish_study_report", fake_publish_report)

    parser = _command_parser("ablations", task_name="place-relevance-v2")
    arguments = parser.parse_args(
        [
            "--source-commit",
            SOURCE_COMMIT,
            "--model-revision",
            MODEL_REVISION,
            "--model-name",
            "custom-model",
            "--site",
            "lille",
            "--walltime-seconds",
            "900",
            "--policy-type",
            "night",
            "--gpu-memory-mb",
            "9000",
            "--max-workers",
            "2",
            "--max-continuations",
            "7",
            "--container-image",
            "registry.example/model@sha256:" + "a" * 64,
            "--container-runtime",
            "docker",
            "--keep-remote",
            "--allow-source-commit-update",
        ]
    )

    grid5000_cli._build_ablation_controller(
        arguments,
        task_name="place-relevance-v2",
    )

    assert len(captured) == 1
    call = captured[0]
    assert call["source_commit"] == SOURCE_COMMIT
    assert call["model_revision"] == MODEL_REVISION
    assert call["model_name_or_path"] == "custom-model"
    assert call["sites"] == ("lille",)
    assert call["gpu_memory_mb"] == 9_000
    assert call["walltime_seconds"] == 900
    assert call["policy_type"] == "night"
    assert call["max_workers"] == 2
    assert call["max_continuations"] == 7
    assert call["container_image"] == "registry.example/model@sha256:" + "a" * 64
    assert call["container_runtime"] == "docker"
    assert call["cleanup"] is False
    assert call["allow_source_commit_update"] is True
    assert call["protocol"] is not None
    assert call["emit"] is grid5000_cli._print_progress

    sentinel = object()
    cast(Any, call["publish_report"])(sentinel)
    assert report_calls == [(sentinel, call["protocol"])]


def test_config_from_state_rehydrates_every_persisted_setting() -> None:
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
        facts={
            "sites": ["lille"],
            "requirements": {"gpu_memory_mb": 9_000},
            "allocation": {"policy_type": "day", "walltime_seconds": 900},
            "requested_policy_type": "day",
            "max_continuations": 7,
            "worker_source_commit": "e" * 40,
            "container_image": "registry.example/model@sha256:" + "b" * 64,
            "container_runtime": "docker",
            "cleanup": False,
        },
    )

    config = grid5000_cli._config_from_state(state.to_dict())

    assert config.identity.run_id == identity.run_id
    assert config.training_config.model_revision == MODEL_REVISION
    assert config.sites == ("lille",)
    assert config.requirements.gpu_memory_mb == 9_000
    assert config.walltime_seconds == 900
    assert config.policy_type == "day"
    assert config.max_continuations == 7
    assert config.worker_source_commit == "e" * 40
    assert config.container_image == "registry.example/model@sha256:" + "b" * 64
    assert config.container_runtime == "docker"
    assert config.cleanup is False


def test_state_identity_validation_reports_stable_errors(monkeypatch) -> None:
    with pytest.raises(Grid5000StateError) as error:
        grid5000_cli._state_identity_and_training_config({"identity": []})
    assert str(error.value) == "autonomous state identity is invalid"

    monkeypatch.setattr(
        grid5000_cli.Grid5000RunIdentity,
        "from_payload",
        lambda _payload: cast(Any, object()),
    )
    with pytest.raises(Grid5000StateError) as error:
        grid5000_cli._state_identity_and_training_config(
            {"identity": {"training_config": []}}
        )
    assert str(error.value) == "autonomous training configuration is invalid"


def test_state_training_constructor_errors_are_wrapped() -> None:
    identity = Grid5000RunIdentity(
        source_commit=SOURCE_COMMIT,
        dataset_revision="d" * 40,
        model_name_or_path="jhu-clsp/mmBERT-small",
        model_revision=MODEL_REVISION,
        training_config={"unexpected": True},
    )

    with pytest.raises(Grid5000StateError) as error:
        grid5000_cli._state_identity_and_training_config(
            {"identity": identity.canonical_payload}
        )
    assert str(error.value) == "autonomous training configuration is invalid"


def test_resume_reports_missing_state_with_the_command_error(
    monkeypatch, capsys
) -> None:
    monkeypatch.setattr(
        grid5000_cli.AutonomousStateStore,
        "load",
        lambda _self, _run_id: None,
    )

    exit_code = grid5000_cli.main(
        ["resume", "--run-id", "a" * 20],
    )

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert captured.err == "error: autonomous run state was not found\n"


def test_resume_preserves_requested_run_id_and_legacy_landuse_default(
    monkeypatch, capsys
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
    state_payload = AutonomousRunState(
        run_id=identity.run_id,
        phase="queued",
        identity=identity.canonical_payload,
        facts={},
    ).to_dict()
    legacy_identity = cast(dict[str, object], state_payload["identity"])
    legacy_identity.pop("task_name")
    loaded_ids: list[str] = []

    class LegacyState:
        identity = legacy_identity

        def to_dict(self) -> dict[str, object]:
            return state_payload

    def load(_self: object, run_id: str) -> LegacyState:
        loaded_ids.append(run_id)
        return LegacyState()

    monkeypatch.setattr(grid5000_cli.AutonomousStateStore, "load", load)

    exit_code = grid5000_cli.main(
        ["resume", "--run-id", identity.run_id],
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert loaded_ids == [identity.run_id]
    assert json.loads(captured.out) == state_payload
    assert captured.err == ""


def test_resume_uses_a_persisted_place_relevance_task_name(monkeypatch, capsys) -> None:
    identity = Grid5000RunIdentity(
        source_commit=SOURCE_COMMIT,
        dataset_revision="d" * 40,
        model_name_or_path="jhu-clsp/mmBERT-small",
        model_revision=MODEL_REVISION,
        task_name="place-relevance-v2",
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
        facts={},
    )
    monkeypatch.setattr(
        grid5000_cli.AutonomousStateStore,
        "load",
        lambda _self, _run_id: state,
    )

    exit_code = grid5000_cli.main(
        ["resume", "--run-id", identity.run_id],
        task_name="place-relevance-v2",
    )

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == state.to_dict()


def test_resume_execution_forwards_progress_emitter(monkeypatch, capsys) -> None:
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
        facts={},
    )

    class FakeController:
        def __init__(self, _config, *, emit) -> None:
            emit("resumed")

        def run(self) -> SimpleNamespace:
            return SimpleNamespace(to_dict=lambda: {"phase": "completed"})

    monkeypatch.setattr(
        grid5000_cli.AutonomousStateStore,
        "load",
        lambda _self, _run_id: state,
    )
    monkeypatch.setattr(grid5000_cli, "_current_source_commit", lambda: SOURCE_COMMIT)
    monkeypatch.setattr(grid5000_cli, "AutonomousRunController", FakeController)

    exit_code = grid5000_cli.main(["resume", "--run-id", identity.run_id, "--execute"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.err == "[grid5000] resumed\n"
    assert json.loads(captured.out) == {"phase": "completed"}
