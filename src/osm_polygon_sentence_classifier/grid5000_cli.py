"""Plan, submit, or autonomously run one guarded classifier training run."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from typing import Literal, cast

from .ablation_study import (
    DEFAULT_MODEL_REVISION,
    AblationStudyController,
    AblationStudyError,
    AblationStudyProtocol,
    place_relevance_v2_ablation_protocol,
    publish_study_report,
)
from .grid5000 import (
    DEFAULT_DAY_WALLTIME_SECONDS,
    MAX_WALLTIME_SECONDS,
    ContainerRuntime,
    Grid5000Allocation,
    Grid5000ConfigurationError,
    Grid5000ExecutionError,
    Grid5000Operator,
    Grid5000Plan,
    Grid5000RunIdentity,
    Grid5000StateError,
)
from .grid5000_autonomous import (
    DEFAULT_AUTONOMOUS_WALLTIME_SECONDS,
    AutonomousRunConfig,
    AutonomousRunController,
    AutonomousRunError,
)
from .grid5000_sites import DEFAULT_SITES, SiteRequirements
from .grid5000_state import AutonomousStateStore
from .training import (
    DEFAULT_MODEL_NAME,
    TrainingConfig,
    TrainingError,
    _training_config_payload,
)
from .training_tasks import (
    TaskName,
    default_max_continuations,
    task_contract,
    training_config_for_task,
)


def _add_plan_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--site", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
    )
    parser.add_argument(
        "--walltime-seconds",
        type=int,
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--policy-type",
        choices=("day", "night"),
        default="night",
        help="Grid'5000 policy window; day allocations are limited to one hour",
    )
    parser.add_argument(
        "--publish",
        action="store_true",
        help="publish the completed model to the project Hugging Face repository",
    )
    parser.add_argument(
        "--sync-trackio",
        action="store_true",
        help="publish static Trackio metric snapshots after checkpoints",
    )
    parser.add_argument(
        "--container-image",
        help="run the worker in this preloaded Docker/Podman image",
    )
    parser.add_argument(
        "--container-runtime",
        choices=("auto", "docker", "podman"),
        default="auto",
        help="container runtime to use when --container-image is supplied",
    )


def _parser(task_name: TaskName) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan, submit, or autonomously run one guarded "
            f"{task_name} Grid'5000 training run"
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser("plan", help="print a side-effect-free plan")
    _add_plan_arguments(plan_parser)
    submit_parser = commands.add_parser("submit", help="plan or explicitly submit")
    _add_plan_arguments(submit_parser)
    submit_parser.add_argument(
        "--execute",
        action="store_true",
        help="cross the explicit gate and run policy, quota, and OAR checks",
    )
    run_parser = commands.add_parser(
        "run",
        help="autonomously probe sites, prepare, submit, monitor, and publish",
    )
    _add_autonomous_arguments(run_parser, task_name=task_name)
    run_parser.add_argument(
        "--execute",
        action="store_true",
        help="cross the explicit gate and perform remote/Hugging Face actions",
    )
    resume_parser = commands.add_parser(
        "resume",
        help="resume a durable autonomous run by its run ID",
    )
    resume_parser.add_argument("--run-id", required=True)
    resume_parser.add_argument(
        "--execute",
        action="store_true",
        help="cross the explicit gate and continue remote actions",
    )
    resume_parser.add_argument(
        "--max-continuations",
        type=int,
        help="extend a failed run beyond its persisted continuation limit",
    )
    resume_parser.add_argument(
        "--policy-type",
        choices=("auto", "day", "night"),
        help="override the persisted policy window for a legacy run",
    )
    status_parser = commands.add_parser(
        "status",
        help="print one local autonomous run state",
    )
    status_parser.add_argument("--run-id", required=True)
    if task_name in {"landuse", "place-relevance-v2"}:
        ablations_parser = commands.add_parser(
            "ablations",
            help=(
                f"plan or autonomously run the reproducible {task_name} ablation study"
            ),
        )
        _add_ablation_arguments(ablations_parser, task_name=task_name)
    return parser


def _add_autonomous_arguments(
    parser: argparse.ArgumentParser,
    *,
    task_name: TaskName,
) -> None:
    parser.add_argument(
        "--site",
        action="append",
        help="Grid'5000 frontend; repeat to restrict discovery (default: all sites)",
    )
    parser.add_argument("--source-commit")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument(
        "--walltime-seconds",
        type=int,
        default=DEFAULT_AUTONOMOUS_WALLTIME_SECONDS,
        help="short one-GPU allocation duration (default: 20 minutes)",
    )
    parser.add_argument(
        "--policy-type",
        choices=("auto", "day", "night"),
        default="auto",
    )
    parser.add_argument("--gpu-memory-mb", type=int, default=8_000)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--max-continuations",
        type=int,
        default=default_max_continuations(task_name),
        help=(
            "maximum bounded same-site checkpoint successors "
            f"(default: {default_max_continuations(task_name)})"
        ),
    )
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--sync-trackio", action="store_true")
    parser.add_argument(
        "--container-image",
        help="run each worker in this preloaded Docker/Podman image",
    )
    parser.add_argument(
        "--container-runtime",
        choices=("auto", "docker", "podman"),
        default="auto",
        help="container runtime to use when --container-image is supplied",
    )
    parser.add_argument(
        "--keep-remote",
        action="store_true",
        help="retain the managed per-run remote data after successful verification",
    )


def _add_ablation_arguments(
    parser: argparse.ArgumentParser,
    *,
    task_name: TaskName,
) -> None:
    parser.add_argument(
        "--source-commit",
        help="clean source revision (default: current clean checkout)",
    )
    parser.add_argument(
        "--allow-source-commit-update",
        action="store_true",
        help="adopt a new source revision only for an incomplete, idle study",
    )
    parser.add_argument(
        "--model-revision",
        default=DEFAULT_MODEL_REVISION,
        help="pinned base-model revision",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--site",
        action="append",
        help="Grid'5000 frontend; repeat to restrict discovery (default: all sites)",
    )
    parser.add_argument(
        "--walltime-seconds",
        type=int,
        default=DEFAULT_AUTONOMOUS_WALLTIME_SECONDS,
    )
    parser.add_argument(
        "--policy-type",
        choices=("auto", "day", "night"),
        default="auto",
    )
    parser.add_argument("--gpu-memory-mb", type=int, default=8_000)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--max-continuations",
        type=int,
        default=(
            default_max_continuations(task_name)
            if task_name == "place-relevance-v2"
            else 6
        ),
    )
    parser.add_argument(
        "--container-image",
        help="run each worker in this preloaded Docker/Podman image",
    )
    parser.add_argument(
        "--container-runtime",
        choices=("auto", "docker", "podman"),
        default="auto",
        help="container runtime to use when --container-image is supplied",
    )
    parser.add_argument(
        "--keep-remote",
        action="store_true",
        help="retain exact managed remote study run roots after completion",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="cross the explicit gate and perform Grid'5000 and Hugging Face actions",
    )


def _build_plan(
    arguments: argparse.Namespace,
    *,
    task_name: TaskName,
) -> Grid5000Plan:
    config = training_config_for_task(
        task_name,
        model_name_or_path=arguments.model_name,
        model_revision=arguments.model_revision,
        max_steps=arguments.max_steps,
        publish_to_hub=arguments.publish,
        sync_trackio=arguments.sync_trackio,
    )
    identity = Grid5000RunIdentity(
        source_commit=arguments.source_commit,
        dataset_revision=task_contract(task_name).provenance.repository_revision,
        model_name_or_path=config.model_name_or_path,
        model_revision=arguments.model_revision,
        task_name=task_name,
        training_config=_training_config_payload(config),
    )
    walltime_seconds = arguments.walltime_seconds
    if walltime_seconds is None:
        walltime_seconds = (
            DEFAULT_DAY_WALLTIME_SECONDS
            if arguments.policy_type == "day"
            else MAX_WALLTIME_SECONDS
        )
    return Grid5000Plan(
        identity=identity,
        allocation=Grid5000Allocation(
            site=arguments.site,
            walltime_seconds=walltime_seconds,
            policy_type=arguments.policy_type,
        ),
        container_image=arguments.container_image,
        container_runtime=arguments.container_runtime,
    )


def _current_source_commit() -> str:
    commit = _run_git_command(
        ("git", "rev-parse", "HEAD"),
        failure_message="current source commit could not be resolved",
    )
    source_commit = commit.stdout.strip()
    _validate_source_commit_output(commit.returncode, source_commit)
    status = _run_git_command(
        ("git", "status", "--porcelain"),
        failure_message="current checkout cleanliness could not be verified",
    )
    _validate_clean_checkout(status.returncode, status.stdout)
    return source_commit


def _run_git_command(
    command: tuple[str, ...], *, failure_message: str
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise Grid5000ConfigurationError(failure_message) from error


def _validate_source_commit_output(returncode: int, source_commit: str) -> None:
    if returncode != 0 or len(source_commit) != 40:
        raise Grid5000ConfigurationError(
            "current checkout does not have one pinned source commit"
        )


def _validate_clean_checkout(returncode: int, status_output: str) -> None:
    if returncode != 0 or status_output.strip():
        raise Grid5000ConfigurationError(
            "current checkout must be clean when source commit is implicit"
        )


def _build_autonomous_config(
    arguments: argparse.Namespace,
    *,
    task_name: TaskName,
) -> AutonomousRunConfig:
    source_commit = arguments.source_commit or _current_source_commit()
    training_config = training_config_for_task(
        task_name,
        model_name_or_path=arguments.model_name,
        model_revision=arguments.model_revision,
        max_steps=arguments.max_steps,
        publish_to_hub=arguments.publish,
        sync_trackio=arguments.sync_trackio,
    )
    identity = Grid5000RunIdentity(
        source_commit=source_commit,
        dataset_revision=task_contract(task_name).provenance.repository_revision,
        model_name_or_path=training_config.model_name_or_path,
        model_revision=arguments.model_revision,
        task_name=task_name,
        training_config=_training_config_payload(training_config),
    )
    return AutonomousRunConfig(
        identity=identity,
        training_config=training_config,
        sites=tuple(arguments.site or DEFAULT_SITES),
        requirements=SiteRequirements(gpu_memory_mb=arguments.gpu_memory_mb),
        walltime_seconds=arguments.walltime_seconds,
        policy_type=arguments.policy_type,
        max_workers=arguments.max_workers,
        max_continuations=arguments.max_continuations,
        container_image=arguments.container_image,
        container_runtime=arguments.container_runtime,
        cleanup=not arguments.keep_remote,
    )


def _build_ablation_controller(
    arguments: argparse.Namespace,
    *,
    task_name: TaskName,
) -> AblationStudyController:
    source_commit = arguments.source_commit or _current_source_commit()
    protocol: AblationStudyProtocol | None = (
        place_relevance_v2_ablation_protocol()
        if task_name == "place-relevance-v2"
        else None
    )
    return AblationStudyController(
        source_commit=source_commit,
        model_revision=arguments.model_revision,
        model_name_or_path=arguments.model_name,
        sites=tuple(arguments.site or DEFAULT_SITES),
        gpu_memory_mb=arguments.gpu_memory_mb,
        walltime_seconds=arguments.walltime_seconds,
        policy_type=arguments.policy_type,
        max_workers=arguments.max_workers,
        max_continuations=arguments.max_continuations,
        container_image=arguments.container_image,
        container_runtime=arguments.container_runtime,
        cleanup=not arguments.keep_remote,
        allow_source_commit_update=arguments.allow_source_commit_update,
        protocol=protocol,
        publish_report=(lambda state: publish_study_report(state, protocol=protocol)),
        emit=_print_progress,
    )


def _autonomous_plan_payload(config: AutonomousRunConfig) -> dict[str, object]:
    return {
        "run_id": config.identity.run_id,
        "identity": config.identity.canonical_payload,
        "sites": list(config.sites),
        "walltime_seconds": config.walltime_seconds,
        "policy_type": config.policy_type,
        "gpu_memory_mb": config.requirements.gpu_memory_mb,
        "publish": config.training_config.publish_to_hub,
        "sync_trackio": config.training_config.sync_trackio,
        "max_continuations": config.max_continuations,
        "container_image": config.container_image,
        "container_runtime": config.container_runtime,
        "cleanup": config.cleanup,
    }


def _state_facts(state_payload: Mapping[str, object]) -> Mapping[str, object]:
    facts = state_payload.get("facts")
    if not isinstance(facts, Mapping):
        return {}
    return cast(Mapping[str, object], facts)


def _state_allocation_settings(
    facts: Mapping[str, object],
    *,
    policy_type_override: str | None = None,
) -> tuple[object, object, object]:
    allocation = facts.get("allocation")
    if not isinstance(allocation, Mapping):
        policy = policy_type_override or facts.get("requested_policy_type", "auto")
        return policy, DEFAULT_AUTONOMOUS_WALLTIME_SECONDS, True
    requested_policy = facts.get("requested_policy_type")
    if requested_policy not in {"auto", "day", "night"}:
        requested_policy = allocation.get("policy_type", "auto")
    return (
        policy_type_override or requested_policy,
        allocation.get("walltime_seconds", DEFAULT_AUTONOMOUS_WALLTIME_SECONDS),
        facts.get("cleanup", True),
    )


def _state_continuation_settings(
    state_payload: Mapping[str, object],
    facts: Mapping[str, object],
    *,
    max_continuations_override: int | None,
    worker_source_commit_override: str | None,
) -> tuple[int, str | None]:
    persisted_limit = _persisted_continuation_limit(facts)
    max_continuations = _continuation_limit_with_override(
        state_payload,
        persisted_limit,
        max_continuations_override,
    )
    worker_source_commit = _persisted_worker_source_commit(facts)
    return max_continuations, worker_source_commit_override or worker_source_commit


def _persisted_continuation_limit(facts: Mapping[str, object]) -> int:
    max_continuations = facts.get("max_continuations", 3)
    if (
        isinstance(max_continuations, bool)
        or not isinstance(max_continuations, int)
        or max_continuations <= 0
    ):
        raise Grid5000StateError("autonomous continuation limit is invalid")
    return max_continuations


def _continuation_limit_with_override(
    state_payload: Mapping[str, object],
    persisted_limit: int,
    override: int | None,
) -> int:
    if override is None:
        return persisted_limit
    _validate_continuation_override(override, persisted_limit)
    if state_payload.get("phase") != "failed":
        raise Grid5000StateError("--max-continuations can only extend a failed run")
    return override


def _validate_continuation_override(override: int, persisted_limit: int) -> None:
    if (
        isinstance(override, bool)
        or not isinstance(override, int)
        or override <= persisted_limit
    ):
        raise Grid5000StateError(
            "--max-continuations must be greater than the persisted limit"
        )


def _persisted_worker_source_commit(facts: Mapping[str, object]) -> str | None:
    worker_source_commit = facts.get("worker_source_commit")
    if worker_source_commit is not None and not isinstance(worker_source_commit, str):
        raise Grid5000StateError("autonomous worker source commit is invalid")
    return worker_source_commit


def _state_container_settings(
    facts: Mapping[str, object],
) -> tuple[str | None, ContainerRuntime]:
    image = facts.get("container_image")
    if image is not None and not isinstance(image, str):
        raise Grid5000StateError("autonomous container image is invalid")
    normalized_runtime = _normalize_container_runtime(
        facts.get("container_runtime", "auto")
    )
    try:
        from .grid5000 import _validate_container_settings

        _validate_container_settings(image, normalized_runtime)
    except Grid5000ConfigurationError as error:
        raise Grid5000StateError("autonomous container settings are invalid") from error
    return image, normalized_runtime


def _normalize_container_runtime(value: object) -> ContainerRuntime:
    if value not in {"auto", "docker", "podman"}:
        raise Grid5000StateError("autonomous container runtime is invalid")
    return cast(ContainerRuntime, value)


def _state_sites(facts: Mapping[str, object]) -> tuple[str, ...]:
    sites = facts.get("sites")
    if not isinstance(sites, Sequence) or isinstance(sites, (str, bytes)):
        return DEFAULT_SITES
    return _non_empty_sites(sites)


def _non_empty_sites(sites: Sequence[object]) -> tuple[str, ...]:
    normalized = tuple(site for site in sites if isinstance(site, str))
    return normalized or DEFAULT_SITES


def _state_gpu_memory(facts: Mapping[str, object]) -> int:
    requirements = facts.get("requirements")
    if not isinstance(requirements, Mapping):
        return 8_000
    value = requirements.get("gpu_memory_mb")
    if isinstance(value, bool) or not isinstance(value, int):
        return 8_000
    return value


def _config_from_state(
    state_payload: Mapping[str, object],
    *,
    max_continuations_override: int | None = None,
    worker_source_commit_override: str | None = None,
    policy_type_override: str | None = None,
) -> AutonomousRunConfig:
    identity, training_config = _state_identity_and_training_config(state_payload)
    facts = _state_facts(state_payload)
    policy, walltime, cleanup = _state_allocation_settings(
        facts,
        policy_type_override=policy_type_override,
    )
    max_continuations, worker_source_commit = _state_continuation_settings(
        state_payload,
        facts,
        max_continuations_override=max_continuations_override,
        worker_source_commit_override=worker_source_commit_override,
    )
    container_image, container_runtime = _state_container_settings(facts)
    return AutonomousRunConfig(
        identity=identity,
        training_config=training_config,
        sites=_state_sites(facts),
        requirements=SiteRequirements(gpu_memory_mb=_state_gpu_memory(facts)),
        walltime_seconds=_state_walltime(walltime),
        policy_type=_state_policy(policy),
        max_continuations=max_continuations,
        worker_source_commit=worker_source_commit,
        container_image=container_image,
        container_runtime=container_runtime,
        cleanup=_state_cleanup(cleanup),
    )


def _state_identity_and_training_config(
    state_payload: Mapping[str, object],
) -> tuple[Grid5000RunIdentity, TrainingConfig]:
    identity_payload = state_payload.get("identity")
    if not isinstance(identity_payload, Mapping):
        raise Grid5000StateError("autonomous state identity is invalid")
    identity_payload_typed = cast(
        Mapping[str, object], identity_payload
    )  # pragma: no mutate
    identity = Grid5000RunIdentity.from_payload(identity_payload_typed)
    training_payload = identity_payload.get("training_config")
    if not isinstance(training_payload, Mapping):
        raise Grid5000StateError("autonomous training configuration is invalid")
    try:
        training_config = TrainingConfig(**dict(training_payload))
    except (TypeError, TrainingError) as error:
        raise Grid5000StateError(
            "autonomous training configuration is invalid"
        ) from error
    return identity, training_config


def _state_walltime(value: object) -> int:
    return value if isinstance(value, int) else DEFAULT_AUTONOMOUS_WALLTIME_SECONDS


def _state_policy(value: object) -> Literal["auto", "day", "night"]:
    if value == "day":
        return "day"
    if value == "night":
        return "night"
    return "auto"


def _state_cleanup(value: object) -> bool:
    return value if isinstance(value, bool) else True


def _print_json(payload: dict[str, object]) -> None:
    # ``None`` is equivalent to ``False`` for json.dumps; keep this explicit
    # serialization contract out of mutation generation while testing its output.
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))  # pragma: no mutate


def _print_progress(message: str) -> None:
    """Print human-readable progress without corrupting JSON stdout."""

    print(f"[grid5000] {message}", file=sys.stderr, flush=True)


def _handle_run(arguments: argparse.Namespace, *, task_name: TaskName) -> int:
    autonomous = _build_autonomous_config(arguments, task_name=task_name)
    if not arguments.execute:
        _print_json(_autonomous_plan_payload(autonomous))
        return 0
    result = AutonomousRunController(autonomous, emit=_print_progress).run()
    _print_json(result.to_dict())
    return 0


def _handle_ablations(
    arguments: argparse.Namespace,
    *,
    task_name: TaskName,
) -> int:
    controller = _build_ablation_controller(arguments, task_name=task_name)
    result = controller.run() if arguments.execute else controller.plan()
    _print_json(result)
    return 0


def _handle_status(arguments: argparse.Namespace) -> int:
    state = AutonomousStateStore().load(arguments.run_id)
    if state is None:
        raise Grid5000StateError("autonomous run state was not found")
    _print_json(state.to_dict())
    return 0


def _handle_resume(
    arguments: argparse.Namespace,
    *,
    task_name: TaskName,
) -> int:
    state = AutonomousStateStore().load(arguments.run_id)
    if state is None:
        raise Grid5000StateError("autonomous run state was not found")
    persisted_task = state.identity.get("task_name", "landuse")
    if persisted_task != task_name:
        raise Grid5000StateError(f"run task does not match {task_name!r} command")
    autonomous = _config_from_state(
        state.to_dict(),
        max_continuations_override=arguments.max_continuations,
        policy_type_override=arguments.policy_type,
        worker_source_commit_override=(
            _current_source_commit() if arguments.execute else None
        ),
    )
    if not arguments.execute:
        _print_json(state.to_dict())
        return 0
    result = AutonomousRunController(autonomous, emit=_print_progress).run()
    _print_json(result.to_dict())
    return 0


def _handle_plan_or_submit(
    arguments: argparse.Namespace,
    *,
    task_name: TaskName,
) -> int:
    plan = _build_plan(arguments, task_name=task_name)
    if arguments.command == "plan":
        _print_json(plan.to_dict())
        return 0
    submission = Grid5000Operator(plan).submit(execute=arguments.execute)
    _print_json(
        {
            "executed": submission.executed,
            "job_id": submission.job_id,
            "plan": submission.plan.to_dict(),
        }
    )
    return 0


def _dispatch(arguments: argparse.Namespace, *, task_name: TaskName) -> int:
    handlers = {
        "run": lambda args: _handle_run(args, task_name=task_name),
        "ablations": lambda args: _handle_ablations(args, task_name=task_name),
        "status": _handle_status,
        "resume": lambda args: _handle_resume(args, task_name=task_name),
        "plan": lambda args: _handle_plan_or_submit(args, task_name=task_name),
        "submit": lambda args: _handle_plan_or_submit(args, task_name=task_name),
    }
    handler = handlers.get(arguments.command)
    if handler is None:
        raise Grid5000ConfigurationError("unknown Grid'5000 command")
    return handler(arguments)


def main(
    argv: Sequence[str] | None = None,
    *,
    task_name: TaskName = "landuse",
) -> int:
    """Run a side-effect-free plan unless an explicit execute gate is supplied."""

    parser = _parser(task_name)
    arguments = parser.parse_args(argv)
    try:
        return _dispatch(arguments, task_name=task_name)
    except (
        Grid5000ConfigurationError,
        Grid5000ExecutionError,
        Grid5000StateError,
        AutonomousRunError,
        AblationStudyError,
        TrainingError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
