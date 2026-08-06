"""Plan, submit, or autonomously run one guarded landuse training run."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import fields
from pathlib import Path
from typing import Literal, cast

from .ablation_study import (
    DEFAULT_MODEL_REVISION,
    AblationStudyController,
    AblationStudyError,
    publish_study_report,
)
from .dataset_contract import LANDUSE_DATASET_CONTRACT
from .grid5000 import (
    DEFAULT_DAY_WALLTIME_SECONDS,
    MAX_WALLTIME_SECONDS,
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
from .training import DEFAULT_MODEL_NAME, TrainingConfig, TrainingError


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
        default=None,
        type=int,
    )
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan, submit, or autonomously run one guarded landuse Grid'5000 training run"
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
    _add_autonomous_arguments(run_parser)
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
        default=None,
        help="extend a failed run beyond its persisted continuation limit",
    )
    status_parser = commands.add_parser(
        "status",
        help="print one local autonomous run state",
    )
    status_parser.add_argument("--run-id", required=True)
    ablations_parser = commands.add_parser(
        "ablations",
        help="plan or autonomously run the reproducible landuse ablation study",
    )
    _add_ablation_arguments(ablations_parser)
    return parser


def _add_autonomous_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--site",
        action="append",
        default=None,
        help="Grid'5000 frontend; repeat to restrict discovery (default: all sites)",
    )
    parser.add_argument("--source-commit", default=None)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
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
        default=3,
        help="maximum bounded same-site checkpoint successors (default: 3)",
    )
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--sync-trackio", action="store_true")
    parser.add_argument(
        "--keep-remote",
        action="store_true",
        help="retain the managed per-run remote data after successful verification",
    )


def _add_ablation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--source-commit",
        default=None,
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
        default=None,
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
    parser.add_argument("--max-continuations", type=int, default=6)
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


def _training_config_payload(config: TrainingConfig) -> dict[str, object]:
    payload: dict[str, object] = {}
    for item in fields(config):
        value = getattr(config, item.name)
        if (
            item.name
            in {
                "trainable_layers",
                "class_weight_mode",
                "tracking_project",
                "artifact_namespace",
            }
            and value is None
        ):
            continue
        payload[item.name] = str(value) if isinstance(value, Path) else value
    return payload


def _build_plan(arguments: argparse.Namespace) -> Grid5000Plan:
    config = TrainingConfig(
        model_name_or_path=arguments.model_name,
        model_revision=arguments.model_revision,
        publish_to_hub=arguments.publish,
        sync_trackio=arguments.sync_trackio,
    )
    identity = Grid5000RunIdentity(
        source_commit=arguments.source_commit,
        dataset_revision=LANDUSE_DATASET_CONTRACT.provenance.repository_revision,
        model_name_or_path=config.model_name_or_path,
        model_revision=arguments.model_revision,
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
    )


def _current_source_commit() -> str:
    try:
        result = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise Grid5000ConfigurationError(
            "current source commit could not be resolved"
        ) from error
    source_commit = result.stdout.strip()
    if result.returncode != 0 or len(source_commit) != 40:
        raise Grid5000ConfigurationError(
            "current checkout does not have one pinned source commit"
        )
    try:
        status = subprocess.run(
            ("git", "status", "--porcelain"),
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise Grid5000ConfigurationError(
            "current checkout cleanliness could not be verified"
        ) from error
    if status.returncode != 0 or status.stdout.strip():
        raise Grid5000ConfigurationError(
            "current checkout must be clean when source commit is implicit"
        )
    return source_commit


def _build_autonomous_config(arguments: argparse.Namespace) -> AutonomousRunConfig:
    source_commit = arguments.source_commit or _current_source_commit()
    training_config = TrainingConfig(
        model_name_or_path=arguments.model_name,
        model_revision=arguments.model_revision,
        publish_to_hub=arguments.publish,
        sync_trackio=arguments.sync_trackio,
    )
    identity = Grid5000RunIdentity(
        source_commit=source_commit,
        dataset_revision=LANDUSE_DATASET_CONTRACT.provenance.repository_revision,
        model_name_or_path=training_config.model_name_or_path,
        model_revision=arguments.model_revision,
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
        cleanup=not arguments.keep_remote,
    )


def _build_ablation_controller(
    arguments: argparse.Namespace,
) -> AblationStudyController:
    source_commit = arguments.source_commit or _current_source_commit()
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
        cleanup=not arguments.keep_remote,
        allow_source_commit_update=arguments.allow_source_commit_update,
        publish_report=publish_study_report,
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
        "cleanup": config.cleanup,
    }


def _state_facts(state_payload: Mapping[str, object]) -> Mapping[str, object]:
    facts = state_payload.get("facts", {})
    if not isinstance(facts, Mapping):
        return {}
    return cast(Mapping[str, object], facts)


def _state_allocation_settings(
    facts: Mapping[str, object],
) -> tuple[object, object, object]:
    allocation = facts.get("allocation", {})
    if not isinstance(allocation, Mapping):
        return "auto", DEFAULT_AUTONOMOUS_WALLTIME_SECONDS, True
    return (
        allocation.get("policy_type", "auto"),
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
    max_continuations = facts.get("max_continuations", 3)
    if (
        isinstance(max_continuations, bool)
        or not isinstance(max_continuations, int)
        or max_continuations <= 0
    ):
        raise Grid5000StateError("autonomous continuation limit is invalid")
    if max_continuations_override is not None:
        if (
            isinstance(max_continuations_override, bool)
            or not isinstance(max_continuations_override, int)
            or max_continuations_override <= max_continuations
        ):
            raise Grid5000StateError(
                "--max-continuations must be greater than the persisted limit"
            )
        if state_payload.get("phase") != "failed":
            raise Grid5000StateError("--max-continuations can only extend a failed run")
        max_continuations = max_continuations_override
    worker_source_commit = facts.get("worker_source_commit")
    if worker_source_commit is not None and not isinstance(worker_source_commit, str):
        raise Grid5000StateError("autonomous worker source commit is invalid")
    return max_continuations, worker_source_commit_override or worker_source_commit


def _state_sites(facts: Mapping[str, object]) -> tuple[str, ...]:
    sites = facts.get("sites", DEFAULT_SITES)
    if not isinstance(sites, Sequence) or isinstance(sites, (str, bytes)):
        return DEFAULT_SITES
    normalized = tuple(site for site in sites if isinstance(site, str))
    return normalized or DEFAULT_SITES


def _state_gpu_memory(facts: Mapping[str, object]) -> int:
    requirements = facts.get("requirements", {})
    if not isinstance(requirements, Mapping):
        return 8_000
    value = requirements.get("gpu_memory_mb", 8_000)
    if isinstance(value, bool) or not isinstance(value, int):
        return 8_000
    return value


def _config_from_state(
    state_payload: Mapping[str, object],
    *,
    max_continuations_override: int | None = None,
    worker_source_commit_override: str | None = None,
) -> AutonomousRunConfig:
    identity_payload = state_payload.get("identity")
    if not isinstance(identity_payload, Mapping):
        raise Grid5000StateError("autonomous state identity is invalid")
    identity = Grid5000RunIdentity.from_payload(
        cast(Mapping[str, object], identity_payload)
    )
    training_payload = identity_payload.get("training_config")
    if not isinstance(training_payload, Mapping):
        raise Grid5000StateError("autonomous training configuration is invalid")
    try:
        training_config = TrainingConfig(**dict(training_payload))
    except (TypeError, TrainingError) as error:
        raise Grid5000StateError(
            "autonomous training configuration is invalid"
        ) from error
    facts = _state_facts(state_payload)
    policy, walltime, cleanup = _state_allocation_settings(facts)
    max_continuations, worker_source_commit = _state_continuation_settings(
        state_payload,
        facts,
        max_continuations_override=max_continuations_override,
        worker_source_commit_override=worker_source_commit_override,
    )
    return AutonomousRunConfig(
        identity=identity,
        training_config=training_config,
        sites=_state_sites(facts),
        requirements=SiteRequirements(gpu_memory_mb=_state_gpu_memory(facts)),
        walltime_seconds=(
            walltime
            if isinstance(walltime, int)
            else DEFAULT_AUTONOMOUS_WALLTIME_SECONDS
        ),
        policy_type=(
            cast(Literal["auto", "day", "night"], policy)
            if policy in {"auto", "day", "night"}
            else "auto"
        ),
        max_continuations=max_continuations,
        worker_source_commit=worker_source_commit,
        cleanup=cleanup if isinstance(cleanup, bool) else True,
    )


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _print_progress(message: str) -> None:
    """Print human-readable progress without corrupting JSON stdout."""

    print(f"[grid5000] {message}", file=sys.stderr, flush=True)


def _handle_run(arguments: argparse.Namespace) -> int:
    autonomous = _build_autonomous_config(arguments)
    if not arguments.execute:
        _print_json(_autonomous_plan_payload(autonomous))
        return 0
    result = AutonomousRunController(autonomous, emit=_print_progress).run()
    _print_json(result.to_dict())
    return 0


def _handle_ablations(arguments: argparse.Namespace) -> int:
    controller = _build_ablation_controller(arguments)
    result = controller.run() if arguments.execute else controller.plan()
    _print_json(result)
    return 0


def _handle_status(arguments: argparse.Namespace) -> int:
    state = AutonomousStateStore().load(arguments.run_id)
    if state is None:
        raise Grid5000StateError("autonomous run state was not found")
    _print_json(state.to_dict())
    return 0


def _handle_resume(arguments: argparse.Namespace) -> int:
    state = AutonomousStateStore().load(arguments.run_id)
    if state is None:
        raise Grid5000StateError("autonomous run state was not found")
    autonomous = _config_from_state(
        state.to_dict(),
        max_continuations_override=arguments.max_continuations,
        worker_source_commit_override=(
            _current_source_commit()
            if arguments.execute and arguments.max_continuations is not None
            else None
        ),
    )
    if not arguments.execute:
        _print_json(state.to_dict())
        return 0
    result = AutonomousRunController(autonomous, emit=_print_progress).run()
    _print_json(result.to_dict())
    return 0


def _handle_plan_or_submit(arguments: argparse.Namespace) -> int:
    plan = _build_plan(arguments)
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


def _dispatch(arguments: argparse.Namespace) -> int:
    handlers = {
        "run": _handle_run,
        "ablations": _handle_ablations,
        "status": _handle_status,
        "resume": _handle_resume,
        "plan": _handle_plan_or_submit,
        "submit": _handle_plan_or_submit,
    }
    handler = handlers.get(arguments.command)
    if handler is None:
        raise Grid5000ConfigurationError("unknown Grid'5000 command")
    return handler(arguments)


def main(argv: Sequence[str] | None = None) -> int:
    """Run a side-effect-free plan unless an explicit execute gate is supplied."""

    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        return _dispatch(arguments)
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
