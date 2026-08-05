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
        help="synchronize completed metrics to the static Trackio Space",
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
    status_parser = commands.add_parser(
        "status",
        help="print one local autonomous run state",
    )
    status_parser.add_argument("--run-id", required=True)
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
        default=30 * 60,
        help="short one-GPU allocation duration (default: 30 minutes)",
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


def _training_config_payload(config: TrainingConfig) -> dict[str, object]:
    payload: dict[str, object] = {}
    for item in fields(config):
        value = getattr(config, item.name)
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


def _config_from_state(state_payload: Mapping[str, object]) -> AutonomousRunConfig:
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
    facts = state_payload.get("facts", {})
    allocation = facts.get("allocation", {}) if isinstance(facts, Mapping) else {}
    policy = (
        allocation.get("policy_type", "auto")
        if isinstance(allocation, Mapping)
        else "auto"
    )
    walltime = (
        allocation.get("walltime_seconds", 30 * 60)
        if isinstance(allocation, Mapping)
        else 30 * 60
    )
    cleanup = facts.get("cleanup", True) if isinstance(facts, Mapping) else True
    max_continuations = (
        facts.get("max_continuations", 3) if isinstance(facts, Mapping) else 3
    )
    sites = (
        facts.get("sites", DEFAULT_SITES)
        if isinstance(facts, Mapping)
        else DEFAULT_SITES
    )
    requirements_payload = (
        facts.get("requirements", {}) if isinstance(facts, Mapping) else {}
    )
    gpu_memory = (
        requirements_payload.get("gpu_memory_mb", 8_000)
        if isinstance(requirements_payload, Mapping)
        else 8_000
    )
    if not isinstance(sites, Sequence) or isinstance(sites, (str, bytes)):
        sites = DEFAULT_SITES
    normalized_sites = tuple(site for site in sites if isinstance(site, str))
    if not normalized_sites:
        normalized_sites = DEFAULT_SITES
    return AutonomousRunConfig(
        identity=identity,
        training_config=training_config,
        sites=normalized_sites,
        requirements=SiteRequirements(
            gpu_memory_mb=gpu_memory if isinstance(gpu_memory, int) else 8_000
        ),
        walltime_seconds=walltime if isinstance(walltime, int) else 30 * 60,
        policy_type=(
            cast(Literal["auto", "day", "night"], policy)
            if policy in {"auto", "day", "night"}
            else "auto"
        ),
        max_continuations=(
            max_continuations if isinstance(max_continuations, int) else 3
        ),
        cleanup=cleanup if isinstance(cleanup, bool) else True,
    )


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    """Run a side-effect-free plan unless an explicit execute gate is supplied."""

    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "run":
            autonomous = _build_autonomous_config(arguments)
            if not arguments.execute:
                _print_json(_autonomous_plan_payload(autonomous))
                return 0
            result = AutonomousRunController(autonomous).run()
            _print_json(result.to_dict())
            return 0
        if arguments.command == "status":
            state = AutonomousStateStore().load(arguments.run_id)
            if state is None:
                raise Grid5000StateError("autonomous run state was not found")
            _print_json(state.to_dict())
            return 0
        if arguments.command == "resume":
            state = AutonomousStateStore().load(arguments.run_id)
            if state is None:
                raise Grid5000StateError("autonomous run state was not found")
            autonomous = _config_from_state(state.to_dict())
            if not arguments.execute:
                _print_json(state.to_dict())
                return 0
            result = AutonomousRunController(autonomous).run()
            _print_json(result.to_dict())
            return 0
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
    except (
        Grid5000ConfigurationError,
        Grid5000ExecutionError,
        Grid5000StateError,
        AutonomousRunError,
        TrainingError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
