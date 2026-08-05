"""Plan or explicitly submit one guarded Grid'5000 landuse training run."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import fields
from pathlib import Path

from .dataset_contract import LANDUSE_DATASET_CONTRACT
from .grid5000 import (
    MAX_WALLTIME_SECONDS,
    Grid5000Allocation,
    Grid5000ConfigurationError,
    Grid5000ExecutionError,
    Grid5000Operator,
    Grid5000Plan,
    Grid5000RunIdentity,
    Grid5000StateError,
)
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
        default=MAX_WALLTIME_SECONDS,
        type=int,
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
        description="Plan or submit one guarded landuse Grid'5000 training run"
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
    return parser


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
    return Grid5000Plan(
        identity=identity,
        allocation=Grid5000Allocation(
            site=arguments.site,
            walltime_seconds=arguments.walltime_seconds,
        ),
    )


def _print_json(payload: dict[str, object]) -> None:
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    """Print a plan unless ``submit --execute`` is explicitly requested."""

    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
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
        TrainingError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
