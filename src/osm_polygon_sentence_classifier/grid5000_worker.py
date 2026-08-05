"""Strict compute-node preflight and training boundary for Grid'5000."""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from .config import ConfigurationError, ProjectConfig
from .dataset_contract import LANDUSE_DATASET_CONTRACT
from .grid5000 import (
    COMMAND_TIMEOUT_SECONDS,
    REMOTE_DATA_SUBDIRECTORY,
    CommandResult,
    CommandRunner,
    Grid5000ConfigurationError,
    Grid5000RunIdentity,
    SubprocessCommandRunner,
)
from .training import (
    TrainingConfig,
    TrainingError,
    TrainingResult,
    train_landuse_classifier,
)

_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_JOB_ID_PATTERN = re.compile(r"[1-9][0-9]*")


class WorkerError(RuntimeError):
    """Raised when a Grid'5000 compute-node contract is not satisfied."""


@dataclass(frozen=True, slots=True)
class WorkerFacts:
    """Validated facts passed to the training boundary."""

    job_id: int
    source_commit: str
    cuda_device_name: str


CudaProbe = Callable[[], tuple[bool, int, str]]


def _default_cuda_probe() -> tuple[bool, int, str]:
    try:
        torch = cast(Any, import_module("torch"))
    except ModuleNotFoundError as error:
        raise WorkerError(
            "Grid'5000 worker requires the torch training dependency"
        ) from error
    available = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count())
    device_name = ""
    if available and count == 1:
        device_name = str(torch.cuda.get_device_name(0))
    return available, count, device_name


def _failed_command(result: CommandResult, label: str) -> None:
    if result.returncode != 0:
        raise WorkerError(f"{label} command failed with exit code {result.returncode}")


def _validate_checkout(
    checkout_path: Path,
    expected_source_commit: str,
    runner: CommandRunner,
) -> None:
    try:
        head = runner(
            ("git", "-C", str(checkout_path), "rev-parse", "HEAD"),
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
        _failed_command(head, "git revision")
        if head.stdout.strip() != expected_source_commit:
            raise WorkerError("worker checkout is not at the expected source commit")
        status = runner(
            ("git", "-C", str(checkout_path), "status", "--porcelain"),
            timeout=COMMAND_TIMEOUT_SECONDS,
        )
        _failed_command(status, "git status")
        if status.stdout.strip():
            raise WorkerError("worker checkout is not clean")
    except WorkerError:
        raise
    except Exception as error:
        raise WorkerError("worker checkout validation could not complete") from error


def _validate_cuda(probe: CudaProbe) -> str:
    try:
        cuda_available, device_count, device_name = probe()
    except WorkerError:
        raise
    except Exception as error:
        raise WorkerError("CUDA preflight could not complete") from error
    if not cuda_available:
        raise WorkerError("CUDA is not available on the worker")
    if device_count != 1:
        raise WorkerError("worker must expose exactly one CUDA GPU")
    if not device_name.strip():
        raise WorkerError("CUDA device name is missing")
    return device_name


def validate_compute_node(
    *,
    expected_source_commit: str,
    checkout: Path,
    environ: Mapping[str, str] | None = None,
    platform_name: str | None = None,
    git_runner: CommandRunner | None = None,
    cuda_probe: CudaProbe | None = None,
) -> WorkerFacts:
    """Validate Linux, OAR identity, checkout exactness, and one CUDA GPU."""

    effective_platform = platform_name or sys.platform
    if effective_platform != "linux":
        raise WorkerError("Grid'5000 worker requires a Linux compute node")
    if _REVISION_PATTERN.fullmatch(expected_source_commit) is None:
        raise WorkerError("expected source commit is not a pinned revision")
    environment = os.environ if environ is None else environ
    raw_job_id = environment.get("OAR_JOB_ID")
    if raw_job_id is None or _JOB_ID_PATTERN.fullmatch(raw_job_id) is None:
        raise WorkerError("OAR_JOB_ID must be one positive integer")
    checkout_path = Path(checkout)
    if not checkout_path.is_absolute() or checkout_path.is_symlink():
        raise WorkerError("worker checkout must be an absolute non-symlink path")

    runner = git_runner or SubprocessCommandRunner()
    _validate_checkout(checkout_path, expected_source_commit, runner)
    device_name = _validate_cuda(cuda_probe or _default_cuda_probe)

    return WorkerFacts(
        job_id=int(raw_job_id),
        source_commit=expected_source_commit,
        cuda_device_name=device_name,
    )


def run_landuse_training_worker(
    identity: Grid5000RunIdentity,
    *,
    checkout: Path,
    training_config: TrainingConfig | None = None,
    remote_data_root: Path | None = None,
    environ: Mapping[str, str] | None = None,
    platform_name: str | None = None,
    git_runner: CommandRunner | None = None,
    cuda_probe: CudaProbe | None = None,
    train: Callable[..., TrainingResult] = train_landuse_classifier,
) -> TrainingResult:
    """Preflight one compute node, then call the existing training boundary."""

    if (
        identity.dataset_revision
        != LANDUSE_DATASET_CONTRACT.provenance.repository_revision
    ):
        raise WorkerError("worker dataset revision does not match the pinned contract")
    config_values: dict[str, Any] = dict(identity.training_config)
    try:
        effective_config = training_config or TrainingConfig(**config_values)
    except (TypeError, TrainingError) as error:
        raise WorkerError("worker training configuration is invalid") from error
    if effective_config.model_name_or_path != identity.model_name_or_path:
        raise WorkerError("worker model identity does not match training configuration")
    if effective_config.model_revision != identity.model_revision:
        raise WorkerError("worker model revision does not match training configuration")
    validate_compute_node(
        expected_source_commit=identity.source_commit,
        checkout=checkout,
        environ=environ,
        platform_name=platform_name,
        git_runner=git_runner,
        cuda_probe=cuda_probe,
    )
    data_root = remote_data_root or Path.home() / REMOTE_DATA_SUBDIRECTORY
    try:
        project_config = ProjectConfig.for_remote_root(data_root)
    except (ConfigurationError, Grid5000ConfigurationError) as error:
        raise WorkerError("remote worker data root is unsafe") from error
    return train(config=effective_config, project_config=project_config)


def _identity_from_arguments(
    *,
    run_id: str,
    source_commit: str,
    dataset_revision: str,
    model_revision: str,
    training_config_json: str,
) -> Grid5000RunIdentity:
    try:
        training_config = json.loads(training_config_json)
    except json.JSONDecodeError as error:
        raise WorkerError("worker training configuration is not valid JSON") from error
    if not isinstance(training_config, Mapping):
        raise WorkerError("worker training configuration must be an object")
    try:
        identity = Grid5000RunIdentity(
            source_commit=source_commit,
            dataset_revision=dataset_revision,
            model_name_or_path=training_config["model_name_or_path"],  # type: ignore[arg-type]
            model_revision=model_revision,
            training_config=training_config,
        )
    except (KeyError, Grid5000ConfigurationError) as error:
        raise WorkerError("worker run identity is invalid") from error
    if identity.run_id != run_id:
        raise WorkerError("worker run ID does not match its immutable inputs")
    return identity


def main(argv: Sequence[str] | None = None) -> int:
    """Run the compute-node worker from the fixed remote module command."""

    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--training-config-json", required=True)
    parser.add_argument("--checkout", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        identity = _identity_from_arguments(
            run_id=args.run_id,
            source_commit=args.source_commit,
            dataset_revision=args.dataset_revision,
            model_revision=args.model_revision,
            training_config_json=args.training_config_json,
        )
        run_landuse_training_worker(identity, checkout=args.checkout)
    except WorkerError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CudaProbe",
    "WorkerError",
    "WorkerFacts",
    "main",
    "run_landuse_training_worker",
    "validate_compute_node",
]
