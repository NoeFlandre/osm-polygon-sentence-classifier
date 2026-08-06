"""Strict compute-node preflight and training boundary for Grid'5000."""

from __future__ import annotations

import json
import os
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from .checkpointing import CheckpointError, find_latest_complete_checkpoint
from .config import ConfigurationError, ProjectConfig
from .dataset_contract import LANDUSE_DATASET_CONTRACT
from .grid5000 import (
    COMMAND_TIMEOUT_SECONDS,
    MINIMUM_CUDA_CAPABILITY,
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


def _trackio_segment_run_name(
    base_name: str,
    *,
    run_id: str,
    starting_step: int,
    job_id: int,
) -> str:
    """Name one allocation segment so continuations are legible in Trackio."""

    return (
        f"{base_name} | run-{run_id[:8]} | "
        f"segment-from-{starting_step:04d} | oar-{job_id}"
    )


CudaProbe = Callable[[], tuple[bool, int, str, tuple[int, int]]]


def _default_cuda_probe() -> tuple[bool, int, str, tuple[int, int]]:
    try:
        torch = cast(Any, import_module("torch"))
    except ModuleNotFoundError as error:
        raise WorkerError(
            "Grid'5000 worker requires the torch training dependency"
        ) from error
    available = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count())
    device_name = ""
    capability = (0, 0)
    if available and count == 1:
        device_name = str(torch.cuda.get_device_name(0))
        raw_capability = torch.cuda.get_device_capability(0)
        capability = (int(raw_capability[0]), int(raw_capability[1]))
    return available, count, device_name, capability


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
        cuda_available, device_count, device_name, capability = probe()
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
    if capability < MINIMUM_CUDA_CAPABILITY:
        minimum = ".".join(str(part) for part in MINIMUM_CUDA_CAPABILITY)
        actual = ".".join(str(part) for part in capability)
        raise WorkerError(
            f"CUDA compute capability {actual} is below the required {minimum}"
        )
    return device_name


def _has_hugging_face_auth(environ: Mapping[str, str]) -> bool:
    token = environ.get("HF_TOKEN", "").strip()
    if token:
        return True
    hf_home = Path(environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    try:
        return bool((hf_home / "token").read_text(encoding="utf-8").strip())
    except OSError:
        return False


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
    checkout_source_commit: str | None = None,
    training_config: TrainingConfig | None = None,
    remote_data_root: Path | None = None,
    environ: Mapping[str, str] | None = None,
    platform_name: str | None = None,
    git_runner: CommandRunner | None = None,
    cuda_probe: CudaProbe | None = None,
    train: Callable[..., TrainingResult] = train_landuse_classifier,
    require_checkpoint: bool = False,
) -> TrainingResult:
    """Preflight one compute node, then call the existing training boundary."""

    if (
        identity.dataset_revision
        != LANDUSE_DATASET_CONTRACT.provenance.repository_revision
    ):
        raise WorkerError("worker dataset revision does not match the pinned contract")
    config_values: dict[str, Any] = dict(identity.training_config)
    try:
        identity_config = TrainingConfig(**config_values)
    except (TypeError, TrainingError) as error:
        raise WorkerError("worker training configuration is invalid") from error
    if training_config is not None and training_config != identity_config:
        raise WorkerError("worker training configuration does not match identity")
    effective_config = training_config or identity_config
    if effective_config.model_name_or_path != identity.model_name_or_path:
        raise WorkerError("worker model identity does not match training configuration")
    if effective_config.model_revision != identity.model_revision:
        raise WorkerError("worker model revision does not match training configuration")
    effective_environment = os.environ if environ is None else environ
    if (
        effective_config.publish_to_hub or effective_config.sync_trackio
    ) and not _has_hugging_face_auth(effective_environment):
        raise WorkerError(
            "worker Hugging Face authentication is unavailable for publication"
        )
    worker_facts = validate_compute_node(
        expected_source_commit=checkout_source_commit or identity.source_commit,
        checkout=checkout,
        environ=effective_environment,
        platform_name=platform_name,
        git_runner=git_runner,
        cuda_probe=cuda_probe,
    )
    data_root = remote_data_root or Path.home() / REMOTE_DATA_SUBDIRECTORY
    try:
        project_config = ProjectConfig.for_remote_root(data_root)
    except (ConfigurationError, Grid5000ConfigurationError) as error:
        raise WorkerError("remote worker data root is unsafe") from error
    output_directory = project_config.data_root / effective_config.output_subdirectory
    try:
        checkpoint = find_latest_complete_checkpoint(
            output_directory,
            identity=identity.canonical_payload,
        )
    except CheckpointError as error:
        raise WorkerError("checkpoint evidence is invalid") from error
    if require_checkpoint and checkpoint is None:
        raise WorkerError("no complete checkpoint is available for continuation")
    trackio_config = replace(
        effective_config,
        run_name=_trackio_segment_run_name(
            effective_config.run_name,
            run_id=identity.run_id,
            starting_step=checkpoint.global_step if checkpoint is not None else 0,
            job_id=worker_facts.job_id,
        ),
    )
    return train(
        config=trackio_config,
        project_config=project_config,
        resume_from_checkpoint=checkpoint.path if checkpoint is not None else None,
        checkpoint_identity=identity.canonical_payload,
    )


def write_completion_manifest(
    identity: Grid5000RunIdentity,
    result: TrainingResult,
    *,
    remote_data_root: Path,
) -> Path:
    """Write one atomic, credential-free manifest after successful training."""

    raw_data_root = Path(remote_data_root).expanduser()
    raw_output_directory = Path(result.output_directory).expanduser()
    if not raw_data_root.is_absolute() or not raw_output_directory.is_absolute():
        raise WorkerError("completion paths must be absolute")
    if _contains_symlink(raw_data_root) or _contains_symlink(raw_output_directory):
        raise WorkerError("completion paths must not be symlinks")
    data_root = raw_data_root.resolve()
    output_directory = raw_output_directory.resolve()
    if not output_directory.is_relative_to(data_root):
        raise WorkerError("training output is outside the managed remote data root")
    data_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(data_root, 0o700)
    manifest_path = data_root / "completion.json"
    if manifest_path.is_symlink():
        raise WorkerError("completion manifest cannot be a symlink")
    publication = result.model_publication
    payload: dict[str, object] = {
        "schema_version": 1,
        "run_id": identity.run_id,
        "source_commit": identity.source_commit,
        "dataset_revision": identity.dataset_revision,
        "model_name_or_path": identity.model_name_or_path,
        "model_revision": identity.model_revision,
        "output_directory": str(output_directory.relative_to(data_root)),
        "model_publication": (
            {
                "repository_id": publication.repository_id,
                "commit_id": publication.commit_id,
                "commit_url": publication.commit_url,
                "files": list(publication.files),
            }
            if publication is not None
            else None
        ),
        "tracking_space_id": result.tracking_space_id,
    }
    temporary = manifest_path.with_name(".completion.json.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        os.replace(temporary, manifest_path)
    except (OSError, TypeError, ValueError) as error:
        temporary.unlink(missing_ok=True)
        raise WorkerError("completion manifest cannot be written") from error
    return manifest_path


def _contains_symlink(path: Path) -> bool:
    """Return whether an absolute path or one of its parents is a symlink."""

    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            return True
    return False


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
    parser.add_argument(
        "--checkout-commit",
        default=None,
        help="optional code checkout revision for an identity-preserving resume",
    )
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--training-config-json", required=True)
    parser.add_argument("--checkout", type=Path, default=Path.cwd())
    parser.add_argument(
        "--remote-data-root",
        type=Path,
        default=Path.home() / REMOTE_DATA_SUBDIRECTORY,
    )
    parser.add_argument(
        "--require-checkpoint",
        action="store_true",
        help="fail instead of restarting if no complete checkpoint is found",
    )
    args = parser.parse_args(argv)
    try:
        identity = _identity_from_arguments(
            run_id=args.run_id,
            source_commit=args.source_commit,
            dataset_revision=args.dataset_revision,
            model_revision=args.model_revision,
            training_config_json=args.training_config_json,
        )
        result = run_landuse_training_worker(
            identity,
            checkout=args.checkout,
            checkout_source_commit=args.checkout_commit,
            remote_data_root=args.remote_data_root,
            require_checkpoint=args.require_checkpoint,
        )
        write_completion_manifest(
            identity,
            result,
            remote_data_root=args.remote_data_root,
        )
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
    "write_completion_manifest",
]
