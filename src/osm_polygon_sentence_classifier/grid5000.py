"""Guarded Grid'5000 planning and submission boundaries for classifier training."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, Protocol, cast

from .config import ProjectConfig
from .dataset_contract import LANDUSE_DATASET_CONTRACT

GRID5000_DATASET_REVISION = LANDUSE_DATASET_CONTRACT.provenance.repository_revision
MAX_WALLTIME_SECONDS = 12 * 60 * 60
MAX_DAY_WALLTIME_SECONDS = 60 * 60
DEFAULT_DAY_WALLTIME_SECONDS = 30 * 60
# The UV wheel cache is shared by this project beneath the persistent home
# quota so short jobs can build a node-local runtime without downloading the
# locked dependencies repeatedly. Eight GiB leaves room for two resumable
# checkpoints, the final model, metadata, and publication artifacts beneath
# the persistent home quota.
MINIMUM_HOME_HEADROOM_BYTES = 8 * 1024**3
MINIMUM_CUDA_CAPABILITY: Final[tuple[int, int]] = (7, 5)
COMMAND_TIMEOUT_SECONDS = 120.0
REMOTE_CHECKOUT_SUBDIRECTORY = "osm-polygon-sentence-classifier"
REMOTE_DATA_SUBDIRECTORY = "osm-polygon-sentence-classifier-data"
REMOTE_RUNS_SUBDIRECTORY = "grid5000/runs"
REMOTE_ENVIRONMENT_SUBDIRECTORY = "grid5000/environment"
WORKER_MODULE = "osm_polygon_sentence_classifier.grid5000_worker"
CONTAINER_HOME = "/home/app"
CONTAINER_CHECKOUT = f"{CONTAINER_HOME}/checkout"
CONTAINER_DATA_ROOT = f"{CONTAINER_HOME}/data"
SUPPORTED_TASK_NAMES: Final = frozenset({"landuse", "place-relevance-v2"})

_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_SITE_PATTERN = re.compile(r"[a-z][a-z0-9-]*")
_CUDA_CAPABILITY_LITERAL = r"[0-9]+\.[0-9]+"
_RESOURCE_PROPERTY_PATTERN = re.compile(
    rf"gpu_mem>=[1-9][0-9]* AND production='(?:YES|NO)' AND "
    rf"cpuarch='x86_64' AND "
    rf"gpu_compute_capability IN \('{_CUDA_CAPABILITY_LITERAL}'"
    rf"(?:, '{_CUDA_CAPABILITY_LITERAL}')*\)"
)
_RUN_ID_PATTERN = re.compile(r"[0-9a-f]{20}")
_JOB_ID_PATTERN = re.compile(r"(?:OAR_JOB_ID=|^)([1-9][0-9]*)$", re.MULTILINE)
_CONTAINER_IMAGE_PATTERN = re.compile(r"^\S+@sha256:[0-9a-f]{64}$")
_CONTAINER_PREFLIGHT_CODE = (
    "import pathlib, torch; "
    "checkout = pathlib.Path('/home/app/checkout'); "
    "data = pathlib.Path('/home/app/data'); "
    "assert (checkout / '.git').is_dir(); "
    "assert data.is_dir(); "
    "assert torch.cuda.is_available() and torch.cuda.device_count() == 1; "
    "probe = data / '.container-preflight'; probe.touch(); probe.unlink()"
)
_QUOTA_ROW_PATTERN = re.compile(
    r"^\s*(?P<used>[0-9]+)\*?\s+"
    r"(?P<soft>[0-9]+)\s+"
    r"(?P<hard>[0-9]+)(?:\s|$)"
)
_HOME_QUOTA_COMMAND = (
    "set +e; quota_output=$(quota 2>&1); quota_rc=$?; set -e; "
    'if [ "$quota_rc" -gt 1 ]; then exit "$quota_rc"; fi; '
    "printf '%s\\n' \"$quota_output\""
)

Grid5000Phase = Literal["submitting", "submitted"]
ContainerRuntime = Literal["auto", "docker", "podman"]


class Grid5000ConfigurationError(ValueError):
    """Raised when a Grid'5000 plan or response violates its contract."""


class Grid5000StateError(RuntimeError):
    """Raised when durable submission state is missing, unsafe, or ambiguous."""


class Grid5000ExecutionError(RuntimeError):
    """Raised when a guarded remote preflight or submission fails."""


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Small subprocess result used by injected Grid'5000 command runners."""

    returncode: int
    stdout: str = ""
    stderr: str = ""


class CommandRunner(Protocol):
    """Callable boundary for fixed-argument local or SSH commands."""

    def __call__(self, argv: Sequence[str], *, timeout: float) -> CommandResult: ...


class SubprocessCommandRunner:
    """Run an already constructed argument vector without a shell."""

    def __call__(self, argv: Sequence[str], *, timeout: float) -> CommandResult:
        try:
            result = subprocess.run(
                tuple(argv),
                capture_output=True,
                check=False,
                text=True,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise Grid5000ExecutionError(
                "Grid'5000 command could not complete"
            ) from error
        return CommandResult(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise Grid5000ConfigurationError(
            "training_config must contain only JSON-compatible finite values"
        ) from error


def _canonical_training_config(
    value: Mapping[str, object],
) -> tuple[str, dict[str, object]]:
    if not isinstance(value, Mapping):
        raise Grid5000ConfigurationError("training_config must be a mapping")
    try:
        normalized = json.loads(_canonical_json(dict(value)))
    except json.JSONDecodeError as error:
        raise Grid5000ConfigurationError(
            "training_config must be a JSON object"
        ) from error
    if not isinstance(normalized, dict) or any(
        not isinstance(key, str) for key in normalized
    ):
        raise Grid5000ConfigurationError("training_config must be a JSON object")
    canonical = _canonical_json(normalized)
    return canonical, normalized


def _require_revision(name: str, value: object) -> str:
    if not isinstance(value, str) or _REVISION_PATTERN.fullmatch(value) is None:
        raise Grid5000ConfigurationError(
            f"{name} must be exactly 40 lowercase hexadecimal characters"
        )
    return value


def _require_non_empty(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\n" in value
        or "\r" in value
    ):
        raise Grid5000ConfigurationError(
            f"{name} must be a non-empty single-line string"
        )
    return value


def _validate_container_settings(
    container_image: str | None,
    container_runtime: ContainerRuntime,
) -> None:
    if not isinstance(container_runtime, str) or container_runtime not in {
        "auto",
        "docker",
        "podman",
    }:
        raise Grid5000ConfigurationError(
            "container_runtime must be auto, docker, or podman"
        )
    if container_image is None:
        if container_runtime != "auto":
            raise Grid5000ConfigurationError(
                "container_runtime requires an explicit container_image"
            )
        return
    image = _require_non_empty("container_image", container_image)
    if not _CONTAINER_IMAGE_PATTERN.fullmatch(image):
        raise Grid5000ConfigurationError(
            "container_image must include an immutable sha256 digest"
        )


@dataclass(frozen=True, slots=True)
class Grid5000RunIdentity:
    """Immutable inputs whose canonical digest identifies one training run."""

    source_commit: str
    dataset_revision: str
    model_name_or_path: str
    model_revision: str
    training_config: Mapping[str, object]
    task_name: str = "landuse"
    _training_config_json: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _require_revision("source_commit", self.source_commit)
        _require_revision("dataset_revision", self.dataset_revision)
        _require_revision("model_revision", self.model_revision)
        _require_non_empty("model_name_or_path", self.model_name_or_path)
        if (
            not isinstance(self.task_name, str)
            or self.task_name not in SUPPORTED_TASK_NAMES
        ):
            supported = ", ".join(sorted(SUPPORTED_TASK_NAMES))
            raise Grid5000ConfigurationError(f"task_name must be one of: {supported}")
        canonical, normalized = _canonical_training_config(self.training_config)
        object.__setattr__(self, "training_config", MappingProxyType(normalized))
        object.__setattr__(self, "_training_config_json", canonical)

    @property
    def canonical_payload(self) -> dict[str, object]:
        """Return the JSON-safe identity payload used for hashing and state."""

        return {
            "dataset_revision": self.dataset_revision,
            "model_name_or_path": self.model_name_or_path,
            "model_revision": self.model_revision,
            "source_commit": self.source_commit,
            "task_name": self.task_name,
            "training_config": json.loads(self._training_config_json),
        }

    @property
    def canonical_json(self) -> str:
        """Return the stable, sorted identity JSON representation."""

        return _canonical_json(self.canonical_payload)

    @property
    def fingerprint(self) -> str:
        """Return the full SHA-256 digest of the canonical identity."""

        return hashlib.sha256(self.canonical_json.encode("utf-8")).hexdigest()

    @property
    def run_id(self) -> str:
        """Return the short deterministic identifier used for local state."""

        return self.fingerprint[:20]

    @property
    def training_config_json(self) -> str:
        """Return the canonical training configuration for the remote worker."""

        return self._training_config_json

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> Grid5000RunIdentity:
        """Reconstruct and validate an identity persisted in durable state."""

        try:
            source_commit = payload["source_commit"]
            dataset_revision = payload["dataset_revision"]
            model_name_or_path = payload["model_name_or_path"]
            model_revision = payload["model_revision"]
            training_config = payload["training_config"]
            task_name = payload.get("task_name", "landuse")
        except KeyError as error:
            raise Grid5000StateError(
                "durable state has an incomplete run identity"
            ) from error
        if (
            not isinstance(source_commit, str)
            or not isinstance(dataset_revision, str)
            or not isinstance(model_name_or_path, str)
            or not isinstance(model_revision, str)
            or not isinstance(task_name, str)
            or not isinstance(training_config, Mapping)
        ):
            raise Grid5000StateError("durable state has an invalid run identity")
        try:
            return cls(
                source_commit=source_commit,
                dataset_revision=dataset_revision,
                model_name_or_path=model_name_or_path,
                model_revision=model_revision,
                training_config=cast(Mapping[str, object], training_config),
                task_name=task_name,
            )
        except Grid5000ConfigurationError as error:
            raise Grid5000StateError(
                "durable state has an invalid run identity"
            ) from error


@dataclass(frozen=True, slots=True)
class Grid5000Allocation:
    """One bounded, policy-compliant OAR allocation request."""

    site: str
    walltime_seconds: int
    queue: str = "default"
    resource_type: str = "exotic"
    policy_type: str = "night"
    gpu_count: int = 1
    resource_property: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.site, str) or _SITE_PATTERN.fullmatch(self.site) is None:
            raise Grid5000ConfigurationError(
                "site must be a lowercase Grid'5000 site name"
            )
        if self.queue not in {"default", "production"}:
            raise Grid5000ConfigurationError("queue must be 'default' or 'production'")
        if self.resource_type not in {"standard", "exotic"}:
            raise Grid5000ConfigurationError(
                "resource_type must be 'standard' or 'exotic'"
            )
        if self.policy_type not in {"day", "night"}:
            raise Grid5000ConfigurationError("policy_type must be 'day' or 'night'")
        if (
            isinstance(self.gpu_count, bool)
            or not isinstance(self.gpu_count, int)
            or self.gpu_count != 1
        ):
            raise Grid5000ConfigurationError("gpu_count must be exactly 1")
        if (
            isinstance(self.walltime_seconds, bool)
            or not isinstance(self.walltime_seconds, int)
            or self.walltime_seconds <= 0
            or self.walltime_seconds > MAX_WALLTIME_SECONDS
        ):
            raise Grid5000ConfigurationError(
                "walltime_seconds must be between 1 second and 12 hours"
            )
        if (
            self.policy_type == "day"
            and self.walltime_seconds > MAX_DAY_WALLTIME_SECONDS
        ):
            raise Grid5000ConfigurationError(
                "day policy walltime_seconds must be at most one hour"
            )
        if self.resource_property is not None and (
            _RESOURCE_PROPERTY_PATTERN.fullmatch(self.resource_property) is None
        ):
            raise Grid5000ConfigurationError(
                "resource_property must be a generated GPU capability filter"
            )

    @property
    def walltime(self) -> str:
        """Return the OAR ``HH:MM:SS`` walltime."""

        hours, remainder = divmod(self.walltime_seconds, 3_600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def scheduler_command(self, worker_command: str) -> tuple[str, ...]:
        """Return an OAR argv tuple containing exactly one worker command."""

        if not worker_command or "\n" in worker_command or "\r" in worker_command:
            raise Grid5000ConfigurationError("worker_command must be a single line")
        command: list[str] = ["oarsub", "-q", self.queue]
        if self.resource_property is not None:
            command.extend(("-p", self.resource_property))
        if self.resource_type == "exotic":
            command.extend(("-t", self.resource_type))
        command.extend(
            (
                "-t",
                self.policy_type,
                "-l",
                f"gpu={self.gpu_count},walltime={self.walltime}",
                worker_command,
            )
        )
        return tuple(command)


def _ssh_argv(site: str, remote_command: str) -> tuple[str, ...]:
    if _SITE_PATTERN.fullmatch(site) is None:
        raise Grid5000ConfigurationError("site must be a lowercase Grid'5000 site name")
    return (
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "ServerAliveInterval=10",
        "-o",
        "ServerAliveCountMax=2",
        site,
        remote_command,
    )


@dataclass(frozen=True, slots=True)
class Grid5000Plan:
    """Reproducible identity, worker command, and scheduler request."""

    identity: Grid5000RunIdentity
    allocation: Grid5000Allocation
    resume_from_checkpoint: bool = False
    checkout_commit: str | None = None
    container_image: str | None = None
    container_runtime: ContainerRuntime = "auto"

    def __post_init__(self) -> None:
        if self.checkout_commit is not None:
            _require_revision("checkout_commit", self.checkout_commit)
        _validate_container_settings(self.container_image, self.container_runtime)

    @property
    def worker_command(self) -> str:
        """Return the fixed compute-node command for the training worker.

        The locked dependencies are built into node-local scratch from the
        project-scoped UV wheel cache. Model caches, outputs, checkpoints, and
        Trackio state remain in the managed run root.
        """

        if self.container_image is not None:
            return self._container_worker_command()

        return self._uv_worker_command()

    def _uv_worker_command(self) -> str:
        checkout_args: tuple[str, ...] = ()
        if self.checkout_commit is not None:
            checkout_args = ("--checkout-commit", self.checkout_commit)
        worker_args = (
            "run",
            "--locked",
            "python",
            "-m",
            WORKER_MODULE,
            "--task-name",
            self.identity.task_name,
            "--run-id",
            self.identity.run_id,
            "--source-commit",
            self.identity.source_commit,
            *checkout_args,
            "--dataset-revision",
            self.identity.dataset_revision,
            "--model-revision",
            self.identity.model_revision,
            "--training-config-json",
            self.identity.training_config_json,
        )
        worker_command = shlex.join(
            (
                *worker_args[:2],
                "--no-dev",
                "--extra",
                "training",
                *worker_args[2:],
            )
        )
        remote_run_root = (
            f'"$HOME/{REMOTE_DATA_SUBDIRECTORY}/{REMOTE_RUNS_SUBDIRECTORY}/'
            f'{self.identity.run_id}"'
        )
        runtime_root = '"${TMPDIR:-/tmp}/osm-polygon-sentence-classifier-${OAR_JOB_ID}"'
        worker_command += ' --remote-data-root "$remote_run_root"'
        if self.resume_from_checkpoint:
            worker_command += " --require-checkpoint"
        return (
            f'cd "$HOME/{REMOTE_CHECKOUT_SUBDIRECTORY}" && '
            "umask 077 && set -eu && "
            f"remote_run_root={remote_run_root} && "
            f"runtime_root={runtime_root} && "
            'mkdir -p "$runtime_root" && '
            'export UV_PROJECT_ENVIRONMENT="$runtime_root/environment" && '
            'export UV_CACHE_DIR="$HOME/.cache/osm-polygon-sentence-classifier/uv" && '
            'cpu_architecture="$(uname -m)"; '
            '[ "$cpu_architecture" = "x86_64" ] || '
            '{ echo "unsupported compute-node architecture: $cpu_architecture" >&2; exit 78; }; '
            'uv_bin="$(command -v uv || true)"; '
            '[ -n "$uv_bin" ] || uv_bin="$HOME/.local/bin/uv"; '
            'test -x "$uv_bin"; '
            '"$uv_bin" --version >/dev/null 2>&1 || '
            '{ echo "uv is not executable on compute-node architecture $cpu_architecture" >&2; exit 78; }; '
            'if [ -d "$HOME/.cache/osm-polygon-sentence-classifier/wheels" ]; '
            'then torch_wheel="$(find '
            '"$HOME/.cache/osm-polygon-sentence-classifier/wheels" '
            '-maxdepth 1 -type f -name "torch-*.whl" -print -quit)"; '
            'if [ -n "$torch_wheel" ]; then '
            'cp "$torch_wheel" "$runtime_root/torch.whl"; '
            '"$uv_bin" venv "$UV_PROJECT_ENVIRONMENT" --allow-existing '
            "--no-python-downloads >/dev/null; "
            '"$uv_bin" pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" '
            '--no-index --no-deps --find-links "$runtime_root" torch; fi; '
            "fi && "
            'exec "$uv_bin" ' + worker_command
        )

    def _container_worker_command(self) -> str:
        if self.container_image is None:
            raise Grid5000ConfigurationError(
                "container worker command requires a container image"
            )
        worker_args = (
            "python",
            "-m",
            WORKER_MODULE,
            "--task-name",
            self.identity.task_name,
            "--run-id",
            self.identity.run_id,
            "--source-commit",
            self.identity.source_commit,
            "--dataset-revision",
            self.identity.dataset_revision,
            "--model-revision",
            self.identity.model_revision,
            "--training-config-json",
            self.identity.training_config_json,
            "--checkout",
            CONTAINER_CHECKOUT,
            "--remote-data-root",
            CONTAINER_DATA_ROOT,
        )
        worker_command = shlex.join(worker_args)
        if self.resume_from_checkpoint:
            worker_command += " --require-checkpoint"

        image = shlex.quote(self.container_image)
        checkout = f'"$HOME/{REMOTE_CHECKOUT_SUBDIRECTORY}"'
        data_root = (
            f'"$HOME/{REMOTE_DATA_SUBDIRECTORY}/{REMOTE_RUNS_SUBDIRECTORY}/'
            f'{self.identity.run_id}"'
        )
        if self.container_runtime == "auto":
            runtime_selection = (
                'container_runtime=""; '
                "if command -v docker >/dev/null 2>&1 "
                "&& docker info >/dev/null 2>&1; then "
                'container_runtime="docker"; '
                "elif command -v podman >/dev/null 2>&1 "
                "&& podman info >/dev/null 2>&1; then "
                'container_runtime="podman"; '
                "else "
                'echo "an accessible Docker or Podman runtime is unavailable on the compute node" >&2; '
                "exit 78; fi; "
            )
        else:
            runtime = shlex.quote(self.container_runtime)
            runtime_selection = (
                f"container_runtime={runtime}; "
                'command -v "$container_runtime" >/dev/null 2>&1 || '
                ' { echo "requested container runtime is unavailable on the compute node" >&2; exit 78; }; '
            )
        return (
            "set -euo pipefail; umask 077; "
            f"checkout={checkout}; data_root={data_root}; "
            '[ -d "$checkout/.git" ] || '
            '{ echo "container checkout mount is unavailable" >&2; exit 78; }; '
            '[ -d "$data_root" ] || '
            '{ echo "container data mount is unavailable" >&2; exit 78; }; '
            + runtime_selection
            + '"$container_runtime" info >/dev/null 2>&1 || '
            + '{ echo "container runtime is unavailable or inaccessible on the compute node" >&2; exit 78; }; '
            + f'"$container_runtime" image inspect {image} >/dev/null 2>&1 || '
            + '{ echo "pinned container image is unavailable on the compute node" >&2; exit 78; }; '
            + 'cuda_visible_devices="${CUDA_VISIBLE_DEVICES:-}"; '
            + 'case "$cuda_visible_devices" in '
            + '""|*,*|*[[:space:]]*) '
            + '{ echo "exactly one OAR-assigned CUDA_VISIBLE_DEVICES value is required" >&2; exit 78; } ;; '
            + "esac; "
            + 'if [ "$container_runtime" = "docker" ]; then '
            + 'gpu_args=(--gpus "device=$cuda_visible_devices"); '
            + 'else gpu_args=(--device "nvidia.com/gpu=$cuda_visible_devices"); fi; '
            + '"$container_runtime" run --rm --network none --read-only --tmpfs /tmp:rw,noexec,nosuid '
            + '--user "$(id -u):$(id -g)" "${gpu_args[@]}" '
            + "--env HOME=/home/app --env CUDA_VISIBLE_DEVICES=0 "
            + '--mount "type=bind,src=$checkout,dst=/home/app/checkout,readonly" '
            + '--mount "type=bind,src=$data_root,dst=/home/app/data" '
            + f"{image} python -c {shlex.quote(_CONTAINER_PREFLIGHT_CODE)} >/dev/null 2>&1 || "
            + '{ echo "container preflight failed: image, mounts, writable data, or one-GPU CUDA access is unavailable" >&2; exit 78; }; '
            + "token_args=(); "
            + 'if [ -f "$HOME/.cache/huggingface/token" ]; then '
            + 'test ! -L "$data_root/cache" && '
            + 'test ! -L "$data_root/cache/huggingface"; '
            + 'mkdir -p -m 700 "$data_root/cache/huggingface"; '
            + 'token_args=(--mount "type=bind,src=$HOME/.cache/huggingface/token,'
            + 'dst=/home/app/data/cache/huggingface/token,readonly"); '
            + "fi; "
            + 'exec "$container_runtime" run --rm --read-only '
            + "--tmpfs /tmp:rw,noexec,nosuid "
            + '--user "$(id -u):$(id -g)" '
            + "--env HOME=/home/app --env HF_HOME=/home/app/data/cache/huggingface --env OAR_JOB_ID "
            + "--env CUDA_VISIBLE_DEVICES=0 --env PYTHONPATH=/home/app/checkout/src "
            + '--mount "type=bind,src=$checkout,dst=/home/app/checkout,readonly" '
            + '--mount "type=bind,src=$data_root,dst=/home/app/data" '
            + '"${gpu_args[@]}" "${token_args[@]}" '
            f"{image} {worker_command}"
        )

    @property
    def scheduler_command(self) -> tuple[str, ...]:
        """Return the unquoted OAR command arguments."""

        return self.allocation.scheduler_command(self.worker_command)

    @property
    def remote_checkout_command(self) -> tuple[str, ...]:
        """Return a read-only exact-commit and clean-checkout guard."""

        checkout = f'"$HOME/{REMOTE_CHECKOUT_SUBDIRECTORY}"'
        checkout_commit = self.checkout_commit or self.identity.source_commit
        remote_command = (
            f"test -d {checkout}/.git && "
            f'test "$(git -C {checkout} rev-parse HEAD)" = '
            f"{checkout_commit} && "
            f'test -z "$(git -C {checkout} status --porcelain)"'
        )
        return _ssh_argv(self.allocation.site, remote_command)

    @property
    def policy_site_command(self) -> tuple[str, ...]:
        """Return the read-only site policy check over bounded SSH."""

        return _ssh_argv(
            self.allocation.site,
            "usagepolicycheck -l --sites " + shlex.quote(self.allocation.site),
        )

    @property
    def policy_total_command(self) -> tuple[str, ...]:
        """Return the read-only total policy check over bounded SSH."""

        return _ssh_argv(self.allocation.site, "usagepolicycheck -t")

    @property
    def quota_command(self) -> tuple[str, ...]:
        """Return the read-only home-quota check over bounded SSH."""

        return _ssh_argv(self.allocation.site, _HOME_QUOTA_COMMAND)

    @property
    def submission_command(self) -> tuple[str, ...]:
        """Return the one fixed OAR submission command over bounded SSH."""

        return _ssh_argv(
            self.allocation.site,
            shlex.join(self.scheduler_command),
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe human-readable plan."""

        return {
            "allocation": {
                "gpu_count": self.allocation.gpu_count,
                "policy_type": self.allocation.policy_type,
                "queue": self.allocation.queue,
                "resource_type": self.allocation.resource_type,
                "resource_property": self.allocation.resource_property,
                "site": self.allocation.site,
                "walltime": self.allocation.walltime,
                "walltime_seconds": self.allocation.walltime_seconds,
            },
            "identity": self.identity.canonical_payload,
            "container_image": self.container_image,
            "container_runtime": self.container_runtime,
            "resume_from_checkpoint": self.resume_from_checkpoint,
            "remote_checkout_command": shlex.join(self.remote_checkout_command),
            "run_id": self.identity.run_id,
            "scheduler_command": list(self.scheduler_command),
            "submission_command": shlex.join(self.submission_command),
        }


@dataclass(frozen=True, slots=True)
class QuotaUsage:
    """Byte-denominated Grid'5000 home quota facts."""

    used_bytes: int
    soft_limit_bytes: int
    hard_limit_bytes: int

    @property
    def soft_headroom_bytes(self) -> int:
        """Return headroom that remains beneath the soft limit."""

        return max(0, self.soft_limit_bytes - self.used_bytes)

    @property
    def soft_limit_exceeded(self) -> bool:
        """Return whether usage is already above the soft limit."""

        return self.used_bytes > self.soft_limit_bytes


def parse_quota_output(output: str) -> QuotaUsage:
    """Parse the first usable quota row emitted by ``quota``."""

    if not isinstance(output, str):
        raise Grid5000ConfigurationError("quota output must be text")
    for line in output.splitlines():
        match = _QUOTA_ROW_PATTERN.match(line)
        if match is None:
            continue
        used_kib = int(match.group("used"))
        soft_kib = int(match.group("soft"))
        hard_kib = int(match.group("hard"))
        if soft_kib <= 0 or hard_kib < soft_kib:
            raise Grid5000ConfigurationError("home quota limits are invalid")
        return QuotaUsage(
            used_bytes=used_kib * 1024,
            soft_limit_bytes=soft_kib * 1024,
            hard_limit_bytes=hard_kib * 1024,
        )
    raise Grid5000ConfigurationError("home quota output has no usable data row")


@dataclass(frozen=True, slots=True)
class Grid5000State:
    """Durable evidence for one submission attempt."""

    identity: Grid5000RunIdentity
    phase: Grid5000Phase
    scheduler_command: tuple[str, ...]
    job_id: int | None = None
    submission_command: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.phase not in ("submitting", "submitted"):
            raise Grid5000StateError("durable state has an unsupported phase")
        if not self.scheduler_command or any(
            not isinstance(value, str) for value in self.scheduler_command
        ):
            raise Grid5000StateError("durable state has no scheduler command")
        if not self.submission_command:
            object.__setattr__(self, "submission_command", self.scheduler_command)
        elif any(not isinstance(value, str) for value in self.submission_command):
            raise Grid5000StateError("durable state submission command is invalid")
        if self.phase == "submitting" and self.job_id is not None:
            raise Grid5000StateError("submitting state cannot contain a job ID")
        if self.phase == "submitted" and (
            isinstance(self.job_id, bool)
            or not isinstance(self.job_id, int)
            or self.job_id <= 0
        ):
            raise Grid5000StateError("submitted state must contain one positive job ID")

    def to_dict(self) -> dict[str, object]:
        """Return the state document written to disk."""

        return {
            "identity": self.identity.canonical_payload,
            "job_id": self.job_id,
            "phase": self.phase,
            "scheduler_command": list(self.scheduler_command),
            "submission_command": list(self.submission_command),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> Grid5000State:
        try:
            identity_payload = payload["identity"]
            scheduler_command = payload["scheduler_command"]
            phase = payload["phase"]
            job_id = payload["job_id"]
        except KeyError as error:
            raise Grid5000StateError("durable state is incomplete") from error
        if not isinstance(identity_payload, Mapping):
            raise Grid5000StateError("durable state identity is invalid")
        if not isinstance(scheduler_command, list) or any(
            not isinstance(value, str) for value in scheduler_command
        ):
            raise Grid5000StateError("durable state scheduler command is invalid")
        submission_command = payload.get("submission_command", scheduler_command)
        if not isinstance(submission_command, list) or any(
            not isinstance(value, str) for value in submission_command
        ):
            raise Grid5000StateError("durable state submission command is invalid")
        if not isinstance(phase, str) or phase not in ("submitting", "submitted"):
            raise Grid5000StateError("durable state phase is invalid")
        if job_id is not None and (
            isinstance(job_id, bool) or not isinstance(job_id, int)
        ):
            raise Grid5000StateError("durable state job ID is invalid")
        return cls(
            identity=Grid5000RunIdentity.from_payload(
                cast(Mapping[str, object], identity_payload)
            ),
            phase=phase,
            scheduler_command=tuple(cast(str, value) for value in scheduler_command),
            job_id=cast(int | None, job_id),
            submission_command=tuple(cast(str, value) for value in submission_command),
        )


def _check_managed_mode(path: Path, expected: int, message: str) -> None:
    if path.is_symlink() or not path.exists():
        raise Grid5000StateError(message)
    if path.stat().st_mode & 0o777 != expected:
        raise Grid5000StateError(message)


def _reject_symlink_components(path: Path) -> None:
    if not path.is_absolute():
        raise Grid5000StateError("state root must be an absolute path")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise Grid5000StateError("state root cannot contain symlink components")


class Grid5000StateStore:
    """Secure local state below the approved external data root."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = (
            Path(root)
            if root is not None
            else ProjectConfig().data_root / "grid5000/runs"
        )
        _reject_symlink_components(self.root)

    def _validate_run_id(self, run_id: str) -> None:
        if _RUN_ID_PATTERN.fullmatch(run_id) is None:
            raise Grid5000StateError("run ID is invalid")

    def _run_directory(self, run_id: str) -> Path:
        self._validate_run_id(run_id)
        return self.root / run_id

    def _ensure_run_directory(self, run_id: str) -> Path:
        _reject_symlink_components(self.root)
        if self.root.is_symlink():
            raise Grid5000StateError("state root cannot be a symlink")
        try:
            self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(self.root, 0o700)
        except OSError as error:
            raise Grid5000StateError("state root cannot be created securely") from error
        run_directory = self._run_directory(run_id)
        if run_directory.is_symlink():
            raise Grid5000StateError("run state directory cannot be a symlink")
        try:
            run_directory.mkdir(mode=0o700, exist_ok=True)
            os.chmod(run_directory, 0o700)
        except OSError as error:
            raise Grid5000StateError(
                "run state directory cannot be created securely"
            ) from error
        return run_directory

    def _state_path(self, run_id: str) -> Path:
        return self._run_directory(run_id) / "state.json"

    def load(self, run_id: str) -> Grid5000State | None:
        """Load one state document, rejecting ambiguous filesystem state."""

        self._validate_run_id(run_id)
        _reject_symlink_components(self.root)
        if self.root.is_symlink():
            raise Grid5000StateError("state root cannot be a symlink")
        run_directory = self._run_directory(run_id)
        if run_directory.is_symlink():
            raise Grid5000StateError("run state directory cannot be a symlink")
        if not run_directory.exists():
            return None
        _check_managed_mode(
            run_directory, 0o700, "run state directory permissions are unsafe"
        )
        lock_path = run_directory / ".intent.lock"
        if lock_path.exists() or lock_path.is_symlink():
            raise Grid5000StateError("submission state is ambiguous")
        state_path = run_directory / "state.json"
        if not state_path.exists() or state_path.is_symlink():
            raise Grid5000StateError("run state document is missing or unsafe")
        _check_managed_mode(
            state_path, 0o600, "run state document permissions are unsafe"
        )
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise Grid5000StateError("run state document cannot be read") from error
        if not isinstance(payload, Mapping):
            raise Grid5000StateError("run state document must be an object")
        state = Grid5000State.from_dict(payload)
        if state.identity.run_id != run_id:
            raise Grid5000StateError("run state identity does not match its directory")
        return state

    def _write_atomic(self, run_directory: Path, state: Grid5000State) -> None:
        temporary_path: Path | None = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                dir=run_directory, prefix=".state-", suffix=".tmp"
            )
            temporary_path = Path(temporary_name)
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(state.to_dict(), handle, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, run_directory / "state.json")
            temporary_path = None
            directory_descriptor = os.open(run_directory, os.O_RDONLY)
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
        except OSError as error:
            raise Grid5000StateError(
                "run state document cannot be written securely"
            ) from error
        finally:
            if temporary_path is not None:
                with suppress(OSError):
                    temporary_path.unlink()

    def save(self, state: Grid5000State) -> None:
        """Atomically write a state document, allowing a phase update."""

        run_directory = self._ensure_run_directory(state.identity.run_id)
        lock_path = run_directory / ".intent.lock"
        if lock_path.exists() or lock_path.is_symlink():
            raise Grid5000StateError("submission state is ambiguous")
        self._write_atomic(run_directory, state)

    def create_submitting(self, state: Grid5000State) -> None:
        """Claim a run ID and write its pre-submit intent exactly once."""

        if state.phase != "submitting":
            raise Grid5000StateError("submission intent must be in submitting phase")
        run_directory = self._ensure_run_directory(state.identity.run_id)
        state_path = run_directory / "state.json"
        lock_path = run_directory / ".intent.lock"
        if state_path.exists() or state_path.is_symlink():
            raise Grid5000StateError("run already has durable submission state")
        try:
            descriptor = os.open(
                lock_path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                0o600,
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(state.identity.run_id)
                handle.flush()
                os.fsync(handle.fileno())
            self._write_atomic(run_directory, state)
            lock_path.unlink()
        except FileExistsError as error:
            raise Grid5000StateError("submission state is ambiguous") from error
        except OSError as error:
            raise Grid5000StateError(
                "submission intent cannot be recorded securely"
            ) from error


@dataclass(frozen=True, slots=True)
class Grid5000Submission:
    """Result of either planning or one explicitly executed submission."""

    plan: Grid5000Plan
    executed: bool
    job_id: int | None = None


def parse_job_id(output: str) -> int:
    """Extract exactly one positive OAR job ID from submission output."""

    matches = _JOB_ID_PATTERN.findall(output.strip())
    if len(matches) != 1:
        raise Grid5000ConfigurationError("submission did not return one job ID")
    return int(matches[0])


class Grid5000Operator:
    """Plan by default and submit only after an explicit execution gate."""

    def __init__(
        self,
        plan: Grid5000Plan,
        *,
        state_store: Grid5000StateStore | None = None,
        runner: CommandRunner | None = None,
        command_timeout: float = COMMAND_TIMEOUT_SECONDS,
    ) -> None:
        if command_timeout <= 0:
            raise Grid5000ConfigurationError("command_timeout must be positive")
        self.plan = plan
        self.state_store = state_store or Grid5000StateStore()
        self.runner = runner or SubprocessCommandRunner()
        self.command_timeout = command_timeout

    def _run_checked(self, argv: Sequence[str], label: str) -> CommandResult:
        try:
            result = self.runner(argv, timeout=self.command_timeout)
        except Grid5000ExecutionError:
            raise
        except Exception as error:
            raise Grid5000ExecutionError(
                f"{label} command could not complete"
            ) from error
        if result.returncode != 0:
            raise Grid5000ExecutionError(
                f"{label} command failed with exit code {result.returncode}"
            )
        return result

    def submit(self, *, execute: bool = False) -> Grid5000Submission:
        """Return a plan, or execute one guarded, non-retrying submission."""

        if not execute:
            return Grid5000Submission(plan=self.plan, executed=False)

        existing = self.state_store.load(self.plan.identity.run_id)
        if existing is not None:
            if existing.phase == "submitted":
                raise Grid5000StateError("run already has a recorded submission")
            raise Grid5000StateError("run has an ambiguous submitting state")

        self._run_checked(self.plan.remote_checkout_command, "remote checkout")
        self._run_checked(self.plan.policy_site_command, "site policy")
        self._run_checked(self.plan.policy_total_command, "total policy")
        quota_result = self._run_checked(self.plan.quota_command, "home quota")
        quota = parse_quota_output(quota_result.stdout)
        if quota.soft_headroom_bytes < MINIMUM_HOME_HEADROOM_BYTES:
            raise Grid5000ExecutionError(
                "Grid'5000 home soft quota has insufficient safe headroom"
            )

        intent = Grid5000State(
            identity=self.plan.identity,
            phase="submitting",
            scheduler_command=self.plan.scheduler_command,
            submission_command=self.plan.submission_command,
        )
        self.state_store.create_submitting(intent)
        result = self._run_checked(self.plan.submission_command, "OAR submission")
        try:
            job_id = parse_job_id(result.stdout)
        except Grid5000ConfigurationError as error:
            raise Grid5000ExecutionError(
                "OAR submission returned an invalid job ID"
            ) from error
        self.state_store.save(
            Grid5000State(
                identity=self.plan.identity,
                phase="submitted",
                scheduler_command=self.plan.scheduler_command,
                job_id=job_id,
                submission_command=self.plan.submission_command,
            )
        )
        return Grid5000Submission(plan=self.plan, executed=True, job_id=job_id)


__all__ = [
    "COMMAND_TIMEOUT_SECONDS",
    "CONTAINER_CHECKOUT",
    "CONTAINER_DATA_ROOT",
    "CONTAINER_HOME",
    "ContainerRuntime",
    "DEFAULT_DAY_WALLTIME_SECONDS",
    "GRID5000_DATASET_REVISION",
    "MAX_DAY_WALLTIME_SECONDS",
    "MAX_WALLTIME_SECONDS",
    "MINIMUM_CUDA_CAPABILITY",
    "MINIMUM_HOME_HEADROOM_BYTES",
    "CommandResult",
    "CommandRunner",
    "Grid5000Allocation",
    "Grid5000ConfigurationError",
    "Grid5000ExecutionError",
    "Grid5000Operator",
    "Grid5000Phase",
    "Grid5000Plan",
    "Grid5000RunIdentity",
    "Grid5000State",
    "Grid5000StateError",
    "Grid5000StateStore",
    "Grid5000Submission",
    "QuotaUsage",
    "REMOTE_CHECKOUT_SUBDIRECTORY",
    "REMOTE_DATA_SUBDIRECTORY",
    "REMOTE_ENVIRONMENT_SUBDIRECTORY",
    "REMOTE_RUNS_SUBDIRECTORY",
    "SUPPORTED_TASK_NAMES",
    "SubprocessCommandRunner",
    "parse_job_id",
    "parse_quota_output",
]
