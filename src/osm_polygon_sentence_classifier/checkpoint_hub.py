"""Identity-bound checkpoint discovery and restoration from the model Hub."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path, PurePosixPath
from typing import Any

from .checkpointing import CheckpointInfo, find_complete_checkpoint
from .huggingface_http import configure_huggingface_http
from .publication import _model_artifact_prefix

_CHECKPOINT_ROOT_PATTERN = re.compile(
    r"^(?P<root>.+)/step-(?P<step>[1-9][0-9]*)/(?P<name>[^/]+)$"
)
_WEIGHT_PATTERN = re.compile(
    r"(?:model|pytorch_model)(?:-[0-9]{5}-of-[0-9]{5})?\.(?:bin|safetensors)$"
)
_REQUIRED_FILES = frozenset(
    {
        "checkpoint-manifest.json",
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    }
)


class HubCheckpointError(RuntimeError):
    """Raised when a Hub checkpoint cannot be safely restored."""


@dataclass(frozen=True, slots=True)
class PublishedCheckpoint:
    """One complete checkpoint published under a run-specific Hub prefix."""

    repository_id: str
    prefix: str
    step: int
    files: tuple[str, ...]


ManifestLoader = Callable[[str, str], Path]


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
        raise HubCheckpointError(
            "checkpoint identity is not JSON-compatible"
        ) from error


def _default_api() -> Any:
    try:
        configure_huggingface_http()
        return import_module("huggingface_hub").HfApi()
    except Exception as error:
        raise HubCheckpointError(
            "Hugging Face checkpoint access is unavailable"
        ) from error


def _default_manifest_loader(repository_id: str, path: str) -> Path:
    try:
        configure_huggingface_http()
        hub = import_module("huggingface_hub")
        download = getattr(hub, "hf_hub_download", None)
        if not callable(download):
            raise HubCheckpointError("Hugging Face checkpoint download is unavailable")
        downloaded = download(
            repo_id=repository_id,
            filename=path,
            repo_type="model",
        )
    except HubCheckpointError:
        raise
    except Exception as error:
        raise HubCheckpointError("checkpoint manifest download failed") from error
    if not isinstance(downloaded, (str, Path)):
        raise HubCheckpointError(
            "checkpoint manifest download returned an invalid path"
        )
    return Path(downloaded)


def _complete_step_files(
    paths: tuple[str, ...], *, root: str
) -> dict[int, tuple[str, ...]]:
    grouped: dict[int, list[str]] = {}
    for path in paths:
        match = _CHECKPOINT_ROOT_PATTERN.fullmatch(path)
        if match is None or match.group("root") != root:
            continue
        grouped.setdefault(int(match.group("step")), []).append(match.group("name"))
    complete: dict[int, tuple[str, ...]] = {}
    for step, names in grouped.items():
        unique_names = frozenset(names)
        if not unique_names >= _REQUIRED_FILES or not any(
            _WEIGHT_PATTERN.fullmatch(name) for name in unique_names
        ):
            continue
        complete[step] = tuple(sorted(unique_names))
    return complete


def _manifest_matches(
    path: Path,
    *,
    identity: Mapping[str, object],
    step: int,
) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(payload, Mapping)
        and payload.get("schema_version") == 1
        and payload.get("global_step") == step
        and _canonical_json(payload.get("identity")) == _canonical_json(dict(identity))
    )


def latest_published_checkpoint(
    identity: Mapping[str, object],
    *,
    repository_id: str,
    hub_api: Any | None = None,
    manifest_loader: ManifestLoader | None = None,
) -> PublishedCheckpoint | None:
    """Return the newest complete Hub checkpoint matching ``identity``."""

    if not isinstance(repository_id, str) or not repository_id.strip():
        raise HubCheckpointError("model repository ID is invalid")
    prefix = _model_artifact_prefix(identity)
    root = f"{prefix}/checkpoints"
    api = hub_api or _default_api()
    try:
        entries = tuple(
            entry.path
            for entry in api.list_repo_tree(
                repo_id=repository_id,
                repo_type="model",
                path_in_repo=root,
                recursive=True,
            )
        )
    except Exception as error:
        raise HubCheckpointError(
            "published checkpoint inventory could not be read"
        ) from error
    paths = tuple(path for path in entries if isinstance(path, str))
    complete_steps = _complete_step_files(paths, root=root)
    if not complete_steps:
        return None
    load_manifest = manifest_loader or _default_manifest_loader
    for step in sorted(complete_steps, reverse=True):
        manifest_path = f"{root}/step-{step}/checkpoint-manifest.json"
        try:
            manifest = load_manifest(repository_id, manifest_path)
        except HubCheckpointError:
            continue
        if not _manifest_matches(manifest, identity=identity, step=step):
            continue
        return PublishedCheckpoint(
            repository_id=repository_id,
            prefix=prefix,
            step=step,
            files=complete_steps[step],
        )
    return None


def _reject_symlinks(root: Path) -> None:
    if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise HubCheckpointError("downloaded checkpoint contains a symlink")


def restore_published_checkpoint(
    output_directory: str | Path,
    *,
    identity: Mapping[str, object],
    repository_id: str,
) -> CheckpointInfo:
    """Download the newest complete Hub checkpoint into a worker output dir."""

    output = Path(output_directory)
    if not output.is_absolute() or output.is_symlink():
        raise HubCheckpointError("checkpoint output directory is unsafe")
    checkpoint = latest_published_checkpoint(
        identity,
        repository_id=repository_id,
    )
    if checkpoint is None:
        raise HubCheckpointError("no complete published checkpoint matches the run")
    try:
        configure_huggingface_http()
        hub = import_module("huggingface_hub")
        download = getattr(hub, "snapshot_download", None)
        if not callable(download):
            raise HubCheckpointError("Hugging Face checkpoint download is unavailable")
        output.mkdir(parents=True, exist_ok=True, mode=0o700)
        with tempfile.TemporaryDirectory(
            prefix=".hub-checkpoint-",
            dir=str(output.parent),
        ) as temporary:
            downloaded = download(
                repo_id=checkpoint.repository_id,
                repo_type="model",
                allow_patterns=[
                    f"{checkpoint.prefix}/checkpoints/step-{checkpoint.step}/*"
                ],
                local_dir=temporary,
            )
            if not isinstance(downloaded, (str, Path)):
                raise HubCheckpointError("checkpoint download returned an invalid path")
            source = Path(downloaded) / PurePosixPath(
                checkpoint.prefix,
                "checkpoints",
                f"step-{checkpoint.step}",
            )
            if not source.is_dir() or not source.is_relative_to(Path(temporary)):
                raise HubCheckpointError("checkpoint download path is unsafe")
            _reject_symlinks(source)
            destination = output / f"checkpoint-{checkpoint.step}"
            if destination.exists() or destination.is_symlink():
                raise HubCheckpointError("checkpoint destination already exists")
            shutil.copytree(source, destination)
            restored = find_complete_checkpoint(
                destination,
                identity=identity,
            )
            if restored is None:
                raise HubCheckpointError("downloaded checkpoint failed validation")
            return restored
    except HubCheckpointError:
        raise
    except Exception as error:
        raise HubCheckpointError("published checkpoint restoration failed") from error


__all__ = [
    "HubCheckpointError",
    "PublishedCheckpoint",
    "latest_published_checkpoint",
    "restore_published_checkpoint",
]
