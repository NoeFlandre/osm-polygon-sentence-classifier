"""Identity-bound checkpoint writing and safe checkpoint discovery."""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CHECKPOINT_MANIFEST_FILENAME = "checkpoint-manifest.json"
_CHECKPOINT_NAME_PATTERN = re.compile(r"checkpoint-([1-9][0-9]*)$")
_WEIGHT_NAME_PATTERN = re.compile(
    r"(?:model|pytorch_model)(?:-[0-9]{5}-of-[0-9]{5})?\.(?:bin|safetensors)$"
)
_CANONICAL_JSON_OPTIONS: dict[str, Any] = {
    "allow_nan": False,  # pragma: no mutate
    "ensure_ascii": False,  # pragma: no mutate
}
_ATOMIC_JSON_OPTIONS: dict[str, Any] = {"allow_nan": False}  # pragma: no mutate


class CheckpointError(RuntimeError):
    """Raised when checkpoint evidence violates the continuation contract."""


@dataclass(frozen=True, slots=True)
class CheckpointInfo:
    """One complete checkpoint that matches an immutable run identity."""

    path: Path
    global_step: int


def _canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            **_CANONICAL_JSON_OPTIONS,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise CheckpointError("checkpoint identity is not JSON-compatible") from error


def _contains_symlink(path: Path) -> bool:
    if not path.is_absolute():
        raise CheckpointError("checkpoint path must be absolute")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            return True
    return False


def _regular_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def _is_safe_checkpoint_directory(checkpoint_directory: Path) -> bool:
    return (
        checkpoint_directory.is_dir()
        and not checkpoint_directory.is_symlink()
        and not _contains_symlink(checkpoint_directory)
    )


def _has_checkpoint_files(checkpoint_directory: Path) -> bool:
    required = (
        CHECKPOINT_MANIFEST_FILENAME,
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    )
    if not all(_regular_file(checkpoint_directory / name) for name in required):
        return False
    return any(
        _regular_file(path) and _WEIGHT_NAME_PATTERN.fullmatch(path.name)
        for path in checkpoint_directory.iterdir()
    )


def _metadata_matches(
    manifest_payload: object,
    trainer_state: object,
    *,
    global_step: int,
    expected_identity: str,
) -> bool:
    if not isinstance(manifest_payload, Mapping) or not isinstance(
        trainer_state, Mapping
    ):
        return False
    if not _metadata_steps_match(manifest_payload, trainer_state, global_step):
        return False
    return _metadata_identity_matches(manifest_payload, expected_identity)


def _metadata_steps_match(
    manifest_payload: Mapping[Any, Any],
    trainer_state: Mapping[Any, Any],
    global_step: int,
) -> bool:
    if manifest_payload.get("schema_version") != 1:
        return False
    if manifest_payload.get("global_step") != global_step:
        return False
    return trainer_state.get("global_step") == global_step


def _metadata_identity_matches(
    manifest_payload: Mapping[Any, Any], expected_identity: str
) -> bool:
    try:
        identity = _canonical_json(manifest_payload.get("identity"))
    except CheckpointError:
        return False
    return identity == expected_identity


def _require_step(checkpoint_directory: Path, global_step: int) -> None:
    _require_positive_step(global_step)
    match = _CHECKPOINT_NAME_PATTERN.fullmatch(checkpoint_directory.name)
    if match is None or int(match.group(1)) != global_step:
        raise CheckpointError("checkpoint directory name does not match global_step")


def _require_positive_step(global_step: object) -> None:
    if (
        isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or global_step <= 0
    ):
        raise CheckpointError("checkpoint global_step must be positive")


def write_checkpoint_manifest(
    checkpoint_directory: Path,
    *,
    identity: Mapping[str, object],
    global_step: int,
) -> Path:
    """Atomically record identity after Trainer has fully saved a checkpoint."""

    directory = Path(checkpoint_directory)
    _require_real_checkpoint_directory(directory)
    _require_step(directory, global_step)
    payload = _checkpoint_manifest_payload(identity, global_step)
    manifest = directory / CHECKPOINT_MANIFEST_FILENAME
    temporary = directory / f".{CHECKPOINT_MANIFEST_FILENAME}.tmp"
    _require_real_manifest_paths(manifest, temporary)
    _write_manifest_atomically(temporary, manifest, payload)
    return manifest


def _require_real_checkpoint_directory(directory: Path) -> None:
    if (
        not directory.is_absolute()
        or _contains_symlink(directory)
        or not directory.is_dir()
        or directory.is_symlink()
    ):
        raise CheckpointError("checkpoint directory must be a real absolute directory")


def _checkpoint_manifest_payload(
    identity: Mapping[str, object], global_step: int
) -> dict[str, object]:
    return {
        "global_step": global_step,
        "identity": json.loads(_canonical_json(dict(identity))),
        "schema_version": 1,
    }


def _require_real_manifest_paths(manifest: Path, temporary: Path) -> None:
    if manifest.is_symlink() or temporary.is_symlink():
        raise CheckpointError("checkpoint manifest cannot be a symlink")


def _write_manifest_atomically(
    temporary: Path, manifest: Path, payload: Mapping[str, object]
) -> None:
    try:
        temporary.write_text(
            json.dumps(payload, **_ATOMIC_JSON_OPTIONS, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.chmod(temporary, 0o600)
        os.replace(temporary, manifest)
    except (OSError, TypeError, ValueError) as error:
        temporary.unlink(missing_ok=True)
        raise CheckpointError("checkpoint manifest cannot be written") from error


def _is_complete_checkpoint(
    checkpoint_directory: Path,
    *,
    expected_identity: str,
) -> CheckpointInfo | None:
    if not _is_safe_checkpoint_directory(checkpoint_directory):
        return None
    global_step = _checkpoint_step(checkpoint_directory)
    if global_step is None:
        return None
    metadata = _read_checkpoint_metadata(checkpoint_directory)
    if metadata is None:
        return None
    manifest_payload, trainer_state = metadata
    if not _metadata_matches(
        manifest_payload,
        trainer_state,
        global_step=global_step,
        expected_identity=expected_identity,
    ):
        return None
    return CheckpointInfo(path=checkpoint_directory, global_step=global_step)


def _checkpoint_step(checkpoint_directory: Path) -> int | None:
    match = _CHECKPOINT_NAME_PATTERN.fullmatch(checkpoint_directory.name)
    return None if match is None else int(match.group(1))


def _read_checkpoint_metadata(
    checkpoint_directory: Path,
) -> tuple[object, object] | None:
    try:
        if not _has_checkpoint_files(checkpoint_directory):
            return None
        manifest_payload = json.loads(
            (checkpoint_directory / CHECKPOINT_MANIFEST_FILENAME).read_text(
                encoding="utf-8"
            )
        )
        trainer_state = json.loads(
            (checkpoint_directory / "trainer_state.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return None
    return manifest_payload, trainer_state


def find_complete_checkpoint(
    checkpoint_directory: Path,
    *,
    identity: Mapping[str, object],
) -> CheckpointInfo | None:
    """Return an exact identity-matching checkpoint, if it is complete."""

    directory = Path(checkpoint_directory)
    if not directory.is_absolute():
        raise CheckpointError("checkpoint directory must not be symlinked")
    if directory.is_symlink():
        return None
    if _contains_symlink(directory):
        raise CheckpointError("checkpoint directory must not be symlinked")
    expected_identity = _canonical_json(dict(identity))
    return _is_complete_checkpoint(
        directory,
        expected_identity=expected_identity,
    )


def find_latest_complete_checkpoint(
    output_directory: Path,
    *,
    identity: Mapping[str, object],
) -> CheckpointInfo | None:
    """Return the newest complete identity-matching checkpoint, if any."""

    output = Path(output_directory)
    _require_safe_output_directory(output)
    if not output.exists():
        return None
    if not output.is_dir():
        raise CheckpointError("checkpoint output path must be a directory")
    candidates = _checkpoint_candidates(output)
    complete = _complete_candidates(candidates, identity)
    return max(complete, key=lambda item: item.global_step) if complete else None


def _require_safe_output_directory(output: Path) -> None:
    if not output.is_absolute() or _contains_symlink(output):
        raise CheckpointError("checkpoint output directory must not be symlinked")


def _checkpoint_candidates(output: Path) -> tuple[Path, ...]:
    try:
        return tuple(
            path
            for path in output.iterdir()
            if _CHECKPOINT_NAME_PATTERN.fullmatch(path.name)
        )
    except OSError as error:
        raise CheckpointError("checkpoint output directory cannot be read") from error


def _complete_candidates(
    candidates: tuple[Path, ...], identity: Mapping[str, object]
) -> tuple[CheckpointInfo, ...]:
    return tuple(
        result
        for candidate in candidates
        if (
            result := find_complete_checkpoint(
                candidate,
                identity=identity,
            )
        )
        is not None
    )


__all__ = [
    "CHECKPOINT_MANIFEST_FILENAME",
    "CheckpointError",
    "CheckpointInfo",
    "find_complete_checkpoint",
    "find_latest_complete_checkpoint",
    "write_checkpoint_manifest",
]
