"""Validated model and checkpoint publication to the project Hub repository."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from .checkpointing import CheckpointError, find_complete_checkpoint
from .config import SOURCE_DATASET_ID, TARGET_MODEL_REPOSITORY_ID


class ModelPublicationError(RuntimeError):
    """Raised when a completed model cannot be published safely."""


@dataclass(frozen=True, slots=True)
class ModelPublicationResult:
    """Facts returned by one final model-repository commit."""

    repository_id: str
    commit_id: str
    commit_url: str
    files: tuple[str, ...]


OperationFactory = Callable[..., Any]

_WEIGHT_PATTERN = re.compile(
    r"(?:model|pytorch_model)(?:-\d{5}-of-\d{5})?\.(?:bin|safetensors)$"
)
_ALLOWED_ROOT_NAMES = frozenset(
    {
        "added_tokens.json",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "pytorch_model.bin.index.json",
        "README.md",
        "model.safetensors.index.json",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "training_args.bin",
        "vocab.json",
        "vocab.txt",
    }
)
_ALLOWED_CHECKPOINT_NAMES = frozenset(
    {
        "README.md",
        "checkpoint-manifest.json",
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "optimizer.pt",
        "pytorch_model.bin.index.json",
        "rng_state.pth",
        "scaler.pt",
        "scheduler.pt",
        "trainer_state.json",
    }
)


def _require_non_blank(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or "\n" in value
        or "\r" in value
    ):
        raise ModelPublicationError(f"{field} must be a non-blank single-line string")
    return value


def _default_hub_api() -> Any:
    try:
        hub = import_module("huggingface_hub")
        return hub.HfApi()
    except Exception as error:
        raise ModelPublicationError(
            "Hugging Face publication requires the training dependencies"
        ) from error


def _default_operation_factory() -> OperationFactory:
    try:
        hub = import_module("huggingface_hub")
        return hub.CommitOperationAdd
    except Exception as error:
        raise ModelPublicationError(
            "Hugging Face publication requires the training dependencies"
        ) from error


_SENSITIVE_KEY_PARTS = ("credential", "password", "secret", "token")


def _safe_scalar(value: object) -> bool:
    if value is None or isinstance(value, (bool, int, str)):
        return True
    return isinstance(value, float) and math.isfinite(value)


def _safe_scalar_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    safe: dict[str, object] = {}
    for key, item in value.items():
        if (
            isinstance(key, str)
            and not any(part in key.lower() for part in _SENSITIVE_KEY_PARTS)
            and _safe_scalar(item)
        ):
            safe[key] = item
    return dict(sorted(safe.items()))


def _safe_line(value: object, fallback: str) -> str:
    if (
        isinstance(value, str)
        and value.strip()
        and "\n" not in value
        and "\r" not in value
    ):
        return value.strip()
    return fallback


def render_model_card(
    *,
    identity: Mapping[str, object],
    training_metrics: Mapping[str, object] | None = None,
    checkpoint_step: int | None = None,
    trackio_space_id: str | None = None,
) -> str:
    """Render a deterministic, credential-free model card from run facts."""

    task_name = _safe_line(identity.get("task_name"), "landuse")
    model_name = _safe_line(identity.get("model_name_or_path"), "not recorded")
    model_revision = _safe_line(identity.get("model_revision"), "not pinned")
    dataset_revision = _safe_line(identity.get("dataset_revision"), "not recorded")
    source_commit = _safe_line(identity.get("source_commit"), "not recorded")
    training_config = _safe_scalar_mapping(identity.get("training_config"))
    metrics = _safe_scalar_mapping(training_metrics)
    if (
        isinstance(checkpoint_step, int)
        and not isinstance(checkpoint_step, bool)
        and checkpoint_step >= 0
    ):
        progress = f"checkpoint at step {checkpoint_step}"
    else:
        progress = "final model"
    trackio_link = None
    if (
        isinstance(trackio_space_id, str)
        and trackio_space_id.strip()
        and "\n" not in trackio_space_id
        and "\r" not in trackio_space_id
    ):
        trackio_link = "https://huggingface.co/spaces/" + trackio_space_id.strip()

    tracking_section = (
        f"[Open the Trackio dashboard]({trackio_link}). "
        "Metrics are published as static snapshots after complete checkpoints "
        "and final publication."
        if trackio_link is not None
        else "Trackio was not enabled for this run."
    )
    return (
        "---\n"
        "library_name: transformers\n"
        "pipeline_tag: text-classification\n"
        "tags:\n"
        "- landuse\n"
        "- text-classification\n"
        "---\n\n"
        "# OSM Polygon Landuse Sentence Classifier\n\n"
        f"This model classifies whether a sentence is relevant to landuse. "
        f"The recorded training status is **{progress}**.\n\n"
        "## Training data\n\n"
        f"- Dataset: [{SOURCE_DATASET_ID}]"
        f"(https://huggingface.co/datasets/{SOURCE_DATASET_ID})\n"
        f"- Dataset revision: `{dataset_revision}`\n"
        f"- Task: `{task_name}`\n"
        "- Labels: `no` (0), `yes` (1)\n\n"
        "## Model and provenance\n\n"
        f"- Base model: `{model_name}`\n"
        f"- Base-model revision: `{model_revision}`\n"
        f"- Source-code commit: `{source_commit}`\n"
        f"- Model repository: [{TARGET_MODEL_REPOSITORY_ID}]"
        f"(https://huggingface.co/{TARGET_MODEL_REPOSITORY_ID})\n\n"
        "## Training configuration\n\n"
        "```json\n"
        f"{json.dumps(training_config, ensure_ascii=False, sort_keys=True, indent=2)}\n"
        "```\n\n"
        "## Metrics\n\n"
        "```json\n"
        f"{json.dumps(metrics, ensure_ascii=False, sort_keys=True, indent=2)}\n"
        "```\n\n"
        "## Experiment tracking\n\n"
        f"{tracking_section}\n"
    )


def ensure_model_repository(
    repository_id: object,
    *,
    hub_api: Any | None = None,
) -> None:
    """Create the dedicated model repository without uploading an artifact."""

    repository = _require_non_blank(repository_id, "repository_id")
    try:
        api = hub_api or _default_hub_api()
        api.create_repo(
            repo_id=repository,
            repo_type="model",
            exist_ok=True,
        )
    except ModelPublicationError:
        raise
    except Exception as error:
        raise ModelPublicationError(
            "Hugging Face model repository setup failed"
        ) from error


def _final_model_files(directory: Path) -> tuple[Path, ...]:
    if not directory.exists() or not directory.is_dir() or directory.is_symlink():
        raise ModelPublicationError("model output must be a real directory")
    files = tuple(sorted(path for path in directory.iterdir() if path.is_file()))
    names = {path.name for path in files}
    if "config.json" not in names:
        raise ModelPublicationError("model output is missing config.json")
    if not any(_WEIGHT_PATTERN.fullmatch(name) for name in names):
        raise ModelPublicationError("model output is missing model weights")
    if not names & {
        "tokenizer.json",
        "tokenizer_config.json",
        "spiece.model",
        "sentencepiece.bpe.model",
        "tokenizer.model",
        "vocab.json",
        "vocab.txt",
    }:
        raise ModelPublicationError("model output is missing tokenizer files")
    selected = tuple(
        path
        for path in files
        if path.name in _ALLOWED_ROOT_NAMES or _WEIGHT_PATTERN.fullmatch(path.name)
    )
    if not selected:
        raise ModelPublicationError("model output contains no publishable files")
    return selected


def _commit_facts(info: Any) -> tuple[str, str]:
    commit_id = getattr(info, "oid", None)
    commit_url = getattr(info, "commit_url", None)
    if not isinstance(commit_id, str) or not commit_id.strip():
        raise ModelPublicationError("Hugging Face returned an invalid model commit ID")
    if not isinstance(commit_url, str) or not commit_url.strip():
        raise ModelPublicationError("Hugging Face returned an invalid model commit URL")
    return commit_id, commit_url


def publish_model_directory(
    directory: str | Path,
    repository_id: object,
    *,
    hub_api: Any | None = None,
    operation_factory: OperationFactory | None = None,
) -> ModelPublicationResult:
    """Validate and commit only the final top-level model files."""

    repository = _require_non_blank(repository_id, "repository_id")
    output_directory = Path(directory)
    files = _final_model_files(output_directory)
    factory = operation_factory or _default_operation_factory()
    operations: list[Any] = []
    try:
        for path in files:
            operations.append(
                factory(
                    path_in_repo=path.name,
                    path_or_fileobj=str(path),
                )
            )
    except Exception as error:
        raise ModelPublicationError(
            "Hugging Face model operations could not be constructed"
        ) from error

    api = hub_api or _default_hub_api()
    try:
        info = api.create_commit(
            repo_id=repository,
            repo_type="model",
            operations=operations,
            commit_message="Publish completed landuse classifier",
            revision="main",
        )
    except Exception as error:
        raise ModelPublicationError("Hugging Face model publication failed") from error
    commit_id, commit_url = _commit_facts(info)
    return ModelPublicationResult(
        repository_id=repository,
        commit_id=commit_id,
        commit_url=commit_url,
        files=tuple(path.name for path in files),
    )


def _complete_checkpoint_files(
    directory: Path,
    *,
    identity: Mapping[str, object],
) -> tuple[Path, ...]:
    try:
        selected = find_complete_checkpoint(
            directory,
            identity=identity,
        )
    except CheckpointError as error:
        raise ModelPublicationError("checkpoint evidence is invalid") from error
    if selected is None:
        raise ModelPublicationError("model output is not a complete checkpoint")
    try:
        files = tuple(
            sorted(
                path
                for path in directory.iterdir()
                if path.is_file()
                and not path.is_symlink()
                and (
                    path.name in _ALLOWED_CHECKPOINT_NAMES
                    or _WEIGHT_PATTERN.fullmatch(path.name)
                )
            )
        )
    except OSError as error:
        raise ModelPublicationError("checkpoint output cannot be read") from error
    if not files:
        raise ModelPublicationError("checkpoint output contains no files")
    return files


def _checkpoint_path_in_repo(path: Path, checkpoint: Path) -> str:
    if path == checkpoint / "README.md":
        return "README.md"
    return f"checkpoints/last-checkpoint/{path.name}"


def publish_checkpoint_directory(
    directory: str | Path,
    repository_id: object,
    *,
    identity: Mapping[str, object],
    hub_api: Any | None = None,
    operation_factory: OperationFactory | None = None,
) -> ModelPublicationResult:
    """Commit one complete checkpoint as the repository's latest snapshot."""

    repository = _require_non_blank(repository_id, "repository_id")
    checkpoint = Path(directory)
    files = _complete_checkpoint_files(checkpoint, identity=identity)
    factory = operation_factory or _default_operation_factory()
    operations: list[Any] = []
    try:
        for path in files:
            operations.append(
                factory(
                    path_in_repo=_checkpoint_path_in_repo(path, checkpoint),
                    path_or_fileobj=str(path),
                )
            )
    except Exception as error:
        raise ModelPublicationError(
            "Hugging Face checkpoint operations could not be constructed"
        ) from error

    api = hub_api or _default_hub_api()
    try:
        info = api.create_commit(
            repo_id=repository,
            repo_type="model",
            operations=operations,
            commit_message=f"Publish checkpoint {checkpoint.name}",
            revision="main",
        )
    except Exception as error:
        raise ModelPublicationError(
            "Hugging Face checkpoint publication failed"
        ) from error
    commit_id, commit_url = _commit_facts(info)
    return ModelPublicationResult(
        repository_id=repository,
        commit_id=commit_id,
        commit_url=commit_url,
        files=tuple(_checkpoint_path_in_repo(path, checkpoint) for path in files),
    )


__all__ = [
    "ModelPublicationError",
    "ModelPublicationResult",
    "ensure_model_repository",
    "publish_checkpoint_directory",
    "publish_model_directory",
    "render_model_card",
]
