"""Validated final model publication to the project Hugging Face repository."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any


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


__all__ = [
    "ModelPublicationError",
    "ModelPublicationResult",
    "ensure_model_repository",
    "publish_model_directory",
]
