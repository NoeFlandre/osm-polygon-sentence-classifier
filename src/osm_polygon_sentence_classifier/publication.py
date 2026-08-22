"""Validated model and checkpoint publication to the project Hub repository."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path, PurePosixPath
from typing import Protocol, cast

from .checkpointing import CheckpointError, find_complete_checkpoint
from .config import SOURCE_DATASET_ID, TARGET_MODEL_REPOSITORY_ID
from .huggingface_http import configure_huggingface_http
from .tracking import (
    TRACKIO_BUCKET_ID,
    TRACKIO_SPACE_ID,
    V2_TRACKIO_BUCKET_ID,
    V2_TRACKIO_SPACE_ID,
)


class ModelPublicationError(RuntimeError):
    """Raised when a completed model cannot be published safely."""


@dataclass(frozen=True, slots=True)
class ModelPublicationResult:
    """Facts returned by one final model-repository commit."""

    repository_id: str
    commit_id: str
    commit_url: str
    files: tuple[str, ...]


OperationFactory = Callable[..., object]


class _HubCommitProtocol(Protocol):
    def create_commit(self, **kwargs: object) -> object: ...


class _HubRepositoryProtocol(Protocol):
    def create_repo(self, **kwargs: object) -> object: ...


class _HubApiProtocol(_HubCommitProtocol, _HubRepositoryProtocol, Protocol):
    pass


_WEIGHT_PATTERN = re.compile(
    r"(?:model|pytorch_model)(?:-\d{5}-of-\d{5})?\.(?:bin|safetensors)$"
)
_CHECKPOINT_NAME_PATTERN = re.compile(r"checkpoint-([1-9][0-9]*)$")
_RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{8,64}")
_MODEL_EXPERIMENT_ROOT = "experiments"
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


def _default_hub_api() -> _HubApiProtocol:
    try:
        configure_huggingface_http()
        hub = import_module("huggingface_hub")
        return cast(_HubApiProtocol, hub.HfApi())
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


def _safe_metadata_item(key: object, value: object) -> tuple[str, object] | None:
    if not isinstance(key, str):
        return None
    if any(part in key.lower() for part in _SENSITIVE_KEY_PARTS):
        return None
    if not _safe_scalar(value):
        return None
    return key, value


def _safe_scalar_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        return {}
    safe: dict[str, object] = {}
    for key, item in value.items():
        safe_item = _safe_metadata_item(key, item)
        if safe_item is not None:
            safe_key, safe_value = safe_item
            safe[safe_key] = safe_value
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


def _slug(value: object, fallback: str) -> str:
    text = _safe_line(value, fallback)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-._")
    return slug[:80] or fallback


def _task_metadata(identity: Mapping[str, object]) -> tuple[str, str, str]:
    """Return the safe task name, display label, and Hub tag for one run."""

    task_name = _safe_line(identity.get("task_name"), "landuse")
    if task_name == "place-relevance-v2":
        return task_name, "place relevance", "place-relevance"
    return task_name, task_name.replace("-", " "), _slug(task_name, "task")


def _publication_run_id(identity: Mapping[str, object]) -> str:
    supplied = identity.get("run_id")
    if isinstance(supplied, str) and _RUN_ID_PATTERN.fullmatch(supplied):
        return supplied
    canonical = json.dumps(
        dict(identity),
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[
        :20
    ]  # pragma: no mutate


def _training_config_value(
    identity: Mapping[str, object],
    key: str,
) -> object:
    training_config = identity.get("training_config")
    if isinstance(training_config, Mapping):
        return training_config.get(key)
    return None


def _valid_artifact_namespace(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    if not value.strip():
        return None
    parts = tuple(value.strip("/").split("/"))
    if any(part in {"", ".", ".."} for part in parts):
        return None
    return "/".join(parts)


def _model_artifact_prefix(identity: Mapping[str, object]) -> str:
    namespace = _valid_artifact_namespace(
        _training_config_value(identity, "artifact_namespace")
    )
    if namespace is not None:
        return f"{namespace}/run-{_publication_run_id(identity)}"
    experiment = _slug(_training_config_value(identity, "run_name"), "landuse")
    return f"{_MODEL_EXPERIMENT_ROOT}/{experiment}/run-{_publication_run_id(identity)}"


def _progress_text(checkpoint_step: int | None) -> str:
    if (
        isinstance(checkpoint_step, int)
        and not isinstance(checkpoint_step, bool)
        and checkpoint_step >= 0
    ):
        return f"checkpoint at step {checkpoint_step}"
    return "final model"


def _trackio_link(trackio_space_id: object) -> str | None:
    if (
        isinstance(trackio_space_id, str)
        and trackio_space_id.strip()
        and "\n" not in trackio_space_id
        and "\r" not in trackio_space_id
    ):
        return "https://huggingface.co/spaces/" + trackio_space_id.strip()
    return None


def _tracking_section(trackio_space_id: str | None) -> str:
    trackio_link = _trackio_link(trackio_space_id)
    if trackio_link is None:
        return "Trackio was not enabled for this run."
    return (
        f"[Open the Trackio dashboard]({trackio_link}). "
        "Metrics are published as static snapshots after complete checkpoints "
        "and final publication."
    )


def render_model_card(
    *,
    identity: Mapping[str, object],
    training_metrics: Mapping[str, object] | None = None,
    checkpoint_step: int | None = None,
    trackio_space_id: str | None = None,
) -> str:
    """Render a deterministic, credential-free model card from run facts."""

    task_name, task_label, task_tag = _task_metadata(identity)
    model_name = _safe_line(identity.get("model_name_or_path"), "not recorded")
    model_revision = _safe_line(identity.get("model_revision"), "not pinned")
    dataset_revision = _safe_line(identity.get("dataset_revision"), "not recorded")
    source_commit = _safe_line(identity.get("source_commit"), "not recorded")
    training_config = _safe_scalar_mapping(identity.get("training_config"))
    metrics = _safe_scalar_mapping(training_metrics)
    progress = _progress_text(checkpoint_step)
    tracking_section = _tracking_section(trackio_space_id)
    return (
        "---\n"
        "library_name: transformers\n"
        "pipeline_tag: text-classification\n"
        "tags:\n"
        f"- {task_tag}\n"
        "- text-classification\n"
        "---\n\n"
        f"# OSM Polygon {task_label.title()} Sentence Classifier\n\n"
        f"This model classifies whether a sentence is relevant to {task_label}. "
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


def render_repository_readme(
    *,
    identity: Mapping[str, object],
    trackio_space_id: str | None = None,
) -> str:
    """Render the stable root guide for the organized model repository."""

    prefix = _model_artifact_prefix(identity)
    task_name, task_label, task_tag = _task_metadata(identity)
    model_name = _safe_line(identity.get("model_name_or_path"), "not recorded")
    model_revision = _safe_line(identity.get("model_revision"), "not pinned")
    trackio_link = _trackio_link(trackio_space_id)
    tracking_line = (
        f"- Current run Trackio dashboard: [{trackio_space_id}]({trackio_link})\n"
        if trackio_link is not None
        else ""
    )
    return (
        "---\n"
        "library_name: transformers\n"
        "pipeline_tag: text-classification\n"
        "tags:\n"
        f"- {task_tag}\n"
        "- text-classification\n"
        "---\n\n"
        "# OSM Polygon Sentence Classifier\n\n"
        "This public repository contains organized, immutable outputs for the "
        "OSM polygon sentence-classification studies.\n\n"
        "## Repository layout\n\n"
        f"- Final model: `{prefix}/final/` (load with the Transformers "
        f"`subfolder` argument).\n"
        f"- Complete checkpoints: `{prefix}/checkpoints/step-N/`.\n"
        "- Each run has its own experiment and run directory; no model files "
        "are stored at the repository root.\n\n"
        "## Study registry\n\n"
        "- Completed landuse ablations: [`studies/landuse-v1/README.md`]"
        "(studies/landuse-v1/README.md).\n"
        "- Protocol: [`studies/landuse-v1/study.json`]"
        "(studies/landuse-v1/study.json).\n"
        "- Results: [`studies/landuse-v1/results.json`]"
        "(studies/landuse-v1/results.json).\n\n"
        "- Worldwide V2 baseline: [`studies/place-relevance-v2/README.md`]"
        "(studies/place-relevance-v2/README.md).\n"
        "- Protocol: [`studies/place-relevance-v2/study.json`]"
        "(studies/place-relevance-v2/study.json).\n"
        "- Results: [`studies/place-relevance-v2/results.json`]"
        "(studies/place-relevance-v2/results.json).\n\n"
        "- Worldwide V2 ablations: [`studies/place-relevance-v2-ablations/README.md`]"
        "(studies/place-relevance-v2-ablations/README.md).\n"
        "- Ablation protocol: [`studies/place-relevance-v2-ablations/study.json`]"
        "(studies/place-relevance-v2-ablations/study.json).\n"
        "- Ablation results: [`studies/place-relevance-v2-ablations/results.json`]"
        "(studies/place-relevance-v2-ablations/results.json).\n\n"
        "## Experiment tracking\n\n"
        f"- V1 landuse: [{TRACKIO_SPACE_ID}]"
        f"(https://huggingface.co/spaces/{TRACKIO_SPACE_ID}).\n"
        f"  Static data bucket: `{TRACKIO_BUCKET_ID}`.\n"
        f"- V2 place relevance: [{V2_TRACKIO_SPACE_ID}]"
        f"(https://huggingface.co/spaces/{V2_TRACKIO_SPACE_ID}).\n"
        f"  Static data bucket: `{V2_TRACKIO_BUCKET_ID}`.\n\n"
        "## Training identity\n\n"
        f"- Task: `{task_name}`\n"
        f"- Task description: {task_label}\n"
        f"- Base model: `{model_name}`\n"
        f"- Base-model revision: `{model_revision}`\n"
        f"- Run directory: `{prefix}`\n"
        f"{tracking_line}\n"
        "The generated model card inside the final directory contains the "
        "recorded configuration and evaluation metrics."
    )


def ensure_model_repository(
    repository_id: object,
    *,
    hub_api: _HubRepositoryProtocol | None = None,
) -> None:
    """Create the dedicated model repository without uploading an artifact."""

    repository = _require_non_blank(repository_id, "repository_id")
    try:
        api = hub_api or _default_hub_api()
        api.create_repo(
            repo_id=repository,
            repo_type="model",
            private=False,
            exist_ok=True,
        )
    except ModelPublicationError:
        raise
    except Exception as error:
        raise ModelPublicationError(
            "Hugging Face model repository setup failed"
        ) from error


def _is_real_directory(directory: Path) -> bool:
    return directory.exists() and directory.is_dir() and not directory.is_symlink()


def _is_symlink(path: Path) -> bool:
    return path.is_symlink()


def _model_directory_entries(directory: Path) -> tuple[Path, ...]:
    try:
        return tuple(directory.iterdir())
    except OSError as error:
        raise ModelPublicationError("model output cannot be read") from error


def _require_no_model_symlinks(entries: Sequence[Path]) -> None:
    if any(_is_symlink(path) for path in entries):
        raise ModelPublicationError("model output cannot contain symlinks")


def _regular_model_files(entries: Sequence[Path]) -> tuple[Path, ...]:
    return tuple(sorted(path for path in entries if path.is_file()))


def _model_output_files(directory: Path) -> tuple[Path, ...]:
    if not _is_real_directory(directory):
        raise ModelPublicationError("model output must be a real directory")
    entries = _model_directory_entries(directory)
    _require_no_model_symlinks(entries)
    return _regular_model_files(entries)


def _has_model_weights(names: set[str]) -> bool:
    return any(_WEIGHT_PATTERN.fullmatch(name) for name in names)


def _has_tokenizer_files(names: set[str]) -> bool:
    return bool(
        names
        & {
            "tokenizer.json",
            "tokenizer_config.json",
            "spiece.model",
            "sentencepiece.bpe.model",
            "tokenizer.model",
            "vocab.json",
            "vocab.txt",
        }
    )


def _validate_model_output_files(files: Sequence[Path]) -> None:
    names = {path.name for path in files}
    if "config.json" not in names:
        raise ModelPublicationError("model output is missing config.json")
    if not _has_model_weights(names):
        raise ModelPublicationError("model output is missing model weights")
    if not _has_tokenizer_files(names):
        raise ModelPublicationError("model output is missing tokenizer files")


def _publishable_model_files(files: Sequence[Path]) -> tuple[Path, ...]:
    selected = tuple(
        path
        for path in files
        if path.name in _ALLOWED_ROOT_NAMES or _WEIGHT_PATTERN.fullmatch(path.name)
    )
    if not selected:
        raise ModelPublicationError("model output contains no publishable files")
    return selected


def _final_model_files(directory: Path) -> tuple[Path, ...]:
    files = _model_output_files(directory)
    _validate_model_output_files(files)
    return _publishable_model_files(files)


def _commit_facts(info: object) -> tuple[str, str]:
    commit_id = getattr(info, "oid", None)
    commit_url = getattr(info, "commit_url", None)
    if not isinstance(commit_id, str) or not commit_id.strip():
        raise ModelPublicationError("Hugging Face returned an invalid model commit ID")
    if not isinstance(commit_url, str) or not commit_url.strip():
        raise ModelPublicationError("Hugging Face returned an invalid model commit URL")
    return commit_id, commit_url


def _commit_publication(
    *,
    api: _HubCommitProtocol,
    repository: str,
    operations: Sequence[object],
    commit_message: str,
    published_paths: Sequence[str],
    failure_message: str,
) -> ModelPublicationResult:
    try:
        info = api.create_commit(
            repo_id=repository,
            repo_type="model",
            operations=operations,
            commit_message=commit_message,
            revision="main",
        )
    except Exception as error:
        raise ModelPublicationError(failure_message) from error
    commit_id, commit_url = _commit_facts(info)
    return ModelPublicationResult(
        repository_id=repository,
        commit_id=commit_id,
        commit_url=commit_url,
        files=tuple(published_paths),
    )


def _repository_readme_operation(
    repository_readme: str | None,
    factory: OperationFactory,
) -> tuple[object, str] | None:
    if repository_readme is None:
        return None
    if not isinstance(repository_readme, str) or not repository_readme.strip():
        raise ModelPublicationError("repository README must be non-empty")
    return (
        factory(
            path_in_repo="README.md",
            path_or_fileobj=repository_readme.encode(),
        ),
        "README.md",
    )


def _model_operations(
    files: Sequence[Path],
    *,
    factory: OperationFactory,
    artifact_prefix: str | None,
    repository_readme: str | None,
) -> tuple[list[object], list[str]]:
    operations: list[object] = []
    published_paths: list[str] = []
    readme_operation = _repository_readme_operation(repository_readme, factory)
    if readme_operation is not None:
        operation, path_in_repo = readme_operation
        operations.append(operation)
        published_paths.append(path_in_repo)
    for path in files:
        path_in_repo = (
            f"{artifact_prefix}/final/{path.name}"
            if artifact_prefix is not None
            else path.name
        )
        operations.append(
            factory(
                path_in_repo=path_in_repo,
                path_or_fileobj=str(path),
            )
        )
        published_paths.append(path_in_repo)
    return operations, published_paths


def publish_model_directory(
    directory: str | Path,
    repository_id: object,
    *,
    identity: Mapping[str, object] | None = None,
    repository_readme: str | None = None,
    hub_api: _HubCommitProtocol | None = None,
    operation_factory: OperationFactory | None = None,
) -> ModelPublicationResult:
    """Validate and commit final artifacts under an immutable run directory."""

    repository = _require_non_blank(repository_id, "repository_id")
    output_directory = Path(directory)
    files = _final_model_files(output_directory)
    factory = operation_factory or _default_operation_factory()
    artifact_prefix = _model_artifact_prefix(identity) if identity is not None else None
    try:
        operations, published_paths = _model_operations(
            files,
            factory=factory,
            artifact_prefix=artifact_prefix,
            repository_readme=repository_readme,
        )
    except Exception as error:
        raise ModelPublicationError(
            "Hugging Face model operations could not be constructed"
        ) from error

    api = hub_api or _default_hub_api()
    return _commit_publication(
        api=api,
        repository=repository,
        operations=operations,
        commit_message="Publish completed classifier model",
        published_paths=published_paths,
        failure_message="Hugging Face model publication failed",
    )


def _safe_study_path_parts(path: PurePosixPath) -> bool:
    return (
        not path.is_absolute()
        and bool(path.parts)
        and not any(part in {"", ".", ".."} for part in path.parts)
    )


def _safe_study_path_text(raw_path: object) -> bool:
    if not isinstance(raw_path, str):
        return False
    return all(marker not in raw_path for marker in ("\\", "\n", "\r"))


def _is_safe_study_document_path(raw_path: object, path: PurePosixPath) -> bool:
    return _safe_study_path_parts(path) and _safe_study_path_text(raw_path)


def _study_document_operation(
    raw_path: str,
    content: str,
    factory: OperationFactory,
) -> tuple[object, str]:
    path = PurePosixPath(raw_path)
    if not _is_safe_study_document_path(raw_path, path):
        raise ModelPublicationError("study document path is unsafe")
    if not isinstance(content, str) or not content.strip():
        raise ModelPublicationError("study document content is empty")
    normalized_path = path.as_posix()
    return (
        factory(
            path_in_repo=normalized_path,
            path_or_fileobj=content.encode(),
        ),
        normalized_path,
    )


def _study_operations(
    documents: Mapping[str, str],
    factory: OperationFactory,
) -> tuple[list[object], list[str]]:
    operations: list[object] = []
    published_paths: list[str] = []
    for raw_path, content in sorted(documents.items()):
        operation, normalized_path = _study_document_operation(
            raw_path,
            content,
            factory,
        )
        operations.append(operation)
        published_paths.append(normalized_path)
    return operations, published_paths


def _safe_study_operations(
    documents: Mapping[str, str],
    factory: OperationFactory,
) -> tuple[list[object], list[str]]:
    try:
        return _study_operations(documents, factory)
    except ModelPublicationError:
        raise
    except Exception as error:
        raise ModelPublicationError(
            "study document operations could not be constructed"
        ) from error


def publish_study_documents(
    repository_id: object,
    documents: Mapping[str, str],
    *,
    hub_api: _HubCommitProtocol | None = None,
    operation_factory: OperationFactory | None = None,
) -> ModelPublicationResult:
    """Commit generated study documentation under explicit repository paths."""

    repository = _require_non_blank(repository_id, "repository_id")
    if not documents:
        raise ModelPublicationError("study documents cannot be empty")
    factory = operation_factory or _default_operation_factory()
    operations, published_paths = _safe_study_operations(documents, factory)

    api = hub_api or _default_hub_api()
    return _commit_publication(
        api=api,
        repository=repository,
        operations=operations,
        commit_message="Update classifier study report",
        published_paths=published_paths,
        failure_message="study documentation publication failed",
    )


def _complete_checkpoint_files(
    directory: Path,
    *,
    identity: Mapping[str, object],
) -> tuple[Path, ...]:
    _require_complete_checkpoint(directory, identity=identity)
    return _checkpoint_output_files(directory)


def _require_complete_checkpoint(
    directory: Path,
    *,
    identity: Mapping[str, object],
) -> None:
    try:
        selected = find_complete_checkpoint(
            directory,
            identity=identity,
        )
    except CheckpointError as error:
        raise ModelPublicationError("checkpoint evidence is invalid") from error
    if selected is None:
        raise ModelPublicationError("model output is not a complete checkpoint")


def _checkpoint_output_files(directory: Path) -> tuple[Path, ...]:
    def is_publishable(path: Path) -> bool:
        return (
            path.is_file()
            and not path.is_symlink()
            and (
                path.name in _ALLOWED_CHECKPOINT_NAMES
                or _WEIGHT_PATTERN.fullmatch(path.name) is not None
            )
        )

    try:
        files = tuple(
            sorted(path for path in directory.iterdir() if is_publishable(path))
        )
    except OSError as error:
        raise ModelPublicationError("checkpoint output cannot be read") from error
    if not files:
        raise ModelPublicationError("checkpoint output contains no files")
    return files


def _checkpoint_path_in_repo(
    path: Path,
    checkpoint: Path,
    *,
    identity: Mapping[str, object],
) -> str:
    match = _CHECKPOINT_NAME_PATTERN.fullmatch(checkpoint.name)
    if match is None:
        raise ModelPublicationError("checkpoint directory name is invalid")
    prefix = _model_artifact_prefix(identity)
    return f"{prefix}/checkpoints/step-{match.group(1)}/{path.name}"


def _checkpoint_operations(
    files: Sequence[Path],
    checkpoint: Path,
    *,
    identity: Mapping[str, object],
    factory: OperationFactory,
) -> list[object]:
    return [
        factory(
            path_in_repo=_checkpoint_path_in_repo(
                path,
                checkpoint,
                identity=identity,
            ),
            path_or_fileobj=str(path),
        )
        for path in files
    ]


def _checkpoint_published_paths(
    files: Sequence[Path],
    checkpoint: Path,
    *,
    identity: Mapping[str, object],
) -> tuple[str, ...]:
    return tuple(
        _checkpoint_path_in_repo(path, checkpoint, identity=identity) for path in files
    )


def publish_checkpoint_directory(
    directory: str | Path,
    repository_id: object,
    *,
    identity: Mapping[str, object],
    hub_api: _HubCommitProtocol | None = None,
    operation_factory: OperationFactory | None = None,
) -> ModelPublicationResult:
    """Commit one complete checkpoint under its permanent step directory."""

    repository = _require_non_blank(repository_id, "repository_id")
    checkpoint = Path(directory)
    files = _complete_checkpoint_files(checkpoint, identity=identity)
    factory = operation_factory or _default_operation_factory()
    try:
        operations = _checkpoint_operations(
            files,
            checkpoint,
            identity=identity,
            factory=factory,
        )
    except Exception as error:
        raise ModelPublicationError(
            "Hugging Face checkpoint operations could not be constructed"
        ) from error

    published_paths = _checkpoint_published_paths(
        files,
        checkpoint,
        identity=identity,
    )
    api = hub_api or _default_hub_api()
    return _commit_publication(
        api=api,
        repository=repository,
        operations=operations,
        commit_message=f"Publish checkpoint {checkpoint.name}",
        published_paths=published_paths,
        failure_message="Hugging Face checkpoint publication failed",
    )


__all__ = [
    "ModelPublicationError",
    "ModelPublicationResult",
    "ensure_model_repository",
    "publish_checkpoint_directory",
    "publish_model_directory",
    "publish_study_documents",
    "render_model_card",
    "render_repository_readme",
]
