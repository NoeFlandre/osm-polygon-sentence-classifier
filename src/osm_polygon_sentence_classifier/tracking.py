from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from .config import ProjectConfig
from .paths import resolve_managed_path

TRACKING_SUBDIRECTORY = Path("tracking")
TRACKIO_SPACE_ID = "NoeFlandre/osm-polygon-sentence-classifier-trackio"
TRACKIO_BUCKET_ID = "NoeFlandre/osm-polygon-sentence-classifier-trackio-data"


class TrackingError(RuntimeError):
    """Raised when an explicit Trackio synchronization cannot complete."""


@dataclass(frozen=True, slots=True)
class TrackioSettings:
    """Non-secret Trackio settings for a training process."""

    project: str
    directory: Path
    space_id: str = TRACKIO_SPACE_ID
    bucket_id: str = TRACKIO_BUCKET_ID

    def environment(self) -> dict[str, str]:
        """Return environment values needed to keep local Trackio data managed."""

        return {"TRACKIO_DIR": str(self.directory)}


def settings_for(config: ProjectConfig) -> TrackioSettings:
    """Build Trackio settings without importing or initializing Trackio."""

    directory = resolve_managed_path(config.data_root, TRACKING_SUBDIRECTORY)
    return TrackioSettings(project=config.project_name, directory=directory)


def ensure_trackio_resources(
    settings: TrackioSettings,
    *,
    hub_api: Any | None = None,
) -> None:
    """Create the dedicated static Space and bucket idempotently."""

    try:
        api = hub_api
        if api is None:
            hub = import_module("huggingface_hub")
            api = hub.HfApi()
        api.create_repo(
            repo_id=settings.space_id,
            repo_type="space",
            space_sdk="static",
            exist_ok=True,
        )
        api.create_bucket(bucket_id=settings.bucket_id, exist_ok=True)
    except Exception as error:
        raise TrackingError("Trackio Space and bucket provisioning failed") from error


def sync_project_to_static_space(settings: TrackioSettings) -> str:
    """Synchronize the current local project snapshot to the static Space."""

    try:
        trackio: Any = import_module("trackio")
        space_id = trackio.sync(
            project=settings.project,
            space_id=settings.space_id,
            bucket_id=settings.bucket_id,
            sdk="static",
            force=True,
        )
    except Exception as error:
        raise TrackingError("Trackio static Space synchronization failed") from error
    if not isinstance(space_id, str) or not space_id.strip():
        raise TrackingError("Trackio synchronization returned an invalid Space ID")
    return space_id


__all__ = [
    "TRACKIO_BUCKET_ID",
    "TRACKIO_SPACE_ID",
    "TRACKING_SUBDIRECTORY",
    "TrackingError",
    "TrackioSettings",
    "ensure_trackio_resources",
    "settings_for",
    "sync_project_to_static_space",
]
