from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from .config import ProjectConfig
from .paths import resolve_managed_path

TRACKING_SUBDIRECTORY = Path("tracking")
TRACKIO_SPACE_ID = "NoeFlandre/osm-polygon-sentence-classifier-trackio"
TRACKIO_STATIC_SPACE_ID = TRACKIO_SPACE_ID
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
    static_space_id: str = TRACKIO_STATIC_SPACE_ID

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
    """Create the free static Trackio Space and Bucket idempotently."""

    try:
        api = hub_api
        if api is None:
            hub = import_module("huggingface_hub")
            api = hub.HfApi()
        api.create_repo(
            repo_id=settings.static_space_id,
            repo_type="space",
            space_sdk="static",
            private=False,
            exist_ok=True,
        )
        api.create_bucket(
            bucket_id=settings.bucket_id,
            private=False,
            exist_ok=True,
        )
    except Exception as error:
        raise TrackingError("Trackio Space and bucket provisioning failed") from error


def _current_local_run(trackio: Any) -> Any | None:
    context_vars = getattr(trackio, "context_vars", None)
    current_run_context = getattr(context_vars, "current_run", None)
    get_current_run = getattr(current_run_context, "get", None)
    if not callable(get_current_run):
        return None
    try:
        return get_current_run()
    except LookupError:
        return None


def _flush_current_local_run(trackio: Any) -> None:
    """Flush queued local metrics before exporting a static snapshot.

    Trackio currently exposes the local flush operation only on its active Run
    object. Keep this compatibility adapter private and guarded so callers
    that synchronize after a process restart remain valid; when a live Run is
    present, the client lock prevents the background sender from racing the
    synchronous drain.
    """

    current_run = _current_local_run(trackio)
    if current_run is None:
        return
    flush = getattr(current_run, "_flush_queues_inline", None)
    if not callable(flush):
        return
    client_lock = getattr(current_run, "_client_lock", None)
    if client_lock is None:
        flush()
        return
    with client_lock:
        flush()


def _finish_current_local_run(trackio: Any) -> None:
    current_run = _current_local_run(trackio)
    if current_run is None:
        return
    finish = getattr(trackio, "finish", None)
    if not callable(finish):
        raise TrackingError("Trackio finalization is unavailable")
    finish()


def _import_local_fragments(trackio: Any) -> None:
    fragments = getattr(trackio, "fragments", None)
    if fragments is None:
        fragments = import_module("trackio.fragments")
    import_inbox_dir = getattr(fragments, "import_inbox_dir", None)
    if not callable(import_inbox_dir):
        raise TrackingError("Trackio local fragment import is unavailable")
    import_inbox_dir()


def sync_project_to_static_space(
    settings: TrackioSettings,
    *,
    finalize: bool = False,
) -> str:
    """Synchronize the current local project snapshot to the static Space.

    Grid'5000 home storage may make Trackio select append-only JSONL fragments
    instead of SQLite. Import those fragments before the static exporter reads
    the project database. ``finalize`` is used only after training so the
    active Trackio run is closed before the final import and upload.
    """

    try:
        trackio: Any = import_module("trackio")
        if finalize:
            _finish_current_local_run(trackio)
        else:
            _flush_current_local_run(trackio)
        _import_local_fragments(trackio)
        space_id = trackio.sync(
            project=settings.project,
            space_id=settings.static_space_id,
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
    "TRACKIO_STATIC_SPACE_ID",
    "TRACKING_SUBDIRECTORY",
    "TrackingError",
    "TrackioSettings",
    "ensure_trackio_resources",
    "settings_for",
    "sync_project_to_static_space",
]
