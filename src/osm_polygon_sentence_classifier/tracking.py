import re
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from .config import ProjectConfig
from .huggingface_http import configure_huggingface_http
from .paths import resolve_managed_path

TRACKING_SUBDIRECTORY = Path("tracking")
TRACKIO_SPACE_ID = "NoeFlandre/osm-polygon-sentence-classifier-trackio"
TRACKIO_STATIC_SPACE_ID = TRACKIO_SPACE_ID
TRACKIO_BUCKET_ID = "NoeFlandre/osm-polygon-sentence-classifier-trackio-data"
V2_TRACKIO_SPACE_ID = "NoeFlandre/osm-polygon-sentence-classifier-v2-trackio"
V2_TRACKIO_STATIC_SPACE_ID = V2_TRACKIO_SPACE_ID
V2_TRACKIO_BUCKET_ID = "NoeFlandre/osm-polygon-sentence-classifier-v2-trackio-data"
V2_TRACKIO_PROJECTS = frozenset({"place-relevance-v2", "place-relevance-v2-ablations"})


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


def settings_for(
    config: ProjectConfig,
    *,
    project: str | None = None,
) -> TrackioSettings:
    """Build Trackio settings without importing or initializing Trackio."""

    directory = resolve_managed_path(config.data_root, TRACKING_SUBDIRECTORY)
    effective_project = config.project_name if project is None else project
    if not isinstance(effective_project, str) or not effective_project.strip():
        raise TrackingError("Trackio project must be a non-empty string")
    if "\n" in effective_project or "\r" in effective_project:
        raise TrackingError("Trackio project must be a single-line string")
    if effective_project in V2_TRACKIO_PROJECTS:
        return TrackioSettings(
            project=effective_project,
            directory=directory,
            space_id=V2_TRACKIO_SPACE_ID,
            bucket_id=V2_TRACKIO_BUCKET_ID,
            static_space_id=V2_TRACKIO_STATIC_SPACE_ID,
        )
    return TrackioSettings(project=effective_project, directory=directory)


def ensure_trackio_resources(
    settings: TrackioSettings,
    *,
    hub_api: Any | None = None,
) -> None:
    """Create the free static Trackio Space and Bucket idempotently."""

    try:
        api = hub_api
        if api is None:
            configure_huggingface_http()
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


def restore_static_project_snapshot(settings: TrackioSettings) -> None:
    """Restore the previous static snapshot before appending a new remote run.

    Grid'5000 allocations use isolated run directories. Static Trackio exports
    contain Parquet snapshots rather than the source SQLite database, so each
    ablation allocation restores those snapshots into Trackio's local database
    before logging. This keeps the existing dashboard cumulative across jobs.
    """

    try:
        settings.directory.mkdir(parents=True, exist_ok=True)
        project_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", settings.project).strip("_")
        if not project_stem:
            raise TrackingError("Trackio project name cannot produce a local filename")
        files = [
            (
                "metrics.parquet",
                settings.directory / f"{project_stem}.parquet",
            ),
            (
                "aux/system_metrics.parquet",
                settings.directory / f"{project_stem}_system.parquet",
            ),
            (
                "aux/configs.parquet",
                settings.directory / f"{project_stem}_configs.parquet",
            ),
            (
                "aux/traces.parquet",
                settings.directory / f"{project_stem}_traces.parquet",
            ),
        ]
        configure_huggingface_http()
        hub = import_module("huggingface_hub")
        download = getattr(hub, "download_bucket_files", None)
        if not callable(download):
            raise TrackingError("Hugging Face bucket download is unavailable")
        download(settings.bucket_id, files, raise_on_missing_files=False)
        trackio = import_module("trackio")
        storage = getattr(trackio, "SQLiteStorage", None)
        import_from_parquet = getattr(storage, "import_from_parquet", None)
        if not callable(import_from_parquet):
            raise TrackingError("Trackio Parquet import is unavailable")
        import_from_parquet()
    except TrackingError:
        raise
    except Exception as error:
        raise TrackingError("Trackio static snapshot restoration failed") from error


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
    "V2_TRACKIO_BUCKET_ID",
    "V2_TRACKIO_PROJECTS",
    "V2_TRACKIO_SPACE_ID",
    "V2_TRACKIO_STATIC_SPACE_ID",
    "ensure_trackio_resources",
    "restore_static_project_snapshot",
    "settings_for",
    "sync_project_to_static_space",
]
