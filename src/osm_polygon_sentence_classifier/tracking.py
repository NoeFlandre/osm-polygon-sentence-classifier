from dataclasses import dataclass
from pathlib import Path

from .config import ProjectConfig
from .paths import resolve_managed_path

TRACKING_SUBDIRECTORY = Path("tracking")


@dataclass(frozen=True, slots=True)
class TrackioSettings:
    """Non-secret Trackio settings for a future training process."""

    project: str
    directory: Path

    def environment(self) -> dict[str, str]:
        """Return environment values needed to keep local Trackio data managed."""

        return {"TRACKIO_DIR": str(self.directory)}


def settings_for(config: ProjectConfig) -> TrackioSettings:
    """Build Trackio settings without importing or initializing Trackio."""

    directory = resolve_managed_path(config.data_root, TRACKING_SUBDIRECTORY)
    return TrackioSettings(project=config.project_name, directory=directory)
