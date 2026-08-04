from dataclasses import dataclass
from pathlib import Path

from .config import ProjectConfig


class ManagedPathError(ValueError):
    """Raised when a path would escape the managed data root."""


def resolve_managed_path(root: Path, relative_path: str | Path) -> Path:
    """Resolve a child path and require it to remain beneath root."""

    child = Path(relative_path)
    canonical_root = root.resolve()
    if child.is_absolute():
        raise ManagedPathError("path must remain beneath the managed root")

    candidate = (canonical_root / child).resolve()
    if not candidate.is_relative_to(canonical_root):
        raise ManagedPathError("path must remain beneath the managed root")
    return candidate


@dataclass(frozen=True, slots=True)
class ManagedPaths:
    """Application-owned paths derived from the fixed project configuration."""

    config: ProjectConfig

    def child(self, relative_path: str | Path) -> Path:
        return resolve_managed_path(self.config.data_root, relative_path)
