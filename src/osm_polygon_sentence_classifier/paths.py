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

    current = canonical_root
    for component in child.parts:
        current /= component
        _reject_symlink_component(current, canonical_root)

    candidate = (canonical_root / child).resolve()
    _require_managed_child(candidate, canonical_root)
    return candidate


def _reject_symlink_component(current: Path, canonical_root: Path) -> None:
    if not current.is_symlink():
        return
    try:
        resolved_target = current.resolve()
    except RuntimeError as error:
        raise ManagedPathError("symlinked path components are not allowed") from error
    if not resolved_target.is_relative_to(canonical_root):
        raise ManagedPathError("path must remain beneath the managed root")
    raise ManagedPathError("symlinked path components are not allowed")


def _require_managed_child(candidate: Path, canonical_root: Path) -> None:
    if not candidate.is_relative_to(canonical_root):
        raise ManagedPathError("path must remain beneath the managed root")


@dataclass(frozen=True, slots=True)
class ManagedPaths:
    """Application-owned paths derived from the fixed project configuration."""

    config: ProjectConfig

    def child(self, relative_path: str | Path) -> Path:
        return resolve_managed_path(self.config.data_root, relative_path)
