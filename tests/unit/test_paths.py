from pathlib import Path

import pytest

from osm_polygon_sentence_classifier.config import ProjectConfig
from osm_polygon_sentence_classifier.paths import (
    ManagedPathError,
    ManagedPaths,
    resolve_managed_path,
)


def test_relative_child_resolves_under_the_supplied_root(tmp_path: Path) -> None:
    root = tmp_path / "approved"

    result = resolve_managed_path(root, "runs/first")

    assert result == (root / "runs/first").resolve()
    assert result.is_relative_to(root.resolve())


@pytest.mark.parametrize(
    "candidate",
    ["/tmp/outside", "../outside", "runs/../../outside"],
)
def test_absolute_or_traversal_paths_are_rejected(
    tmp_path: Path,
    candidate: str,
) -> None:
    with pytest.raises(ManagedPathError, match="beneath the managed root"):
        resolve_managed_path(tmp_path / "approved", candidate)


def test_symlink_that_escapes_the_root_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "approved"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ManagedPathError, match="beneath the managed root"):
        resolve_managed_path(root, "escape/file.json")


def test_symlinked_directory_component_is_rejected_even_when_it_stays_inside(
    tmp_path: Path,
) -> None:
    root = tmp_path / "approved"
    root.mkdir()
    target = root / "real"
    target.mkdir()
    (root / "audit").symlink_to(target, target_is_directory=True)

    with pytest.raises(ManagedPathError, match="symlink"):
        resolve_managed_path(root, "audit/landuse")


def test_symlink_loop_is_reported_as_a_managed_path_error(tmp_path: Path) -> None:
    root = tmp_path / "approved"
    root.mkdir()
    loop = root / "loop"
    loop.symlink_to(loop)

    with pytest.raises(ManagedPathError, match="symlink"):
        resolve_managed_path(root, "loop/file.json")


def test_application_paths_use_the_fixed_project_root() -> None:
    paths = ManagedPaths(ProjectConfig())

    assert paths.child("tracking") == ProjectConfig().data_root / "tracking"
