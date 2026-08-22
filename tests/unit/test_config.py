from pathlib import Path
from typing import Any, cast

import pytest

import osm_polygon_sentence_classifier.config as config_module
from osm_polygon_sentence_classifier import __version__
from osm_polygon_sentence_classifier.config import (
    ConfigurationError,
    ProjectConfig,
)


def test_package_exposes_the_foundation_version() -> None:
    assert __version__ == "0.1.0"


def test_default_config_identifies_the_landuse_task() -> None:
    config = ProjectConfig()

    assert config.project_name == "osm-polygon-sentence-classifier"
    assert config.task_name == "landuse"
    assert (
        config.source_dataset_id == "NoeFlandre/osm-polygon-wikidata-sentence-relevance"
    )
    assert (
        config.target_model_repository_id
        == "NoeFlandre/osm-polygon-sentence-classifier"
    )
    assert config.data_root == Path(
        "/Volumes/Seagate M3/projects/osm-polygon-sentence-classifier"
    )


def test_data_root_cannot_be_replaced_by_another_local_directory(
    tmp_path: Path,
) -> None:
    with pytest.raises(ConfigurationError, match="approved external data root"):
        ProjectConfig(data_root=tmp_path)


def test_remote_config_allows_only_a_home_scoped_root() -> None:
    root = Path.home() / "osm-polygon-sentence-classifier-data"

    config = ProjectConfig.for_remote_root(root)

    assert config.data_root == root


def test_remote_config_rejects_a_root_outside_home(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="remote data root"):
        ProjectConfig.for_remote_root(tmp_path)


@pytest.mark.parametrize(
    "root",
    [Path("relative-root"), Path.home() / ".." / "outside-root"],
)
def test_remote_config_rejects_relative_or_parent_traversal_roots(root: Path) -> None:
    with pytest.raises(
        ConfigurationError,
        match=r"\Aremote data root must be an absolute home path\Z",
    ):
        ProjectConfig.for_remote_root(root)


def test_remote_config_reports_a_resolution_failure_with_its_cause() -> None:
    class _UnresolvablePath:
        def resolve(self) -> Path:
            raise RuntimeError("resolution loop")

    with pytest.raises(
        ConfigurationError,
        match=r"\Aremote data root cannot be resolved\Z",
    ) as caught:
        config_module._resolved_remote_paths(cast(Any, _UnresolvablePath()))

    assert isinstance(caught.value.__cause__, RuntimeError)


def test_remote_config_rejects_the_filesystem_root_even_when_home_matches() -> None:
    with pytest.raises(
        ConfigurationError,
        match=r"\Aremote data root must be beneath the remote home\Z",
    ):
        config_module._require_remote_home_child(Path("/"), Path("/"))


def test_config_forwards_data_root_assignment_to_the_frozen_setter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ProjectConfig()
    replacement = Path.home() / "remote-project-data"
    calls: list[tuple[ProjectConfig, str, object]] = []

    def record_setattr(instance: ProjectConfig, name: str, value: object) -> None:
        calls.append((instance, name, value))

    monkeypatch.setattr(
        config_module,
        "_frozen_project_config_setattr",
        record_setattr,
    )

    config.__setattr__("data_root", replacement)

    assert calls == [(config, "data_root", replacement)]


def test_config_rejects_assignment_to_other_attributes_with_exact_error() -> None:
    config = ProjectConfig()

    with pytest.raises(
        AttributeError,
        match=r"\Acannot assign to attribute 'task_name'\Z",
    ):
        config.task_name = "polygon-relevance"  # type: ignore[misc]  # ty: ignore[invalid-assignment]
