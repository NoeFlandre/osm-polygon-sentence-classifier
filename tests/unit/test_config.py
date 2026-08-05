from pathlib import Path

import pytest

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


def test_config_is_immutable() -> None:
    config = ProjectConfig()

    with pytest.raises(AttributeError):
        config.task_name = "polygon-relevance"  # type: ignore[misc]  # ty: ignore[invalid-assignment]
