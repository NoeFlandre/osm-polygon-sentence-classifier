from pathlib import Path

import pytest

from osm_polygon_sentence_classifier import __version__
from osm_polygon_sentence_classifier.config import (
    APPROVED_DATA_ROOT,
    PROJECT_NAME,
    SOURCE_DATASET_ID,
    TARGET_MODEL_REPOSITORY_ID,
    ConfigurationError,
    ProjectConfig,
)


def test_package_exposes_the_foundation_version() -> None:
    assert __version__ == "0.1.0"


def test_default_config_identifies_the_landuse_task() -> None:
    config = ProjectConfig()

    assert config.project_name == PROJECT_NAME == "osm-polygon-sentence-classifier"
    assert config.task_name == "landuse"
    assert config.source_dataset_id == SOURCE_DATASET_ID
    assert config.target_model_repository_id == TARGET_MODEL_REPOSITORY_ID
    assert config.data_root == APPROVED_DATA_ROOT


def test_data_root_cannot_be_replaced_by_another_local_directory(
    tmp_path: Path,
) -> None:
    with pytest.raises(ConfigurationError, match="approved external data root"):
        ProjectConfig(data_root=tmp_path)


def test_config_is_immutable() -> None:
    config = ProjectConfig()

    with pytest.raises(AttributeError):
        config.task_name = "polygon-relevance"  # type: ignore[misc]  # ty: ignore[invalid-assignment]
