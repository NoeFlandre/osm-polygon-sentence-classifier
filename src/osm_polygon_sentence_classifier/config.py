from dataclasses import dataclass
from pathlib import Path

PROJECT_NAME = "osm-polygon-sentence-classifier"
TASK_NAME = "landuse"
SOURCE_DATASET_ID = "NoeFlandre/osm-polygon-wikidata-sentence-relevance"
TARGET_MODEL_REPOSITORY_ID = "NoeFlandre/osm-polygon-sentence-classifier"
APPROVED_DATA_ROOT = Path(
    "/Volumes/Seagate M3/projects/osm-polygon-sentence-classifier"
)


class ConfigurationError(ValueError):
    """Raised when immutable project configuration violates the contract."""


@dataclass(frozen=True, slots=True)
class ProjectConfig:
    """Non-secret configuration shared by future local and remote workflows."""

    data_root: Path = APPROVED_DATA_ROOT

    def __post_init__(self) -> None:
        root = Path(self.data_root).expanduser()
        if root != APPROVED_DATA_ROOT:
            raise ConfigurationError(
                "data_root must be the approved external data root: "
                f"{APPROVED_DATA_ROOT}"
            )
        object.__setattr__(self, "data_root", APPROVED_DATA_ROOT)

    @property
    def project_name(self) -> str:
        return PROJECT_NAME

    @property
    def task_name(self) -> str:
        return TASK_NAME

    @property
    def source_dataset_id(self) -> str:
        return SOURCE_DATASET_ID

    @property
    def target_model_repository_id(self) -> str:
        return TARGET_MODEL_REPOSITORY_ID


_frozen_project_config_setattr = ProjectConfig.__setattr__


def _immutable_project_config_setattr(
    self: ProjectConfig, name: str, value: object
) -> None:
    if name != "data_root":
        raise AttributeError(f"cannot assign to attribute '{name}'")
    _frozen_project_config_setattr(self, name, value)


setattr(  # noqa: B010
    ProjectConfig, "__setattr__", _immutable_project_config_setattr
)
