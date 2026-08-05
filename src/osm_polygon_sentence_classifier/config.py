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
    """Non-secret configuration for local or explicitly remote workflows."""

    data_root: Path = APPROVED_DATA_ROOT

    def __post_init__(self) -> None:
        root = Path(self.data_root).expanduser()
        if root != APPROVED_DATA_ROOT:
            raise ConfigurationError(
                "data_root must be the approved external data root: "
                f"{APPROVED_DATA_ROOT}"
            )
        object.__setattr__(self, "data_root", APPROVED_DATA_ROOT)

    @classmethod
    def for_remote_root(cls, data_root: Path) -> "ProjectConfig":
        """Build a configuration rooted beneath the remote user's home.

        The normal constructor remains locked to the approved Seagate root.
        This narrow alternate constructor is for the Grid'5000 compute-node
        worker, whose persistent home storage is a different machine-local
        root. It rejects traversal and roots outside ``Path.home()``.
        """

        candidate = Path(data_root).expanduser()
        if not candidate.is_absolute() or any(
            part in (".", "..") for part in candidate.parts
        ):
            raise ConfigurationError("remote data root must be an absolute home path")
        try:
            root = candidate.resolve()
            home = Path.home().resolve()
        except RuntimeError as error:
            raise ConfigurationError("remote data root cannot be resolved") from error
        if root == Path("/") or not root.is_relative_to(home):
            raise ConfigurationError("remote data root must be beneath the remote home")

        config = cls()
        object.__setattr__(config, "data_root", root)
        return config

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
