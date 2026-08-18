"""Task definitions shared by training and Grid'5000 command boundaries."""

from __future__ import annotations

from typing import Literal

from .dataset_contract import (
    LANDUSE_DATASET_CONTRACT,
    WORLDWIDE_PLACE_RELEVANCE_V2_CONTRACT,
    DatasetContract,
)
from .training import (
    DEFAULT_MODEL_NAME,
    TrainingConfig,
    place_relevance_v2_training_config,
)

TaskName = Literal["landuse", "place-relevance-v2"]
PLACE_RELEVANCE_V2_DEFAULT_MAX_CONTINUATIONS = 40


def task_contract(task_name: TaskName) -> DatasetContract:
    """Return the immutable dataset contract for one training task."""

    return (
        LANDUSE_DATASET_CONTRACT
        if task_name == "landuse"
        else WORLDWIDE_PLACE_RELEVANCE_V2_CONTRACT
    )


def training_config_for_task(
    task_name: TaskName,
    *,
    model_name_or_path: str = DEFAULT_MODEL_NAME,
    model_revision: str | None = None,
    max_steps: int | None = None,
    publish_to_hub: bool = False,
    sync_trackio: bool = False,
) -> TrainingConfig:
    """Build one task configuration without depending on CLI arguments."""

    if task_name == "place-relevance-v2":
        return place_relevance_v2_training_config(
            model_name_or_path=model_name_or_path,
            model_revision=model_revision,
            max_steps=(
                max_steps
                if max_steps is not None
                else place_relevance_v2_training_config().max_steps
            ),
            publish_to_hub=publish_to_hub,
            sync_trackio=sync_trackio,
        )
    return TrainingConfig(
        model_name_or_path=model_name_or_path,
        model_revision=model_revision,
        max_steps=1_000 if max_steps is None else max_steps,
        publish_to_hub=publish_to_hub,
        sync_trackio=sync_trackio,
    )


def default_max_continuations(task_name: TaskName) -> int:
    """Return the bounded continuation budget for one task."""

    return (
        PLACE_RELEVANCE_V2_DEFAULT_MAX_CONTINUATIONS
        if task_name == "place-relevance-v2"
        else 3
    )


__all__ = [
    "PLACE_RELEVANCE_V2_DEFAULT_MAX_CONTINUATIONS",
    "TaskName",
    "default_max_continuations",
    "task_contract",
    "training_config_for_task",
]
