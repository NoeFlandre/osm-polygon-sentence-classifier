from osm_polygon_sentence_classifier.dataset_contract import (
    LANDUSE_DATASET_CONTRACT,
    WORLDWIDE_PLACE_RELEVANCE_V2_CONTRACT,
)
from osm_polygon_sentence_classifier.training_tasks import (
    PLACE_RELEVANCE_V2_DEFAULT_MAX_CONTINUATIONS,
    task_contract,
    training_config_for_task,
)


def test_worldwide_v2_task_definition_is_independent_of_cli_parsing() -> None:
    config = training_config_for_task(
        "place-relevance-v2",
        model_name_or_path="test-model",
        model_revision="a" * 40,
        publish_to_hub=True,
        sync_trackio=True,
    )

    assert task_contract("place-relevance-v2") is WORLDWIDE_PLACE_RELEVANCE_V2_CONTRACT
    assert config.tracking_project == "place-relevance-v2"
    assert config.eval_strategy == "epoch"
    assert config.publish_to_hub is True
    assert config.sync_trackio is True
    assert PLACE_RELEVANCE_V2_DEFAULT_MAX_CONTINUATIONS == 40


def test_task_contract_selects_the_landuse_dataset() -> None:
    assert task_contract("landuse") is LANDUSE_DATASET_CONTRACT


def test_worldwide_v2_defaults_do_not_publish_or_sync() -> None:
    config = training_config_for_task(
        "place-relevance-v2",
        model_name_or_path="test-model",
        model_revision="a" * 40,
    )

    assert config.publish_to_hub is False
    assert config.sync_trackio is False


def test_worldwide_v2_forwards_explicit_publication_flags() -> None:
    config = training_config_for_task(
        "place-relevance-v2",
        model_name_or_path="test-model",
        model_revision="a" * 40,
        publish_to_hub=False,
        sync_trackio=False,
    )

    assert config.publish_to_hub is False
    assert config.sync_trackio is False


def test_landuse_task_uses_its_default_training_step_budget() -> None:
    config = training_config_for_task("landuse", model_name_or_path="test-model")

    assert config.max_steps == 1_000
