from osm_polygon_sentence_classifier.dataset_contract import (
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
    assert PLACE_RELEVANCE_V2_DEFAULT_MAX_CONTINUATIONS == 40
