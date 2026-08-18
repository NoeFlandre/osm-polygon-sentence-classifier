from osm_polygon_sentence_classifier.place_relevance_reporting import (
    render_place_relevance_study_documents,
)


def test_place_relevance_report_is_public_and_reproducible() -> None:
    documents = render_place_relevance_study_documents(
        identity={
            "run_id": "a" * 20,
            "task_name": "place-relevance-v2",
            "dataset_revision": "b" * 40,
            "model_name_or_path": "jhu-clsp/mmBERT-small",
            "model_revision": "c" * 40,
            "training_config": {
                "validation_fraction": 0.1,
                "test_fraction": 0.1,
                "seed": 42,
                "trainable_layers": "head",
                "HF_TOKEN": "must-not-appear",
            },
        },
        metrics={"eval_accuracy": 0.8, "test_macro_f1": 0.7},
        trackio_space_id="NoeFlandre/osm-polygon-sentence-classifier-trackio",
    )

    assert set(documents) == {
        "studies/place-relevance-v2/README.md",
        "studies/place-relevance-v2/data-audit.json",
        "studies/place-relevance-v2/results.json",
        "studies/place-relevance-v2/study.json",
    }
    assert "Worldwide V2" in documents["studies/place-relevance-v2/README.md"]
    assert '"test_fraction": 0.1' in documents["studies/place-relevance-v2/study.json"]
    assert (
        '"clean_rows": 175458'
        in documents["studies/place-relevance-v2/data-audit.json"]
    )
    assert "data-audit.json" in documents["studies/place-relevance-v2/README.md"]
    assert (
        '"test_macro_f1": 0.7' in documents["studies/place-relevance-v2/results.json"]
    )
    assert "HF_TOKEN" not in "".join(documents.values())
