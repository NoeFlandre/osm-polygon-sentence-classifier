from types import SimpleNamespace

import pytest

from osm_polygon_sentence_classifier.training_metrics import (
    MetricsInputError,
    classification_metrics,
    latest_evaluation_metrics,
    latest_training_metrics,
    metrics_for_model_card,
)


def test_classification_metrics_report_binary_quality_metrics() -> None:
    metrics = classification_metrics(
        SimpleNamespace(
            predictions=[[3.0, 1.0], [1.0, 4.0], [2.0, 1.0], [0.0, 5.0]],
            label_ids=[0, 1, 1, 1],
        )
    )

    assert metrics["accuracy"] == pytest.approx(0.75)
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(2 / 3)
    assert metrics["f1"] == pytest.approx(0.8)
    assert metrics["macro_f1"] == pytest.approx(0.7333333333333334)
    assert metrics["balanced_accuracy"] == pytest.approx(5 / 6)
    assert metrics["positive_support"] == pytest.approx(3)
    assert metrics["negative_support"] == pytest.approx(1)


def test_classification_metrics_reject_invalid_predictions_and_labels() -> None:
    with pytest.raises(MetricsInputError, match="predictions and labels are invalid"):
        classification_metrics(SimpleNamespace(predictions=[[1.0, 0.0]], label_ids=[2]))


def test_latest_metric_helpers_keep_only_scalar_values() -> None:
    state = SimpleNamespace(
        log_history=[
            {"loss": 0.8, "ignored": object()},
            {"loss": 0.4, "step": 2, "nested": {"not": "a scalar"}},
        ]
    )

    assert latest_training_metrics(state) == {"loss": 0.4, "step": 2}
    assert latest_evaluation_metrics(state) == {}


def test_metrics_for_model_card_uses_the_latest_evaluation_entry() -> None:
    train_output = SimpleNamespace(metrics={"train_loss": 0.4})
    trainer = SimpleNamespace(
        state=SimpleNamespace(
            log_history=[
                {"eval_loss": 0.3, "eval_accuracy": 0.8, "eval_f1": 0.7},
                {"loss": 0.2},
            ]
        )
    )

    assert metrics_for_model_card(train_output, trainer) == {
        "train_loss": 0.4,
        "eval_accuracy": 0.8,
        "eval_f1": 0.7,
        "eval_loss": 0.3,
    }
