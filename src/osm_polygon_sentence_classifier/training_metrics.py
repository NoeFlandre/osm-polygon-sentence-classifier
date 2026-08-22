"""Pure evaluation and model-card metric helpers for training."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, NoReturn, TypeGuard


class MetricsInputError(ValueError):
    """Raised when a Trainer evaluation payload is malformed."""


_INVALID_INPUT_MESSAGE = "evaluation predictions and labels are invalid"


def _is_card_scalar(value: object) -> bool:
    if value is None or isinstance(value, (bool, int, str)):
        return True
    return isinstance(value, float) and math.isfinite(value)


def latest_training_metrics(state: Any) -> dict[str, object]:
    """Return scalar values from the latest Trainer log entry."""

    return _latest_metric_entry(state, _is_training_entry)


def latest_evaluation_metrics(state: Any) -> dict[str, object]:
    """Return scalar values from the latest Trainer evaluation log entry."""

    return _latest_metric_entry(state, _is_evaluation_entry)


def _state_history(state: Any) -> Sequence[Any]:
    try:
        history = state.log_history
    except AttributeError:
        return ()
    return history if isinstance(history, Sequence) else ()


def _is_training_entry(entry: Mapping[object, object]) -> bool:
    del entry
    return True


def _is_evaluation_entry(entry: Mapping[object, object]) -> bool:
    return any(isinstance(key, str) and key.startswith("eval_") for key in entry)


def _scalar_metrics(entry: Mapping[object, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in entry.items()
        if isinstance(key, str) and _is_card_scalar(value)
    }


def _latest_metric_entry(
    state: Any,
    include: Any,
) -> dict[str, object]:
    history = _state_history(state)
    for entry in reversed(history):
        if isinstance(entry, Mapping) and include(entry):
            return _scalar_metrics(entry)
    return {}


def metrics_for_model_card(train_output: Any, trainer: Any) -> dict[str, object]:
    """Merge final training metrics with the latest evaluation metrics."""

    metrics: dict[str, object] = {}
    raw_training_metrics = getattr(train_output, "metrics", None)
    if isinstance(raw_training_metrics, Mapping):
        metrics.update(raw_training_metrics)
    metrics.update(latest_evaluation_metrics(getattr(trainer, "state", None)))
    return metrics


def _as_python(value: Any) -> object:
    tolist = getattr(value, "tolist", None)
    return tolist() if callable(tolist) else value


def _is_value_sequence(value: object) -> TypeGuard[Sequence[Any]]:
    if isinstance(value, (str, bytes)):
        return False
    return isinstance(value, Sequence)


def _invalid_metrics_input() -> NoReturn:
    raise MetricsInputError(_INVALID_INPUT_MESSAGE)


def _required_evaluation_value(eval_prediction: Any, name: str) -> object:
    try:
        value = getattr(eval_prediction, name)
    except AttributeError:
        return None
    return _as_python(value)


def _raw_evaluation_inputs(eval_prediction: Any) -> tuple[object, object]:
    predictions = _required_evaluation_value(eval_prediction, "predictions")
    labels = _required_evaluation_value(eval_prediction, "label_ids")
    if isinstance(predictions, tuple) and predictions:
        predictions = _as_python(predictions[0])
    return predictions, labels


def _has_matching_inputs(predictions: Sequence[Any], labels: Sequence[Any]) -> bool:
    return bool(predictions) and len(predictions) == len(labels)


def _evaluation_inputs(eval_prediction: Any) -> tuple[Sequence[Any], Sequence[Any]]:
    predictions, labels = _raw_evaluation_inputs(eval_prediction)
    if not _is_value_sequence(predictions):
        _invalid_metrics_input()
    if not _is_value_sequence(labels):
        _invalid_metrics_input()
    if not _has_matching_inputs(predictions, labels):
        _invalid_metrics_input()
    return predictions, labels


def _predicted_label(logits: Any) -> int:
    row = _as_python(logits)
    if not _is_value_sequence(row) or not row:
        _invalid_metrics_input()
    predicted = max(range(len(row)), key=lambda index: row[index])
    if predicted not in (0, 1):
        _invalid_metrics_input()
    return predicted


def _actual_label(label: Any) -> int:
    actual = _as_python(label)
    if isinstance(actual, bool) or not isinstance(actual, int) or actual not in (0, 1):
        _invalid_metrics_input()
    return actual


def _classification_labels(eval_prediction: Any) -> tuple[list[int], list[int]]:
    predictions, labels = _evaluation_inputs(eval_prediction)

    predicted_labels: list[int] = []
    actual_labels: list[int] = []
    for index in range(len(predictions)):
        logits = predictions[index]
        label = labels[index]
        predicted_labels.append(_predicted_label(logits))
        actual_labels.append(_actual_label(label))
    return predicted_labels, actual_labels


def _count_pairs(
    predicted_labels: Sequence[int],
    actual_labels: Sequence[int],
    expected_predicted: int,
    expected_actual: int,
) -> int:
    return sum(
        predicted_labels[index] == expected_predicted
        and actual_labels[index] == expected_actual
        for index in range(len(predicted_labels))
    )


def _confusion_counts(
    predicted_labels: Sequence[int], actual_labels: Sequence[int]
) -> tuple[int, int, int, int]:
    return (
        _count_pairs(predicted_labels, actual_labels, 1, 1),
        _count_pairs(predicted_labels, actual_labels, 0, 0),
        _count_pairs(predicted_labels, actual_labels, 1, 0),
        _count_pairs(predicted_labels, actual_labels, 0, 1),
    )


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _f1_for_class(
    true_positive: int, false_positive: int, false_negative: int
) -> float:
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    return _safe_ratio(2 * precision * recall, precision + recall)


def classification_metrics(eval_prediction: Any) -> dict[str, float]:
    """Compute binary accuracy, precision, recall, and F1 for Trainer evals."""

    predicted_labels, actual_labels = _classification_labels(eval_prediction)
    true_positive, true_negative, false_positive, false_negative = _confusion_counts(
        predicted_labels, actual_labels
    )
    accuracy = _safe_ratio(true_positive + true_negative, len(actual_labels))
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    f1 = _safe_ratio(2 * precision * recall, precision + recall)
    negative_support = true_negative + false_positive
    positive_support = true_positive + false_negative

    negative_f1 = _f1_for_class(true_negative, false_positive, false_negative)
    positive_f1 = _f1_for_class(true_positive, false_negative, false_positive)
    negative_recall = _safe_ratio(true_negative, negative_support)
    positive_recall = recall
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": (negative_f1 + positive_f1) / 2,
        "balanced_accuracy": (negative_recall + positive_recall) / 2,
        "negative_support": float(negative_support),
        "positive_support": float(positive_support),
    }
