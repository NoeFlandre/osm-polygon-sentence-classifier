"""Pure evaluation and model-card metric helpers for training."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


class MetricsInputError(ValueError):
    """Raised when a Trainer evaluation payload is malformed."""


def _is_card_scalar(value: object) -> bool:
    if value is None or isinstance(value, (bool, int, str)):
        return True
    return isinstance(value, float) and math.isfinite(value)


def latest_training_metrics(state: Any) -> dict[str, object]:
    """Return scalar values from the latest Trainer log entry."""

    history = getattr(state, "log_history", ())
    if not isinstance(history, Sequence):
        return {}
    for entry in reversed(history):
        if isinstance(entry, Mapping):
            return {
                key: value
                for key, value in entry.items()
                if isinstance(key, str) and _is_card_scalar(value)
            }
    return {}


def latest_evaluation_metrics(state: Any) -> dict[str, object]:
    """Return scalar values from the latest Trainer evaluation log entry."""

    history = getattr(state, "log_history", ())
    if not isinstance(history, Sequence):
        return {}
    for entry in reversed(history):
        if isinstance(entry, Mapping) and any(
            isinstance(key, str) and key.startswith("eval_") for key in entry
        ):
            return {
                key: value
                for key, value in entry.items()
                if isinstance(key, str) and _is_card_scalar(value)
            }
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


def classification_metrics(eval_prediction: Any) -> dict[str, float]:
    """Compute binary accuracy, precision, recall, and F1 for Trainer evals."""

    predictions = _as_python(getattr(eval_prediction, "predictions", None))
    labels = _as_python(getattr(eval_prediction, "label_ids", None))
    if isinstance(predictions, tuple) and predictions:
        predictions = _as_python(predictions[0])
    if (
        not isinstance(predictions, Sequence)
        or isinstance(predictions, (str, bytes))
        or not isinstance(labels, Sequence)
        or isinstance(labels, (str, bytes))
        or len(predictions) != len(labels)
        or not predictions
    ):
        raise MetricsInputError("evaluation predictions and labels are invalid")

    predicted_labels: list[int] = []
    actual_labels: list[int] = []
    for logits, label in zip(predictions, labels, strict=True):
        row = _as_python(logits)
        actual = _as_python(label)
        if (
            not isinstance(row, Sequence)
            or isinstance(row, (str, bytes))
            or not row
            or isinstance(actual, bool)
            or not isinstance(actual, int)
            or actual not in (0, 1)
        ):
            raise MetricsInputError("evaluation predictions and labels are invalid")
        predicted = max(range(len(row)), key=lambda index: row[index])
        if predicted not in (0, 1):
            raise MetricsInputError("evaluation predictions and labels are invalid")
        predicted_labels.append(predicted)
        actual_labels.append(actual)

    true_positive = sum(
        predicted == actual == 1
        for predicted, actual in zip(predicted_labels, actual_labels, strict=True)
    )
    true_negative = sum(
        predicted == actual == 0
        for predicted, actual in zip(predicted_labels, actual_labels, strict=True)
    )
    false_positive = sum(
        predicted == 1 and actual == 0
        for predicted, actual in zip(predicted_labels, actual_labels, strict=True)
    )
    false_negative = sum(
        predicted == 0 and actual == 1
        for predicted, actual in zip(predicted_labels, actual_labels, strict=True)
    )
    accuracy = (true_positive + true_negative) / len(actual_labels)
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    negative_support = true_negative + false_positive
    positive_support = true_positive + false_negative

    def _f1_for_class(
        true_positive_: int, false_positive_: int, false_negative_: int
    ) -> float:
        precision_ = (
            true_positive_ / (true_positive_ + false_positive_)
            if true_positive_ + false_positive_
            else 0.0
        )
        recall_ = (
            true_positive_ / (true_positive_ + false_negative_)
            if true_positive_ + false_negative_
            else 0.0
        )
        return (
            2 * precision_ * recall_ / (precision_ + recall_)
            if precision_ + recall_
            else 0.0
        )

    negative_f1 = _f1_for_class(true_negative, false_positive, false_negative)
    positive_f1 = _f1_for_class(true_positive, false_negative, false_positive)
    negative_recall = true_negative / negative_support if negative_support else 0.0
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
