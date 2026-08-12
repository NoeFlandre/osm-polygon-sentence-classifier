"""Model parameter-freezing policies for landuse classifier training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, cast

TrainableLayers = Literal["head", "last2"]


class TrainingError(RuntimeError):
    """Raised when training dependencies or configuration are unavailable."""


def configure_trainable_layers(
    model: Any,
    trainable_layers: TrainableLayers | None,
) -> None:
    """Configure which model layers remain trainable for one run."""

    if trainable_layers in {None, "head"}:
        _freeze_encoder_for_head_training(model)
        return
    if trainable_layers != "last2":
        raise TrainingError("unsupported trainable layer mode")
    _configure_last_two_layers(model)


def _freeze_encoder_for_head_training(model: Any) -> None:
    """Freeze the encoder while leaving its classification head trainable."""

    parameters = getattr(model, "parameters", None)
    head = getattr(model, "head", None)
    classifier = getattr(model, "classifier", None)
    head_parameters = getattr(head, "parameters", None)
    classifier_parameters = getattr(classifier, "parameters", None)
    if (
        not callable(parameters)
        or classifier is None
        or not callable(classifier_parameters)
    ):
        raise TrainingError(
            "model must expose a parameters() method and classifier head"
        )

    for parameter in parameters():
        parameter.requires_grad = False
    if callable(head_parameters):
        for parameter in head_parameters():
            parameter.requires_grad = True
    for parameter in classifier_parameters():
        parameter.requires_grad = True


def _encoder_layers(model: Any) -> Sequence[Any]:
    base_model = getattr(model, "base_model", None)
    candidates = (
        getattr(base_model, "layers", None),
        getattr(getattr(base_model, "encoder", None), "layer", None),
        getattr(getattr(base_model, "model", None), "layers", None),
        getattr(getattr(model, "model", None), "layers", None),
        getattr(model, "layers", None),
    )
    for candidate in candidates:
        if isinstance(candidate, Sequence) and not isinstance(candidate, (str, bytes)):
            return candidate
        # torch.nn.ModuleList is ordered and sliceable, but it does not register
        # as collections.abc.Sequence at runtime.
        if (
            not isinstance(candidate, (str, bytes, Mapping))
            and callable(getattr(candidate, "__len__", None))
            and callable(getattr(candidate, "__getitem__", None))
        ):
            return cast(Sequence[Any], candidate)
    raise TrainingError("model does not expose ordered encoder layers")


def _configure_last_two_layers(model: Any) -> None:
    _freeze_all_parameters(model)
    layers = _encoder_layers(model)
    if len(layers) < 2:
        raise TrainingError("model must expose at least two encoder layers")
    _set_layer_parameters(layers, requires_grad=False)
    _set_layer_parameters(layers[-2:], requires_grad=True)
    _enable_classifier_heads(model)


def _freeze_all_parameters(model: Any) -> None:
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        raise TrainingError("model must expose a parameters() method")
    for parameter in parameters():
        parameter.requires_grad = False


def _set_layer_parameters(layers: Sequence[Any], *, requires_grad: bool) -> None:
    for layer in layers:
        layer_parameters = getattr(layer, "parameters", None)
        if not callable(layer_parameters):
            raise TrainingError("encoder layer does not expose parameters()")
        for parameter in layer_parameters():
            parameter.requires_grad = requires_grad


def _enable_classifier_heads(model: Any) -> None:
    head = getattr(model, "head", None)
    classifier = getattr(model, "classifier", None)
    for module in (head, classifier):
        module_parameters = getattr(module, "parameters", None)
        if callable(module_parameters):
            for parameter in module_parameters():
                parameter.requires_grad = True


__all__ = ["TrainingError", "TrainableLayers", "configure_trainable_layers"]
