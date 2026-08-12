"""Checkpoint manifests, publication, and static tracking during training."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import training_metrics as _training_metrics
from .checkpointing import CheckpointError, write_checkpoint_manifest
from .publication import (
    ModelPublicationError,
    publish_checkpoint_directory,
    render_model_card,
)
from .tracking import TrackingError, TrackioSettings, sync_project_to_static_space
from .training_freezing import TrainingError


def write_model_card(
    directory: Path,
    *,
    identity: Mapping[str, object],
    training_metrics: Mapping[str, object] | None = None,
    checkpoint_step: int | None = None,
    trackio_space_id: str | None = None,
) -> None:
    """Write the credential-free README for a model or checkpoint directory."""

    card = render_model_card(
        identity=identity,
        training_metrics=training_metrics,
        checkpoint_step=checkpoint_step,
        trackio_space_id=trackio_space_id,
    )
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
    (directory / "README.md").write_text(card, encoding="utf-8")


def _sync_static_trackio(
    settings: TrackioSettings,
    *,
    failure_message: str,
    finalize: bool = False,
) -> None:
    try:
        sync_project_to_static_space(settings, finalize=finalize)
    except TrackingError as error:
        raise TrainingError(failure_message) from error


class CheckpointManifestCallback:
    """Record, publish, and track one checkpoint after Trainer saves it."""

    def __init__(
        self,
        identity: Mapping[str, object],
        *,
        model_repository_id: str | None = None,
        trackio_space_id: str | None = None,
        tracking_settings: TrackioSettings | None = None,
        hub_api: Any | None = None,
    ) -> None:
        self.identity = dict(identity)
        self.model_repository_id = model_repository_id
        self.trackio_space_id = trackio_space_id
        self.tracking_settings = tracking_settings
        self.hub_api = hub_api
        self._pending_publications: list[Any] = []

    def on_init_end(self, args: Any, state: Any, control: Any, **kwargs: object) -> Any:
        del args, state, kwargs
        return control

    def _wait_for_next_publication(self) -> None:
        if not self._pending_publications:
            return
        future = self._pending_publications.pop(0)
        try:
            future.result()
        except Exception as error:
            raise TrainingError("checkpoint model publication failed") from error

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: object) -> Any:
        del kwargs
        output_directory = getattr(args, "output_dir", None)
        global_step = getattr(state, "global_step", None)
        if not isinstance(output_directory, str) or not isinstance(global_step, int):
            raise TrainingError("checkpoint save did not expose a valid step")
        try:
            write_checkpoint_manifest(
                Path(output_directory) / f"checkpoint-{global_step}",
                identity=self.identity,
                global_step=global_step,
            )
        except CheckpointError as error:
            raise TrainingError("checkpoint manifest could not be written") from error
        if self.model_repository_id is not None:
            checkpoint = Path(output_directory) / f"checkpoint-{global_step}"
            try:
                write_model_card(
                    checkpoint,
                    identity=self.identity,
                    training_metrics={
                        **_training_metrics.latest_training_metrics(state),
                        **_training_metrics.latest_evaluation_metrics(state),
                    },
                    checkpoint_step=global_step,
                    trackio_space_id=self.trackio_space_id,
                )
            except OSError as error:
                raise TrainingError(
                    "checkpoint model card could not be written"
                ) from error
            if self.hub_api is None:
                try:
                    publish_checkpoint_directory(
                        checkpoint,
                        self.model_repository_id,
                        identity=self.identity,
                    )
                except ModelPublicationError as error:
                    raise TrainingError(
                        "checkpoint model publication failed"
                    ) from error
            else:
                run_as_future = getattr(self.hub_api, "run_as_future", None)
                if not callable(run_as_future):
                    raise TrainingError(
                        "checkpoint publication API cannot queue background work"
                    )
                self._pending_publications.append(
                    run_as_future(
                        publish_checkpoint_directory,
                        checkpoint,
                        self.model_repository_id,
                        identity=self.identity,
                        hub_api=self.hub_api,
                    )
                )
                save_total_limit = getattr(args, "save_total_limit", None)
                if (
                    isinstance(save_total_limit, int)
                    and not isinstance(save_total_limit, bool)
                    and save_total_limit > 0
                    and len(self._pending_publications) >= save_total_limit
                ):
                    self._wait_for_next_publication()
        if self.tracking_settings is not None:
            _sync_static_trackio(
                self.tracking_settings,
                failure_message="checkpoint Trackio static snapshot failed",
                finalize=False,
            )
        return control

    def on_train_end(
        self, args: Any, state: Any, control: Any, **kwargs: object
    ) -> Any:
        del args, state, kwargs
        while self._pending_publications:
            self._wait_for_next_publication()
        return control


def make_checkpoint_manifest_callback(
    identity: Mapping[str, object],
    trainer_callback: Any | None,
    *,
    model_repository_id: str | None = None,
    trackio_space_id: str | None = None,
    tracking_settings: TrackioSettings | None = None,
    hub_api: Any | None = None,
) -> Any:
    """Bind the checkpoint callback to an optional Trainer callback base."""

    if trainer_callback is None:
        return CheckpointManifestCallback(
            identity,
            model_repository_id=model_repository_id,
            trackio_space_id=trackio_space_id,
            tracking_settings=tracking_settings,
            hub_api=hub_api,
        )
    callback_type = type(
        "_BoundCheckpointManifestCallback",
        (CheckpointManifestCallback, trainer_callback),
        {},
    )
    return callback_type(
        identity,
        model_repository_id=model_repository_id,
        trackio_space_id=trackio_space_id,
        tracking_settings=tracking_settings,
        hub_api=hub_api,
    )


__all__ = [
    "CheckpointManifestCallback",
    "make_checkpoint_manifest_callback",
    "write_model_card",
]
