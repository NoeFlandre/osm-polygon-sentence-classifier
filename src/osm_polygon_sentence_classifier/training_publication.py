"""Checkpoint manifests, publication, and static tracking during training."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import training_metrics as _training_metrics
from .checkpointing import CheckpointError, write_checkpoint_manifest
from .huggingface_http import is_rate_limit_error
from .publication import (
    ModelPublicationError,
    publish_checkpoint_directory,
    render_model_card,
)
from .tracking import TrackingError, TrackioSettings, sync_project_to_static_space
from .training_freezing import TrainingError

_LOGGER = logging.getLogger(__name__)


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
        if not finalize and is_rate_limit_error(error):
            _LOGGER.warning(
                "Trackio rate limit reached; retaining the local snapshot and "
                "continuing without this checkpoint sync"
            )
            return
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
        hub_checkpoint_steps: int = 1,
    ) -> None:
        if (
            isinstance(hub_checkpoint_steps, bool)
            or not isinstance(hub_checkpoint_steps, int)
            or hub_checkpoint_steps <= 0
        ):
            raise TrainingError("hub_checkpoint_steps must be a positive integer")
        self.identity = dict(identity)
        self.model_repository_id = model_repository_id
        self.trackio_space_id = trackio_space_id
        self.tracking_settings = tracking_settings
        self.hub_api = hub_api
        self.hub_checkpoint_steps = hub_checkpoint_steps
        self._pending_publications: list[Any] = []
        self._hub_rate_limited = False

    def _mark_hub_rate_limited(self) -> None:
        if not self._hub_rate_limited:
            _LOGGER.warning(
                "Hugging Face rate limit reached; retaining local checkpoints "
                "and continuing without further checkpoint commits"
            )
        self._hub_rate_limited = True

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
            if is_rate_limit_error(error):
                self._mark_hub_rate_limited()
                return
            raise TrainingError("checkpoint model publication failed") from error

    def _publish_checkpoint(self, checkpoint: Path) -> None:
        if self._hub_rate_limited:
            return
        try:
            publish_checkpoint_directory(
                checkpoint,
                self.model_repository_id,
                identity=self.identity,
            )
        except ModelPublicationError as error:
            if is_rate_limit_error(error):
                self._mark_hub_rate_limited()
                return
            raise TrainingError("checkpoint model publication failed") from error

    @staticmethod
    def _checkpoint_save_inputs(args: Any, state: Any) -> tuple[Path, int]:
        output_directory = getattr(args, "output_dir", None)
        global_step = getattr(state, "global_step", None)
        if not isinstance(output_directory, str) or not isinstance(global_step, int):
            raise TrainingError("checkpoint save did not expose a valid step")
        return Path(output_directory), global_step

    def _write_checkpoint_manifest(self, checkpoint: Path, global_step: int) -> None:
        try:
            write_checkpoint_manifest(
                checkpoint,
                identity=self.identity,
                global_step=global_step,
            )
        except CheckpointError as error:
            raise TrainingError("checkpoint manifest could not be written") from error

    def _write_checkpoint_card(
        self,
        checkpoint: Path,
        *,
        state: Any,
        global_step: int,
    ) -> None:
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
            raise TrainingError("checkpoint model card could not be written") from error

    def _queue_checkpoint_publication(
        self,
        args: Any,
        checkpoint: Path,
    ) -> None:
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
        if self._publication_limit_reached(args):
            self._wait_for_next_publication()

    def _publication_limit_reached(self, args: Any) -> bool:
        save_total_limit = getattr(args, "save_total_limit", None)
        return (
            isinstance(save_total_limit, int)
            and not isinstance(save_total_limit, bool)
            and save_total_limit > 0
            and len(self._pending_publications) >= save_total_limit
        )

    def _publish_remote_checkpoint(
        self,
        args: Any,
        checkpoint: Path,
        *,
        state: Any,
        global_step: int,
    ) -> None:
        if self.model_repository_id is None:
            return
        self._write_checkpoint_card(checkpoint, state=state, global_step=global_step)
        if self.hub_api is None:
            self._publish_checkpoint(checkpoint)
            return
        self._queue_checkpoint_publication(args, checkpoint)

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: object) -> Any:
        del kwargs
        output_directory, global_step = self._checkpoint_save_inputs(args, state)
        checkpoint = output_directory / f"checkpoint-{global_step}"
        self._write_checkpoint_manifest(checkpoint, global_step)
        remote_checkpoint = global_step % self.hub_checkpoint_steps == 0
        if remote_checkpoint:
            self._publish_remote_checkpoint(
                args,
                checkpoint,
                state=state,
                global_step=global_step,
            )
        if self.tracking_settings is not None and remote_checkpoint:
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
    hub_checkpoint_steps: int = 1,
) -> Any:
    """Bind the checkpoint callback to an optional Trainer callback base."""

    if trainer_callback is None:
        return CheckpointManifestCallback(
            identity,
            model_repository_id=model_repository_id,
            trackio_space_id=trackio_space_id,
            tracking_settings=tracking_settings,
            hub_api=hub_api,
            hub_checkpoint_steps=hub_checkpoint_steps,
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
        hub_checkpoint_steps=hub_checkpoint_steps,
    )


__all__ = [
    "CheckpointManifestCallback",
    "make_checkpoint_manifest_callback",
    "write_model_card",
]
