from threading import RLock
from types import SimpleNamespace

import pytest

from osm_polygon_sentence_classifier.config import ProjectConfig
from osm_polygon_sentence_classifier.tracking import (
    TRACKING_SUBDIRECTORY,
    TRACKIO_BUCKET_ID,
    TRACKIO_SPACE_ID,
    TRACKIO_STATIC_SPACE_ID,
    TrackingError,
    TrackioSettings,
    ensure_trackio_resources,
    settings_for,
    sync_project_to_static_space,
)


def test_tracking_settings_use_the_project_name_and_managed_directory() -> None:
    config = ProjectConfig()
    settings = settings_for(config)
    assert isinstance(settings, TrackioSettings)
    assert settings.project == "osm-polygon-sentence-classifier"
    assert settings.directory == config.data_root / TRACKING_SUBDIRECTORY
    assert settings.directory.is_relative_to(config.data_root)
    assert settings.space_id == TRACKIO_SPACE_ID
    assert settings.bucket_id == TRACKIO_BUCKET_ID


def test_tracking_environment_only_points_trackio_at_managed_storage() -> None:
    settings = settings_for(ProjectConfig())
    assert settings.environment() == {"TRACKIO_DIR": str(settings.directory)}


def test_tracking_sync_uses_the_dedicated_static_space_and_bucket(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeTrackio:
        @staticmethod
        def sync(**kwargs: object) -> str:
            calls.append(kwargs)
            return TRACKIO_STATIC_SPACE_ID

    monkeypatch.setitem(__import__("sys").modules, "trackio", FakeTrackio)

    result = sync_project_to_static_space(settings_for(ProjectConfig()))

    assert result == TRACKIO_STATIC_SPACE_ID
    assert calls == [
        {
            "project": "osm-polygon-sentence-classifier",
            "space_id": TRACKIO_STATIC_SPACE_ID,
            "bucket_id": TRACKIO_BUCKET_ID,
            "sdk": "static",
            "force": True,
        }
    ]


def test_tracking_sync_flushes_queued_metrics_before_static_upload(monkeypatch) -> None:
    calls: list[str] = []

    class FakeRun:
        def __init__(self) -> None:
            self._client_lock = RLock()

        def _flush_queues_inline(self) -> None:
            calls.append("flush")

    run = FakeRun()

    class FakeTrackio:
        context_vars = SimpleNamespace(
            current_run=SimpleNamespace(get=lambda: run),
        )

        @staticmethod
        def sync(**kwargs: object) -> str:
            del kwargs
            calls.append("sync")
            return TRACKIO_STATIC_SPACE_ID

    monkeypatch.setitem(__import__("sys").modules, "trackio", FakeTrackio)

    sync_project_to_static_space(settings_for(ProjectConfig()))

    assert calls == ["flush", "sync"]


def test_tracking_sync_wraps_trackio_failures(monkeypatch) -> None:
    class BrokenTrackio:
        @staticmethod
        def sync(**kwargs: object) -> str:
            del kwargs
            raise RuntimeError("network")

    monkeypatch.setitem(__import__("sys").modules, "trackio", BrokenTrackio)

    with pytest.raises(TrackingError, match="static Space"):
        sync_project_to_static_space(settings_for(ProjectConfig()))


def test_trackio_resource_setup_creates_the_free_static_space_and_bucket() -> None:
    calls: list[tuple[str, dict[str, object]]] = []

    class FakeHub:
        def create_repo(self, **kwargs: object) -> None:
            calls.append(("repo", kwargs))

        def create_bucket(self, **kwargs: object) -> None:
            calls.append(("bucket", kwargs))

    ensure_trackio_resources(
        settings_for(ProjectConfig()),
        hub_api=FakeHub(),
    )

    assert calls == [
        (
            "repo",
            {
                "repo_id": TRACKIO_SPACE_ID,
                "repo_type": "space",
                "space_sdk": "static",
                "exist_ok": True,
            },
        ),
        (
            "bucket",
            {"bucket_id": TRACKIO_BUCKET_ID, "exist_ok": True},
        ),
    ]

    assert TRACKIO_SPACE_ID == TRACKIO_STATIC_SPACE_ID
    assert TRACKIO_SPACE_ID.endswith("-trackio")
