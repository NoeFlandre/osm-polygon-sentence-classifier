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


def test_tracking_sync_wraps_trackio_failures(monkeypatch) -> None:
    class BrokenTrackio:
        @staticmethod
        def sync(**kwargs: object) -> str:
            del kwargs
            raise RuntimeError("network")

    monkeypatch.setitem(__import__("sys").modules, "trackio", BrokenTrackio)

    with pytest.raises(TrackingError, match="static Space"):
        sync_project_to_static_space(settings_for(ProjectConfig()))


def test_trackio_resource_setup_creates_the_live_space_and_bucket() -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    deploy_calls: list[tuple[str, dict[str, object]]] = []

    class FakeHub:
        def create_repo(self, **kwargs: object) -> None:
            calls.append(("repo", kwargs))

        def create_bucket(self, **kwargs: object) -> None:
            calls.append(("bucket", kwargs))

    def deploy_space(space_id: str, **kwargs: object) -> None:
        deploy_calls.append((space_id, kwargs))

    ensure_trackio_resources(
        settings_for(ProjectConfig()),
        hub_api=FakeHub(),
        deploy_space=deploy_space,
    )

    assert calls == [
        (
            "repo",
            {
                "repo_id": TRACKIO_SPACE_ID,
                "repo_type": "space",
                "space_sdk": "gradio",
                "exist_ok": True,
            },
        ),
        (
            "bucket",
            {"bucket_id": TRACKIO_BUCKET_ID, "exist_ok": True},
        ),
    ]

    assert TRACKIO_SPACE_ID.endswith("-trackio-live")
    assert TRACKIO_SPACE_ID != TRACKIO_STATIC_SPACE_ID
    assert deploy_calls == [(TRACKIO_SPACE_ID, {"bucket_id": TRACKIO_BUCKET_ID})]
