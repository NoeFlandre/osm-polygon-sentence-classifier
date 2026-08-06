from pathlib import Path
from threading import RLock
from types import SimpleNamespace
from typing import Any

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


def test_settings_for_uses_an_explicit_study_project_when_requested() -> None:
    settings = settings_for(
        ProjectConfig(),
        project="landuse-ablation-study-v1",
    )

    assert settings.project == "landuse-ablation-study-v1"
    assert settings.space_id == TRACKIO_SPACE_ID
    assert settings.bucket_id == TRACKIO_BUCKET_ID


def test_settings_for_rejects_an_explicit_empty_project_name() -> None:
    with pytest.raises(TrackingError, match="non-empty"):
        settings_for(ProjectConfig(), project="")


def test_restore_static_snapshot_merges_existing_bucket_data(
    tmp_path,
    monkeypatch,
) -> None:
    from osm_polygon_sentence_classifier import tracking

    settings = TrackioSettings(
        project="osm-polygon-sentence-classifier",
        directory=tmp_path / "tracking",
    )
    calls: list[Any] = []

    class FakeHub:
        @staticmethod
        def download_bucket_files(bucket_id, files, **kwargs) -> None:
            calls.append((bucket_id, files, kwargs))
            for _remote, local in files:
                path = tmp_path / "tracking" / Path(local).name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"snapshot")

    class FakeStorage:
        @staticmethod
        def import_from_parquet() -> None:
            calls.append("import")

    monkeypatch.setattr(
        tracking,
        "import_module",
        lambda name: FakeHub
        if name == "huggingface_hub"
        else SimpleNamespace(SQLiteStorage=FakeStorage),
    )

    tracking.restore_static_project_snapshot(settings)

    assert calls[0][0] == settings.bucket_id
    assert calls[-1] == "import"
    assert (settings.directory / "osm-polygon-sentence-classifier.parquet").is_file()


def test_tracking_sync_uses_the_dedicated_static_space_and_bucket(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeFragments:
        @staticmethod
        def import_inbox_dir() -> None:
            pass

    class FakeTrackio:
        fragments = FakeFragments

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

    class FakeFragments:
        @staticmethod
        def import_inbox_dir() -> None:
            calls.append("import")

    class FakeTrackio:
        context_vars = SimpleNamespace(
            current_run=SimpleNamespace(get=lambda: run),
        )
        fragments = FakeFragments

        @staticmethod
        def sync(**kwargs: object) -> str:
            del kwargs
            calls.append("sync")
            return TRACKIO_STATIC_SPACE_ID

    monkeypatch.setitem(__import__("sys").modules, "trackio", FakeTrackio)

    sync_project_to_static_space(settings_for(ProjectConfig()))

    assert calls == ["flush", "import", "sync"]


def test_final_tracking_sync_finishes_run_before_importing_fragments(
    monkeypatch,
) -> None:
    calls: list[str] = []

    class FakeRun:
        def __init__(self) -> None:
            self._client_lock = RLock()

        def _flush_queues_inline(self) -> None:
            calls.append("flush")

    run = FakeRun()

    class FakeFragments:
        @staticmethod
        def import_inbox_dir() -> None:
            calls.append("import")

    class FakeTrackio:
        context_vars = SimpleNamespace(
            current_run=SimpleNamespace(get=lambda: run),
        )
        fragments = FakeFragments

        @staticmethod
        def finish() -> None:
            calls.append("finish")

        @staticmethod
        def sync(**kwargs: object) -> str:
            del kwargs
            calls.append("sync")
            return TRACKIO_STATIC_SPACE_ID

    monkeypatch.setitem(__import__("sys").modules, "trackio", FakeTrackio)

    sync_project_to_static_space(settings_for(ProjectConfig()), finalize=True)

    assert calls == ["finish", "import", "sync"]


def test_tracking_sync_wraps_trackio_failures(monkeypatch) -> None:
    class FakeFragments:
        @staticmethod
        def import_inbox_dir() -> None:
            pass

    class BrokenTrackio:
        fragments = FakeFragments

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
                "private": False,
                "exist_ok": True,
            },
        ),
        (
            "bucket",
            {"bucket_id": TRACKIO_BUCKET_ID, "private": False, "exist_ok": True},
        ),
    ]

    assert TRACKIO_SPACE_ID == TRACKIO_STATIC_SPACE_ID
    assert TRACKIO_SPACE_ID.endswith("-trackio")
