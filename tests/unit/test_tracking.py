from pathlib import Path
from threading import RLock
from types import SimpleNamespace
from typing import Any, cast

import pytest

from osm_polygon_sentence_classifier import tracking
from osm_polygon_sentence_classifier.config import ProjectConfig
from osm_polygon_sentence_classifier.tracking import (
    TRACKING_SUBDIRECTORY,
    TRACKIO_BUCKET_ID,
    TRACKIO_SPACE_ID,
    TRACKIO_STATIC_SPACE_ID,
    V2_TRACKIO_BUCKET_ID,
    V2_TRACKIO_SPACE_ID,
    V2_TRACKIO_STATIC_SPACE_ID,
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


def test_settings_for_separates_worldwide_v2_from_the_v1_dashboard() -> None:
    config = ProjectConfig()
    settings = settings_for(config, project="place-relevance-v2")

    assert settings.project == "place-relevance-v2"
    assert settings.directory == config.data_root / TRACKING_SUBDIRECTORY
    assert settings.space_id == V2_TRACKIO_SPACE_ID
    assert settings.static_space_id == V2_TRACKIO_STATIC_SPACE_ID
    assert settings.bucket_id == V2_TRACKIO_BUCKET_ID


def test_settings_for_separates_worldwide_v2_ablations_from_the_v1_dashboard() -> None:
    settings = settings_for(
        ProjectConfig(),
        project="place-relevance-v2-ablations",
    )

    assert settings.space_id == V2_TRACKIO_SPACE_ID
    assert settings.static_space_id == V2_TRACKIO_STATIC_SPACE_ID
    assert settings.bucket_id == V2_TRACKIO_BUCKET_ID


def test_settings_for_rejects_an_explicit_empty_project_name() -> None:
    with pytest.raises(TrackingError) as error:
        settings_for(ProjectConfig(), project="")

    assert str(error.value) == "Trackio project must be a non-empty string"


@pytest.mark.parametrize("project", ["   ", 42])
def test_settings_for_rejects_invalid_project_names(project: object) -> None:
    with pytest.raises(TrackingError) as error:
        settings_for(ProjectConfig(), project=cast(Any, project))

    assert str(error.value) == "Trackio project must be a non-empty string"


@pytest.mark.parametrize("project", ["line\nbreak", "line\rbreak"])
def test_settings_for_rejects_multiline_project_names(project: str) -> None:
    with pytest.raises(TrackingError) as error:
        settings_for(ProjectConfig(), project=project)

    assert str(error.value) == "Trackio project must be a single-line string"


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


def test_restore_static_snapshot_creates_missing_parent_directories(
    tmp_path,
    monkeypatch,
) -> None:
    settings = TrackioSettings(
        project="study",
        directory=tmp_path / "missing" / "parents" / "tracking",
    )
    calls: list[str] = []
    monkeypatch.setattr(
        tracking,
        "_download_snapshot",
        lambda bucket_id, files: calls.append(f"download:{bucket_id}:{len(files)}"),
    )
    monkeypatch.setattr(
        tracking,
        "_import_snapshot",
        lambda: calls.append("import"),
    )

    tracking.restore_static_project_snapshot(settings)

    assert settings.directory.is_dir()
    assert calls == [f"download:{settings.bucket_id}:4", "import"]


def test_restore_static_snapshot_allows_an_existing_directory(
    tmp_path,
    monkeypatch,
) -> None:
    settings = TrackioSettings(project="study", directory=tmp_path / "tracking")
    settings.directory.mkdir()
    monkeypatch.setattr(tracking, "_download_snapshot", lambda bucket_id, files: None)
    monkeypatch.setattr(tracking, "_import_snapshot", lambda: None)

    tracking.restore_static_project_snapshot(settings)


def test_restore_static_snapshot_preserves_the_failure_message_and_cause(
    tmp_path,
    monkeypatch,
) -> None:
    settings = TrackioSettings(project="study", directory=tmp_path / "tracking")

    def fail_download(bucket_id, files) -> None:
        raise RuntimeError("download failed")

    monkeypatch.setattr(tracking, "_download_snapshot", fail_download)

    with pytest.raises(TrackingError) as error:
        tracking.restore_static_project_snapshot(settings)

    assert str(error.value) == "Trackio static snapshot restoration failed"
    assert isinstance(error.value.__cause__, RuntimeError)


def test_snapshot_files_use_the_exact_remote_and_local_names(tmp_path) -> None:
    settings = TrackioSettings(
        project="XX/Study/Run",
        directory=tmp_path / "tracking",
    )

    assert tracking._snapshot_files(settings) == [
        ("metrics.parquet", settings.directory / "XX_Study_Run.parquet"),
        (
            "aux/system_metrics.parquet",
            settings.directory / "XX_Study_Run_system.parquet",
        ),
        (
            "aux/configs.parquet",
            settings.directory / "XX_Study_Run_configs.parquet",
        ),
        (
            "aux/traces.parquet",
            settings.directory / "XX_Study_Run_traces.parquet",
        ),
    ]


def test_snapshot_files_reject_a_project_that_has_no_safe_stem(tmp_path) -> None:
    settings = TrackioSettings(project="!!!", directory=tmp_path)

    with pytest.raises(TrackingError) as error:
        tracking._snapshot_files(settings)

    assert str(error.value) == ("Trackio project name cannot produce a local filename")


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

    with pytest.raises(TrackingError) as error:
        sync_project_to_static_space(settings_for(ProjectConfig()))

    assert str(error.value) == "Trackio static Space synchronization failed"
    assert isinstance(error.value.__cause__, RuntimeError)


def test_tracking_sync_rejects_blank_and_non_string_space_ids(monkeypatch) -> None:
    class FakeFragments:
        @staticmethod
        def import_inbox_dir() -> None:
            pass

    def fake_trackio(invalid_space_id: object) -> type:
        class FakeTrackio:
            fragments = FakeFragments

            @staticmethod
            def sync(**kwargs: object) -> object:
                del kwargs
                return invalid_space_id

        return FakeTrackio

    for invalid_space_id in ("", "   ", None, 42):
        monkeypatch.setitem(
            __import__("sys").modules,
            "trackio",
            fake_trackio(invalid_space_id),
        )

        with pytest.raises(TrackingError) as error:
            sync_project_to_static_space(settings_for(ProjectConfig()))

        assert str(error.value) == (
            "Trackio synchronization returned an invalid Space ID"
        )


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


def test_trackio_resource_setup_imports_huggingface_api_when_not_supplied(
    monkeypatch,
) -> None:
    calls: list[str] = []

    class FakeHubApi:
        def create_repo(self, **kwargs: object) -> None:
            del kwargs

        def create_bucket(self, **kwargs: object) -> None:
            del kwargs

    def fake_import(name: str) -> object:
        calls.append(name)
        assert name == "huggingface_hub"
        return SimpleNamespace(HfApi=lambda: FakeHubApi())

    monkeypatch.setattr(tracking, "configure_huggingface_http", lambda: None)
    monkeypatch.setattr(tracking, "import_module", fake_import)

    ensure_trackio_resources(settings_for(ProjectConfig()))

    assert calls == ["huggingface_hub"]


def test_trackio_resource_setup_preserves_failure_message_and_cause() -> None:
    class BrokenHub:
        def create_repo(self, **kwargs: object) -> None:
            del kwargs
            raise RuntimeError("hub unavailable")

    with pytest.raises(TrackingError) as error:
        ensure_trackio_resources(settings_for(ProjectConfig()), hub_api=BrokenHub())

    assert str(error.value) == "Trackio Space and bucket provisioning failed"
    assert isinstance(error.value.__cause__, RuntimeError)


def test_download_snapshot_uses_the_exact_optional_dependency_contract(
    tmp_path,
    monkeypatch,
) -> None:
    calls: list[tuple[str, list[tuple[str, Path]], dict[str, object]]] = []

    def download(bucket_id, files, **kwargs) -> None:
        calls.append((bucket_id, files, kwargs))

    def fake_import(name: str) -> object:
        assert name == "huggingface_hub"
        return SimpleNamespace(download_bucket_files=download)

    monkeypatch.setattr(tracking, "configure_huggingface_http", lambda: None)
    monkeypatch.setattr(tracking, "import_module", fake_import)
    files = [("metrics.parquet", tmp_path / "metrics.parquet")]

    tracking._download_snapshot("bucket", files)

    assert calls == [("bucket", files, {"raise_on_missing_files": False})]


def test_download_snapshot_reports_an_unavailable_hub_method(monkeypatch) -> None:
    monkeypatch.setattr(tracking, "configure_huggingface_http", lambda: None)
    monkeypatch.setattr(
        tracking,
        "import_module",
        lambda name: SimpleNamespace(),
    )

    with pytest.raises(TrackingError) as error:
        tracking._download_snapshot("bucket", [])

    assert str(error.value) == "Hugging Face bucket download is unavailable"


def test_import_snapshot_uses_trackio_and_imports_parquet() -> None:
    calls: list[str] = []
    storage = SimpleNamespace(
        import_from_parquet=lambda: calls.append("import"),
    )

    original_import_module = tracking.import_module

    def fake_import(name: str) -> object:
        assert name == "trackio"
        return SimpleNamespace(SQLiteStorage=storage)

    tracking.import_module = cast(Any, fake_import)
    try:
        tracking._import_snapshot()
    finally:
        tracking.import_module = original_import_module

    assert calls == ["import"]


@pytest.mark.parametrize(
    "trackio_module",
    [SimpleNamespace(), SimpleNamespace(SQLiteStorage=SimpleNamespace())],
)
def test_import_snapshot_reports_unavailable_parquet_support(trackio_module) -> None:
    original_import_module = tracking.import_module
    tracking.import_module = cast(Any, lambda name: trackio_module)
    try:
        with pytest.raises(TrackingError) as error:
            tracking._import_snapshot()
    finally:
        tracking.import_module = original_import_module

    assert str(error.value) == "Trackio Parquet import is unavailable"


def test_flush_current_local_run_uses_the_client_lock_when_present() -> None:
    calls: list[str] = []

    class Lock:
        def __enter__(self):
            calls.append("enter")
            return self

        def __exit__(self, *args) -> None:
            calls.append("exit")

    run = SimpleNamespace(
        _flush_queues_inline=lambda: calls.append("flush"),
        _client_lock=Lock(),
    )
    trackio = SimpleNamespace(
        context_vars=SimpleNamespace(
            current_run=SimpleNamespace(get=lambda: run),
        ),
    )

    tracking._flush_current_local_run(trackio)

    assert calls == ["enter", "flush", "exit"]


def test_flush_current_local_run_allows_a_run_without_a_lock() -> None:
    calls: list[str] = []
    run = SimpleNamespace(_flush_queues_inline=lambda: calls.append("flush"))
    trackio = SimpleNamespace(
        context_vars=SimpleNamespace(
            current_run=SimpleNamespace(get=lambda: run),
        ),
    )

    tracking._flush_current_local_run(trackio)

    assert calls == ["flush"]


def test_flush_current_local_run_ignores_a_missing_flush_method() -> None:
    run = SimpleNamespace()
    trackio = SimpleNamespace(
        context_vars=SimpleNamespace(
            current_run=SimpleNamespace(get=lambda: run),
        ),
    )

    tracking._flush_current_local_run(trackio)


def test_finish_current_local_run_requires_finish_support() -> None:
    run = SimpleNamespace()
    trackio = SimpleNamespace(
        context_vars=SimpleNamespace(
            current_run=SimpleNamespace(get=lambda: run),
        ),
    )

    with pytest.raises(TrackingError) as error:
        tracking._finish_current_local_run(trackio)

    assert str(error.value) == "Trackio finalization is unavailable"


def test_import_local_fragments_falls_back_to_the_trackio_fragments_module(
    monkeypatch,
) -> None:
    calls: list[str] = []

    def fake_import(name: str) -> object:
        calls.append(name)
        assert name == "trackio.fragments"
        return SimpleNamespace(import_inbox_dir=lambda: calls.append("import"))

    monkeypatch.setattr(tracking, "import_module", fake_import)

    tracking._import_local_fragments(SimpleNamespace())

    assert calls == ["trackio.fragments", "import"]


def test_import_local_fragments_reports_missing_import_support() -> None:
    with pytest.raises(TrackingError) as error:
        tracking._import_local_fragments(SimpleNamespace(fragments=SimpleNamespace()))

    assert str(error.value) == "Trackio local fragment import is unavailable"
