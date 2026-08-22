import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import osm_polygon_sentence_classifier.checkpoint_hub as checkpoint_hub
from osm_polygon_sentence_classifier.checkpoint_hub import (
    HubCheckpointError,
    PublishedCheckpoint,
    _canonical_json,
    _complete_step_files,
    _copy_and_validate_checkpoint,
    _default_api,
    _downloaded_checkpoint_source,
    _group_step_files,
    _is_complete_step,
    _manifest_matches,
    _matching_published_checkpoint,
    _published_paths,
    _reject_symlinks,
    _restore_checkpoint,
    _validate_repository_id,
    _validate_restore_output,
    latest_published_checkpoint,
    restore_published_checkpoint,
)
from osm_polygon_sentence_classifier.checkpointing import CheckpointInfo


@dataclass(frozen=True)
class _Entry:
    path: str


def test_default_manifest_loader_uses_the_hub_download_api(monkeypatch) -> None:
    calls: list[dict[str, object]] = []
    module_names: list[str] = []

    def download(**kwargs: object) -> str:
        calls.append(kwargs)
        return "/tmp/checkpoint-manifest.json"

    monkeypatch.setattr(checkpoint_hub, "configure_huggingface_http", lambda: None)
    monkeypatch.setattr(
        checkpoint_hub,
        "import_module",
        lambda name: (
            module_names.append(name) or SimpleNamespace(hf_hub_download=download)
        ),
    )

    result = checkpoint_hub._default_manifest_loader("owner/model", "run/manifest.json")

    assert result == Path("/tmp/checkpoint-manifest.json")
    assert calls == [
        {
            "repo_id": "owner/model",
            "filename": "run/manifest.json",
            "repo_type": "model",
        }
    ]
    assert module_names == ["huggingface_hub"]


@pytest.mark.parametrize(
    ("module", "message"),
    [
        (
            SimpleNamespace(),
            "Hugging Face checkpoint download is unavailable",
        ),
        (
            SimpleNamespace(hf_hub_download=lambda **_kwargs: object()),
            "checkpoint manifest download returned an invalid path",
        ),
    ],
)
def test_default_manifest_loader_rejects_invalid_hub_downloads(
    monkeypatch,
    module: object,
    message: str,
) -> None:
    monkeypatch.setattr(checkpoint_hub, "configure_huggingface_http", lambda: None)
    monkeypatch.setattr(checkpoint_hub, "import_module", lambda _name: module)

    with pytest.raises(HubCheckpointError, match=rf"\A{message}\Z"):
        checkpoint_hub._default_manifest_loader("owner/model", "manifest.json")


def test_default_manifest_loader_wraps_download_failures(monkeypatch) -> None:
    monkeypatch.setattr(checkpoint_hub, "configure_huggingface_http", lambda: None)

    def download(**_kwargs: object) -> str:
        raise RuntimeError("network down")

    monkeypatch.setattr(
        checkpoint_hub,
        "import_module",
        lambda _name: SimpleNamespace(hf_hub_download=download),
    )

    with pytest.raises(
        HubCheckpointError,
        match=r"\Acheckpoint manifest download failed\Z",
    ):
        checkpoint_hub._default_manifest_loader("owner/model", "manifest.json")


def test_canonical_json_is_compact_sorted_unicode_and_rejects_invalid_values() -> None:
    assert _canonical_json({"z": "é", "a": [2, 1]}) == '{"a":[2,1],"z":"é"}'

    with pytest.raises(
        HubCheckpointError,
        match=r"\Acheckpoint identity is not JSON-compatible\Z",
    ):
        _canonical_json({"value": float("nan")})

    with pytest.raises(
        HubCheckpointError,
        match=r"\Acheckpoint identity is not JSON-compatible\Z",
    ):
        _canonical_json(object())


def test_default_api_configures_http_and_constructs_hub_api(monkeypatch) -> None:
    calls: list[str] = []
    api = object()
    monkeypatch.setattr(
        checkpoint_hub,
        "configure_huggingface_http",
        lambda: calls.append("configured"),
    )
    monkeypatch.setattr(
        checkpoint_hub,
        "import_module",
        lambda name: SimpleNamespace(HfApi=lambda: (name, api)),
    )

    assert _default_api() == ("huggingface_hub", api)
    assert calls == ["configured"]


def test_default_api_wraps_configuration_failures(monkeypatch) -> None:
    monkeypatch.setattr(
        checkpoint_hub,
        "configure_huggingface_http",
        lambda: (_ for _ in ()).throw(RuntimeError("offline")),
    )

    with pytest.raises(
        HubCheckpointError,
        match=r"\AHugging Face checkpoint access is unavailable\Z",
    ):
        _default_api()


def test_group_step_files_filters_roots_and_malformed_paths() -> None:
    root = "runs/example/checkpoints"
    paths = (
        f"{root}/step-2/model.safetensors",
        f"{root}/step-2/trainer_state.json",
        f"{root}/step-1/optimizer.pt",
        "other/step-3/model.safetensors",
        f"{root}/step-5/optimizer.pt",
        f"{root}/step-0/model.safetensors",
        f"{root}/step-nope/model.safetensors",
        f"{root}/step-2/too/deep.safetensors",
    )
    assert _group_step_files(paths, root=root) == {
        1: ["optimizer.pt"],
        2: ["model.safetensors", "trainer_state.json"],
        5: ["optimizer.pt"],
    }


def test_complete_step_files_keeps_only_complete_steps() -> None:
    root = "runs/example/checkpoints"
    required = (
        "checkpoint-manifest.json",
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    )
    complete = tuple(
        f"{root}/step-3/{name}" for name in (*required, "model.safetensors")
    )
    incomplete = tuple(f"{root}/step-4/{name}" for name in required[:-1])
    assert _complete_step_files((*complete, *incomplete), root=root) == {
        3: tuple(sorted((*required, "model.safetensors")))
    }
    assert _is_complete_step([*required, "model-00001-of-00002.safetensors"])
    assert not _is_complete_step([*required])
    assert not _is_complete_step([*required[1:], "model.safetensors"])


def test_manifest_matches_requires_schema_step_and_identity(tmp_path: Path) -> None:
    identity = {"run_id": "a" * 20}
    manifest = tmp_path / "checkpoint-manifest.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "global_step": 3, "identity": identity}),
        encoding="utf-8",
    )
    assert _manifest_matches(manifest, identity=identity, step=3)

    for payload in (
        {"schema_version": 2, "global_step": 3, "identity": identity},
        {"schema_version": 1, "global_step": 4, "identity": identity},
        {"schema_version": 1, "global_step": 3, "identity": {"run_id": "b" * 20}},
    ):
        manifest.write_text(json.dumps(payload), encoding="utf-8")
        assert not _manifest_matches(manifest, identity=identity, step=3)

    manifest.write_text("not-json", encoding="utf-8")
    assert not _manifest_matches(manifest, identity=identity, step=3)
    assert not _manifest_matches(tmp_path / "missing.json", identity=identity, step=3)

    encodings: list[str | None] = []

    class _Manifest:
        def read_text(self, *, encoding: str | None) -> str:
            encodings.append(encoding)
            return json.dumps(
                {"schema_version": 1, "global_step": 3, "identity": identity}
            )

    assert _manifest_matches(cast(Any, _Manifest()), identity=identity, step=3)
    assert encodings == ["utf-8"]


@pytest.mark.parametrize("repository_id", [None, "", "  ", 42])
def test_validate_repository_id_rejects_empty_or_non_string_values(
    repository_id: object,
) -> None:
    with pytest.raises(
        HubCheckpointError,
        match=r"\Amodel repository ID is invalid\Z",
    ):
        _validate_repository_id(repository_id)  # ty: ignore[invalid-argument-type]


def test_published_paths_filters_invalid_entries_and_wraps_api_errors() -> None:
    calls: list[dict[str, object]] = []

    class _Api:
        def list_repo_tree(self, **kwargs: object) -> tuple[object, ...]:
            calls.append(kwargs)
            return (_Entry("one"), SimpleNamespace(path=42))

    assert _published_paths(_Api(), "owner/model", "root") == ("one",)
    assert calls == [
        {
            "repo_id": "owner/model",
            "repo_type": "model",
            "path_in_repo": "root",
            "recursive": True,
        }
    ]

    class _FailingApi:
        def list_repo_tree(self, **_kwargs: object) -> tuple[object, ...]:
            raise RuntimeError("network")

    with pytest.raises(
        HubCheckpointError,
        match=r"\Apublished checkpoint inventory could not be read\Z",
    ):
        _published_paths(_FailingApi(), "owner/model", "root")


def test_matching_published_checkpoint_skips_bad_manifests_and_uses_newest_match() -> (
    None
):
    identity = {"run_id": "a" * 20}
    calls: list[str] = []

    def load_manifest(repository_id: str, path: str) -> Path:
        assert repository_id == "owner/model"
        calls.append(path)
        if path.endswith("step-20/checkpoint-manifest.json"):
            raise HubCheckpointError("temporarily unavailable")
        manifest = Path("/tmp/checkpoint-hub-test-manifest.json")
        manifest.write_text(
            json.dumps({"schema_version": 1, "global_step": 10, "identity": identity}),
            encoding="utf-8",
        )
        return manifest

    result = _matching_published_checkpoint(
        "owner/model",
        prefix="runs/example",
        root="runs/example/checkpoints",
        complete_steps={10: ("step-10/file",), 20: ("step-20/file",)},
        load_manifest=load_manifest,
        identity=identity,
    )
    assert result == PublishedCheckpoint(
        repository_id="owner/model",
        prefix="runs/example",
        step=10,
        files=("step-10/file",),
    )
    assert calls == [
        "runs/example/checkpoints/step-20/checkpoint-manifest.json",
        "runs/example/checkpoints/step-10/checkpoint-manifest.json",
    ]


def test_matching_published_checkpoint_continues_after_a_nonmatching_manifest(
    tmp_path: Path,
) -> None:
    identity = {"run_id": "a" * 20}

    def load_manifest(_repository_id: str, path: str) -> Path:
        step = int(path.split("step-")[1].split("/")[0])
        manifest = tmp_path / f"manifest-{step}.json"
        manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "global_step": step,
                    "identity": identity if step == 10 else {"run_id": "b" * 20},
                }
            ),
            encoding="utf-8",
        )
        return manifest

    result = _matching_published_checkpoint(
        "owner/model",
        prefix="runs/example",
        root="runs/example/checkpoints",
        complete_steps={10: ("step-10/file",), 20: ("step-20/file",)},
        load_manifest=load_manifest,
        identity=identity,
    )

    assert result is not None
    assert result.step == 10


def test_validate_restore_output_requires_absolute_non_symlink_path(
    tmp_path: Path,
) -> None:
    _validate_restore_output(tmp_path / "models")
    relative = Path("models")
    with pytest.raises(HubCheckpointError, match="output directory is unsafe"):
        _validate_restore_output(relative)

    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(
        HubCheckpointError,
        match=r"\Acheckpoint output directory is unsafe\Z",
    ):
        _validate_restore_output(link)


def test_reject_symlinks_detects_root_and_nested_links(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    _reject_symlinks(root)
    (root / "target").write_text("data", encoding="utf-8")
    (root / "link").symlink_to(root / "target")
    with pytest.raises(
        HubCheckpointError,
        match=r"\Adownloaded checkpoint contains a symlink\Z",
    ):
        _reject_symlinks(root)

    root_link = tmp_path / "root-link"
    root_link.symlink_to(root, target_is_directory=True)
    with pytest.raises(
        HubCheckpointError,
        match=r"\Adownloaded checkpoint contains a symlink\Z",
    ):
        _reject_symlinks(root_link)


def test_downloaded_checkpoint_source_rejects_paths_outside_temporary_directory(
    tmp_path: Path,
) -> None:
    checkpoint = PublishedCheckpoint("owner/model", "runs/example", 3, ())
    safe = tmp_path / "runs/example/checkpoints/step-3"
    safe.mkdir(parents=True)
    assert (
        _downloaded_checkpoint_source(str(tmp_path), checkpoint, str(tmp_path)) == safe
    )

    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(
        HubCheckpointError,
        match=r"\Acheckpoint download path is unsafe\Z",
    ):
        _downloaded_checkpoint_source(str(outside), checkpoint, str(tmp_path))


def test_copy_and_validate_checkpoint_copies_once_and_requires_validation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "model.safetensors").write_bytes(b"model")
    output = tmp_path / "output"
    identity = {"run_id": "a" * 20}
    expected = CheckpointInfo(output / "checkpoint-3", 3)
    monkeypatch.setattr(
        checkpoint_hub, "find_complete_checkpoint", lambda *_args, **_kwargs: expected
    )

    assert (
        _copy_and_validate_checkpoint(source, output, 3, identity=identity) == expected
    )
    assert (output / "checkpoint-3/model.safetensors").read_bytes() == b"model"

    with pytest.raises(
        HubCheckpointError,
        match=r"\Acheckpoint destination already exists\Z",
    ):
        _copy_and_validate_checkpoint(source, output, 3, identity=identity)

    other_output = tmp_path / "other-output"
    monkeypatch.setattr(
        checkpoint_hub, "find_complete_checkpoint", lambda *_args, **_kwargs: None
    )
    with pytest.raises(
        HubCheckpointError,
        match=r"\Adownloaded checkpoint failed validation\Z",
    ):
        _copy_and_validate_checkpoint(source, other_output, 3, identity=identity)


@pytest.mark.parametrize(
    ("module", "message"),
    [
        (
            SimpleNamespace(),
            "Hugging Face checkpoint download is unavailable",
        ),
        (
            SimpleNamespace(snapshot_download=lambda **_kwargs: object()),
            "checkpoint download returned an invalid path",
        ),
    ],
)
def test_restore_checkpoint_rejects_invalid_snapshot_downloads(
    tmp_path: Path,
    monkeypatch,
    module: object,
    message: str,
) -> None:
    checkpoint = PublishedCheckpoint("owner/model", "runs/example", 3, ())
    module_names: list[str] = []
    monkeypatch.setattr(checkpoint_hub, "configure_huggingface_http", lambda: None)
    monkeypatch.setattr(
        checkpoint_hub,
        "import_module",
        lambda name: (module_names.append(name) or module),
    )

    with pytest.raises(HubCheckpointError, match=rf"\A{message}\Z"):
        _restore_checkpoint(
            tmp_path / "output",
            checkpoint,
            identity={"run_id": "a" * 20},
        )
    assert module_names == ["huggingface_hub"]


def test_restore_published_checkpoint_requires_a_matching_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[tuple[object, dict[str, object]]] = []

    def no_checkpoint(identity: object, **kwargs: object) -> None:
        calls.append((identity, kwargs))
        return None

    monkeypatch.setattr(checkpoint_hub, "latest_published_checkpoint", no_checkpoint)
    identity = {"run_id": "a" * 20}
    repository_id = "owner/model"
    with pytest.raises(
        HubCheckpointError,
        match=r"\Ano complete published checkpoint matches the run\Z",
    ):
        restore_published_checkpoint(
            tmp_path / "output",
            identity=identity,
            repository_id=repository_id,
        )
    assert calls == [(identity, {"repository_id": repository_id})]


def test_latest_published_checkpoint_requires_complete_identity_bound_files(
    tmp_path: Path,
) -> None:
    identity = {
        "run_id": "a" * 20,
        "task_name": "place-relevance-v2",
        "training_config": {
            "artifact_namespace": "studies/place-relevance-v2/baseline"
        },
    }
    prefix = "studies/place-relevance-v2/baseline/run-" + "a" * 20
    root = f"{prefix}/checkpoints"

    def paths(step: int, *, complete: bool) -> tuple[str, ...]:
        names = {
            "checkpoint-manifest.json",
            "trainer_state.json",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
            "model.safetensors",
        }
        if not complete:
            names.remove("optimizer.pt")
        return tuple(f"{root}/step-{step}/{name}" for name in names)

    entries = tuple(
        _Entry(path) for path in (*paths(10, complete=True), *paths(20, complete=False))
    )
    manifest = tmp_path / "checkpoint-manifest.json"
    manifest.write_text(
        json.dumps({"schema_version": 1, "global_step": 10, "identity": identity}),
        encoding="utf-8",
    )

    class _Api:
        def list_repo_tree(self, **kwargs: object) -> tuple[_Entry, ...]:
            assert kwargs == {
                "repo_id": "NoeFlandre/osm-polygon-sentence-classifier",
                "repo_type": "model",
                "path_in_repo": root,
                "recursive": True,
            }
            return entries

    manifest_calls: list[tuple[str, str]] = []

    def load_manifest(repository_id: str, path: str) -> Path:
        manifest_calls.append((repository_id, path))
        return manifest

    checkpoint = latest_published_checkpoint(
        identity,
        repository_id="NoeFlandre/osm-polygon-sentence-classifier",
        hub_api=_Api(),
        manifest_loader=load_manifest,
    )

    assert checkpoint == PublishedCheckpoint(
        repository_id="NoeFlandre/osm-polygon-sentence-classifier",
        prefix=prefix,
        step=10,
        files=tuple(
            sorted(path.rsplit("/", 1)[-1] for path in paths(10, complete=True))
        ),
    )
    assert manifest_calls == [
        (
            "NoeFlandre/osm-polygon-sentence-classifier",
            f"{root}/step-10/checkpoint-manifest.json",
        )
    ]


def test_restore_published_checkpoint_downloads_and_validates_the_latest_step(
    tmp_path: Path,
    monkeypatch,
) -> None:
    identity = {
        "run_id": "a" * 20,
        "task_name": "place-relevance-v2",
        "training_config": {
            "artifact_namespace": "studies/place-relevance-v2/baseline"
        },
    }
    checkpoint = PublishedCheckpoint(
        repository_id="NoeFlandre/osm-polygon-sentence-classifier",
        prefix="studies/place-relevance-v2/baseline/run-" + "a" * 20,
        step=12,
        files=(),
    )
    monkeypatch.setattr(
        checkpoint_hub,
        "latest_published_checkpoint",
        lambda *_args, **_kwargs: checkpoint,
    )
    captured: dict[str, object] = {}
    module_names: list[str] = []
    temporary_location: Path | None = None

    def fake_snapshot_download(**kwargs: object) -> str:
        nonlocal temporary_location
        captured.update(kwargs)
        local_dir = Path(str(kwargs["local_dir"]))
        temporary_location = local_dir
        source = (
            local_dir / checkpoint.prefix / "checkpoints" / f"step-{checkpoint.step}"
        )
        source.mkdir(parents=True)
        for filename in (
            "model.safetensors",
            "optimizer.pt",
            "scheduler.pt",
            "rng_state.pth",
        ):
            (source / filename).write_bytes(b"checkpoint")
        (source / "trainer_state.json").write_text(
            json.dumps({"global_step": checkpoint.step}), encoding="utf-8"
        )
        (source / "checkpoint-manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "global_step": checkpoint.step,
                    "identity": identity,
                }
            ),
            encoding="utf-8",
        )
        return str(local_dir)

    monkeypatch.setattr(
        checkpoint_hub,
        "import_module",
        lambda name: (
            module_names.append(name)
            or SimpleNamespace(snapshot_download=fake_snapshot_download)
        ),
    )

    output = tmp_path / "nested" / "models"
    restored = restore_published_checkpoint(
        output,
        identity=identity,
        repository_id=checkpoint.repository_id,
    )

    assert restored.global_step == 12
    assert restored.path == output / "checkpoint-12"
    assert output.stat().st_mode & 0o777 == 0o700
    assert module_names == ["huggingface_hub"]
    assert temporary_location is not None
    assert temporary_location.name.startswith(".hub-checkpoint-")
    assert temporary_location.parent == output.parent
    assert captured["repo_id"] == checkpoint.repository_id
    assert captured["repo_type"] == "model"
    assert captured["allow_patterns"] == [f"{checkpoint.prefix}/checkpoints/step-12/*"]

    with pytest.raises(
        HubCheckpointError,
        match=r"\Acheckpoint destination already exists\Z",
    ):
        restore_published_checkpoint(
            output,
            identity=identity,
            repository_id=checkpoint.repository_id,
        )


def test_restore_checkpoint_wraps_unexpected_failures(
    tmp_path: Path,
    monkeypatch,
) -> None:
    checkpoint = PublishedCheckpoint("owner/model", "runs/example", 3, ())
    monkeypatch.setattr(checkpoint_hub, "configure_huggingface_http", lambda: None)
    monkeypatch.setattr(
        checkpoint_hub,
        "import_module",
        lambda _name: SimpleNamespace(
            snapshot_download=lambda **_kwargs: (_ for _ in ()).throw(
                RuntimeError("network down")
            )
        ),
    )

    with pytest.raises(
        HubCheckpointError,
        match=r"\Apublished checkpoint restoration failed\Z",
    ):
        _restore_checkpoint(
            tmp_path / "output",
            checkpoint,
            identity={"run_id": "a" * 20},
        )
