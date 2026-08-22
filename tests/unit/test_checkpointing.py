import json
from pathlib import Path
from typing import Any, cast

import pytest

import osm_polygon_sentence_classifier.checkpointing as checkpointing
from osm_polygon_sentence_classifier.checkpointing import (
    CHECKPOINT_MANIFEST_FILENAME,
    CheckpointError,
    find_latest_complete_checkpoint,
    write_checkpoint_manifest,
)

IDENTITY: dict[str, object] = {
    "run_id": "a" * 20,
    "source_commit": "b" * 40,
    "dataset_revision": "c" * 40,
    "model_name_or_path": "test-model",
    "model_revision": "d" * 40,
}


def _checkpoint(
    output: Path,
    step: int,
    *,
    identity: dict[str, object] = IDENTITY,
    complete: bool = True,
) -> Path:
    directory = output / f"checkpoint-{step}"
    directory.mkdir(parents=True)
    (directory / "model.safetensors").write_bytes(b"weights")
    (directory / "trainer_state.json").write_text(
        json.dumps({"global_step": step}), encoding="utf-8"
    )
    if complete:
        (directory / "optimizer.pt").write_bytes(b"optimizer")
        (directory / "scheduler.pt").write_bytes(b"scheduler")
        (directory / "rng_state.pth").write_bytes(b"rng")
        write_checkpoint_manifest(
            directory,
            identity=identity,
            global_step=step,
        )
    return directory


def test_latest_complete_checkpoint_ignores_partial_and_wrong_identity(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    _checkpoint(output, 100)
    _checkpoint(output, 200, complete=False)
    _checkpoint(output, 300, identity={**IDENTITY, "run_id": "e" * 20})

    result = find_latest_complete_checkpoint(output, identity=IDENTITY)

    assert result is not None
    assert result.path == output / "checkpoint-100"
    assert result.global_step == 100


def test_checkpoint_manifest_is_atomic_and_identity_bound(tmp_path: Path) -> None:
    checkpoint = _checkpoint(tmp_path, 42)

    manifest = checkpoint / CHECKPOINT_MANIFEST_FILENAME
    payload = json.loads(manifest.read_text(encoding="utf-8"))

    assert payload == {
        "global_step": 42,
        "identity": IDENTITY,
        "schema_version": 1,
    }
    assert manifest.stat().st_mode & 0o777 == 0o600


def test_canonical_json_is_compact_sorted_and_rejects_non_finite_values() -> None:
    assert checkpointing._canonical_json({"z": "é", "a": 1}) == ('{"a":1,"z":"é"}')

    with pytest.raises(
        CheckpointError,
        match=r"\Acheckpoint identity is not JSON-compatible\Z",
    ):
        checkpointing._canonical_json({"value": float("nan")})


def test_find_complete_checkpoint_rejects_a_relative_path_with_exact_error() -> None:
    with pytest.raises(
        CheckpointError,
        match=r"\Acheckpoint directory must not be symlinked\Z",
    ):
        checkpointing.find_complete_checkpoint(
            Path("relative/checkpoint-1"), identity=IDENTITY
        )


def test_contains_symlink_rejects_relative_paths_with_exact_error() -> None:
    with pytest.raises(
        CheckpointError,
        match=r"\Acheckpoint path must be absolute\Z",
    ):
        checkpointing._contains_symlink(Path("relative"))


def test_find_complete_checkpoint_rejects_a_symlinked_parent(tmp_path: Path) -> None:
    target = tmp_path / "target"
    output = target / "output"
    output.mkdir(parents=True)
    _checkpoint(output, 1)
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(
        CheckpointError,
        match=r"\Acheckpoint directory must not be symlinked\Z",
    ):
        checkpointing.find_complete_checkpoint(
            link / "output" / "checkpoint-1",
            identity=IDENTITY,
        )


def test_safe_checkpoint_directory_requires_a_real_directory(tmp_path: Path) -> None:
    regular_file = tmp_path / "checkpoint-1"
    regular_file.write_bytes(b"not a directory")

    assert checkpointing._is_safe_checkpoint_directory(regular_file) is False


@pytest.mark.parametrize(
    "missing_name",
    [
        CHECKPOINT_MANIFEST_FILENAME,
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    ],
)
def test_incomplete_checkpoint_missing_any_required_file_is_ignored(
    tmp_path: Path, missing_name: str
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    checkpoint = _checkpoint(output, 1)
    (checkpoint / missing_name).unlink()

    assert checkpointing._has_checkpoint_files(checkpoint) is False
    assert find_latest_complete_checkpoint(output, identity=IDENTITY) is None


def test_checkpoint_requires_a_regular_weight_file(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    checkpoint = _checkpoint(output, 1)
    (checkpoint / "model.safetensors").unlink()
    (checkpoint / "model.safetensors").mkdir()

    assert find_latest_complete_checkpoint(output, identity=IDENTITY) is None


def test_complete_checkpoint_has_all_required_files(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    checkpoint = _checkpoint(output, 1)

    assert checkpointing._has_checkpoint_files(checkpoint) is True


def test_checkpoint_file_contract_uses_exact_trainer_artifact_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    required_names = {
        CHECKPOINT_MANIFEST_FILENAME,
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    }

    class _FakePath:
        def __init__(self, name: str = "checkpoint-1") -> None:
            self.name = name

        def __truediv__(self, name: str) -> "_FakePath":
            return _FakePath(name)

        def iterdir(self) -> tuple["_FakePath", ...]:
            return (_FakePath("model.safetensors"),)

    for missing_name in required_names:
        monkeypatch.setattr(
            checkpointing,
            "_regular_file",
            lambda path, missing_name=missing_name: path.name != missing_name,
        )
        assert checkpointing._has_checkpoint_files(cast(Any, _FakePath())) is False


@pytest.mark.parametrize(
    ("manifest_update", "trainer_update"),
    [
        ({"schema_version": 2}, {}),
        ({"global_step": 2}, {}),
        ({}, {"global_step": 2}),
        ({"identity": {**IDENTITY, "run_id": "z" * 20}}, {}),
    ],
)
def test_checkpoint_metadata_must_match_the_expected_identity_and_step(
    tmp_path: Path,
    manifest_update: dict[str, object],
    trainer_update: dict[str, object],
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    checkpoint = _checkpoint(output, 1)

    manifest = checkpoint / CHECKPOINT_MANIFEST_FILENAME
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    manifest_payload.update(manifest_update)
    manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")

    trainer_state = checkpoint / "trainer_state.json"
    trainer_payload = json.loads(trainer_state.read_text(encoding="utf-8"))
    trainer_payload.update(trainer_update)
    trainer_state.write_text(json.dumps(trainer_payload), encoding="utf-8")

    assert find_latest_complete_checkpoint(output, identity=IDENTITY) is None


@pytest.mark.parametrize("payload", [[], "invalid"])
def test_checkpoint_metadata_must_be_mapping_objects(
    tmp_path: Path, payload: object
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    checkpoint = _checkpoint(output, 1)
    (checkpoint / CHECKPOINT_MANIFEST_FILENAME).write_text(
        json.dumps(payload), encoding="utf-8"
    )

    assert find_latest_complete_checkpoint(output, identity=IDENTITY) is None


def test_checkpoint_rejects_a_non_json_identity_value(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    checkpoint = _checkpoint(output, 1)
    manifest = checkpoint / CHECKPOINT_MANIFEST_FILENAME
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["identity"] = float("nan")
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    assert find_latest_complete_checkpoint(output, identity=IDENTITY) is None


def test_latest_checkpoint_selects_the_highest_complete_step(tmp_path: Path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    _checkpoint(output, 10)
    _checkpoint(output, 20)

    result = find_latest_complete_checkpoint(output, identity=IDENTITY)

    assert result is not None
    assert result.path == output / "checkpoint-20"
    assert result.global_step == 20


@pytest.mark.parametrize(
    ("directory_name", "global_step", "message"),
    [
        ("checkpoint-1", 0, "checkpoint global_step must be positive"),
        ("checkpoint-1", True, "checkpoint global_step must be positive"),
        ("checkpoint-1", 1.5, "checkpoint global_step must be positive"),
        (
            "checkpoint-1",
            2,
            "checkpoint directory name does not match global_step",
        ),
        (
            "not-a-checkpoint",
            1,
            "checkpoint directory name does not match global_step",
        ),
    ],
)
def test_write_manifest_rejects_invalid_step_contracts(
    tmp_path: Path,
    directory_name: str,
    global_step: object,
    message: str,
) -> None:
    checkpoint = tmp_path / directory_name
    checkpoint.mkdir()

    with pytest.raises(CheckpointError, match=rf"\A{message}\Z"):
        write_checkpoint_manifest(
            checkpoint,
            identity=IDENTITY,
            global_step=global_step,  # ty: ignore[invalid-argument-type]
        )


def test_write_manifest_accepts_the_first_positive_step(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()

    manifest = write_checkpoint_manifest(
        checkpoint,
        identity=IDENTITY,
        global_step=1,
    )

    assert manifest.is_file()


def test_write_manifest_requires_a_real_absolute_directory(tmp_path: Path) -> None:
    regular_file = tmp_path / "checkpoint-1"
    regular_file.write_bytes(b"not a directory")

    with pytest.raises(
        CheckpointError,
        match=r"\Acheckpoint directory must be a real absolute directory\Z",
    ):
        write_checkpoint_manifest(regular_file, identity=IDENTITY, global_step=1)


def test_write_manifest_rejects_a_symlinked_parent(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    checkpoint = link / "checkpoint-1"
    checkpoint.mkdir()

    with pytest.raises(
        CheckpointError,
        match=r"\Acheckpoint directory must be a real absolute directory\Z",
    ):
        write_checkpoint_manifest(checkpoint, identity=IDENTITY, global_step=1)


@pytest.mark.parametrize(
    "symlink_name",
    [CHECKPOINT_MANIFEST_FILENAME, ".checkpoint-manifest.json.tmp"],
)
def test_write_manifest_rejects_symlinked_manifest_paths(
    tmp_path: Path, symlink_name: str
) -> None:
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    target = tmp_path / "target"
    target.write_text("target", encoding="utf-8")
    (checkpoint / symlink_name).symlink_to(target)

    with pytest.raises(
        CheckpointError,
        match=r"\Acheckpoint manifest cannot be a symlink\Z",
    ):
        write_checkpoint_manifest(checkpoint, identity=IDENTITY, global_step=1)


def test_atomic_manifest_write_is_sorted_and_utf8_encoded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    temporary = tmp_path / "manifest.tmp"
    manifest = tmp_path / CHECKPOINT_MANIFEST_FILENAME
    encodings: list[object] = []
    original_write_text = cast(Any, Path.write_text)

    def record_write_text(
        path: Path, data: str, *args: object, **kwargs: object
    ) -> int:
        encodings.append(kwargs.get("encoding"))
        return original_write_text(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", record_write_text)
    checkpointing._write_manifest_atomically(
        temporary,
        manifest,
        {"z": "é", "a": 1},
    )

    assert encodings == ["utf-8"]
    assert manifest.read_text(encoding="utf-8") == '{"a": 1, "z": "\\u00e9"}\n'


def test_atomic_manifest_write_rejects_nan_with_exact_error(tmp_path: Path) -> None:
    with pytest.raises(
        CheckpointError,
        match=r"\Acheckpoint manifest cannot be written\Z",
    ):
        checkpointing._write_manifest_atomically(
            tmp_path / "manifest.tmp",
            tmp_path / CHECKPOINT_MANIFEST_FILENAME,
            {"value": float("nan")},
        )


def test_atomic_manifest_write_cleans_a_missing_temporary_file_after_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    temporary = tmp_path / "manifest.tmp"

    def fail_write_text(*args: object, **kwargs: object) -> int:
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", fail_write_text)

    with pytest.raises(
        CheckpointError,
        match=r"\Acheckpoint manifest cannot be written\Z",
    ):
        checkpointing._write_manifest_atomically(
            temporary,
            tmp_path / CHECKPOINT_MANIFEST_FILENAME,
            {"a": 1},
        )

    assert not temporary.exists()


def test_checkpoint_metadata_is_read_as_utf8(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    _checkpoint(output, 1)
    encodings: list[object] = []
    paths: list[Path] = []
    original_read_text = cast(Any, Path.read_text)

    def record_read_text(path: Path, *args: object, **kwargs: object) -> str:
        encodings.append(kwargs.get("encoding"))
        paths.append(path)
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", record_read_text)

    assert find_latest_complete_checkpoint(output, identity=IDENTITY) is not None
    assert encodings == ["utf-8", "utf-8"]
    assert [path.name for path in paths] == [
        CHECKPOINT_MANIFEST_FILENAME,
        "trainer_state.json",
    ]


def test_latest_checkpoint_rejects_a_file_output_with_exact_error(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    output.write_bytes(b"not a directory")

    with pytest.raises(
        CheckpointError,
        match=r"\Acheckpoint output path must be a directory\Z",
    ):
        find_latest_complete_checkpoint(output, identity=IDENTITY)


def test_latest_checkpoint_rejects_a_symlinked_output_with_exact_error(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    output = tmp_path / "output"
    output.symlink_to(target, target_is_directory=True)

    with pytest.raises(
        CheckpointError,
        match=r"\Acheckpoint output directory must not be symlinked\Z",
    ):
        find_latest_complete_checkpoint(output, identity=IDENTITY)


def test_checkpoint_candidates_preserves_the_read_error_message() -> None:
    class _UnreadableDirectory:
        def iterdir(self) -> object:
            raise OSError("permission denied")

    with pytest.raises(
        CheckpointError,
        match=r"\Acheckpoint output directory cannot be read\Z",
    ):
        checkpointing._checkpoint_candidates(cast(Any, _UnreadableDirectory()))


def test_checkpointing_rejects_a_symlinked_output(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "output"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(CheckpointError, match="symlink"):
        find_latest_complete_checkpoint(link, identity=IDENTITY)
