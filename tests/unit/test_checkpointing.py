import json
from pathlib import Path

import pytest

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


def test_checkpointing_rejects_a_symlinked_output(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "output"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(CheckpointError, match="symlink"):
        find_latest_complete_checkpoint(link, identity=IDENTITY)
