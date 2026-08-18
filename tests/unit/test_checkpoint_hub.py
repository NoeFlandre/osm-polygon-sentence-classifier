import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import osm_polygon_sentence_classifier.checkpoint_hub as checkpoint_hub
from osm_polygon_sentence_classifier.checkpoint_hub import (
    PublishedCheckpoint,
    latest_published_checkpoint,
    restore_published_checkpoint,
)


@dataclass(frozen=True)
class _Entry:
    path: str


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
            assert kwargs["path_in_repo"] == root
            return entries

    checkpoint = latest_published_checkpoint(
        identity,
        repository_id="NoeFlandre/osm-polygon-sentence-classifier",
        hub_api=_Api(),
        manifest_loader=lambda _repo, _path: manifest,
    )

    assert checkpoint is not None
    assert checkpoint.step == 10


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

    def fake_snapshot_download(**kwargs: object) -> str:
        captured.update(kwargs)
        local_dir = Path(str(kwargs["local_dir"]))
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
        lambda _name: SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    restored = restore_published_checkpoint(
        tmp_path / "models",
        identity=identity,
        repository_id=checkpoint.repository_id,
    )

    assert restored.global_step == 12
    assert restored.path == tmp_path / "models" / "checkpoint-12"
    assert captured["repo_id"] == checkpoint.repository_id
    assert captured["repo_type"] == "model"
    assert captured["allow_patterns"] == [f"{checkpoint.prefix}/checkpoints/step-12/*"]
