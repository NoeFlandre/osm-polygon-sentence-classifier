import json
import os
import subprocess
from collections.abc import Sequence
from pathlib import Path

from osm_polygon_sentence_classifier.grid5000 import CommandResult
from osm_polygon_sentence_classifier.grid5000_remote import (
    REMOTE_CHECKOUT_SUBDIRECTORY,
    REMOTE_DATA_SUBDIRECTORY,
    Grid5000Remote,
)

IDENTITY: dict[str, object] = {
    "run_id": "a" * 20,
    "source_commit": "b" * 40,
    "dataset_revision": "c" * 40,
    "model_name_or_path": "test-model",
    "model_revision": "d" * 40,
}


class _RecordingRemoteRunner:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], str | None]] = []

    def __call__(
        self,
        argv: Sequence[str],
        *,
        timeout: float,
        input_text: str | None = None,
    ) -> CommandResult:
        del timeout
        command = tuple(argv)
        self.calls.append((command, input_text))
        remote_command = command[-1]
        if "REMOTE_PREPARED" in remote_command:
            return CommandResult(returncode=0, stdout="REMOTE_PREPARED reused=false\n")
        if "HF_AUTH_INSTALLED" in remote_command:
            return CommandResult(returncode=0, stdout="HF_AUTH_INSTALLED\n")
        if "REMOTE_CLEANED" in remote_command:
            return CommandResult(returncode=0, stdout="REMOTE_CLEANED\n")
        if "checkpoint-manifest.json" in remote_command:
            return CommandResult(returncode=0, stdout="CHECKPOINT_READY\n")
        return CommandResult(returncode=0, stdout='{"run_id":"a"}\n')


def test_prepare_stages_an_exact_clean_checkout_without_destructive_cleanup() -> None:
    runner = _RecordingRemoteRunner()
    remote = Grid5000Remote("grenoble", runner=runner)

    result = remote.prepare(
        run_id="a" * 20,
        source_commit="b" * 40,
    )

    assert result.run_id == "a" * 20
    assert len(runner.calls) == 1
    command = runner.calls[0][0][-1]
    assert REMOTE_CHECKOUT_SUBDIRECTORY in command
    assert REMOTE_DATA_SUBDIRECTORY in command
    assert "git clone --no-tags" in command
    assert "checkout --detach" in command
    assert "status --porcelain" in command
    assert "rm -rf" not in command
    assert '[ ! -L "$data_root" ]' in command
    assert '[ ! -L "$run_root" ]' in command


def test_install_hugging_face_token_uses_stdin_and_never_command_text() -> None:
    runner = _RecordingRemoteRunner()
    remote = Grid5000Remote("nancy", runner=runner)
    token = "hf_secret_token_value"

    remote.install_hugging_face_token(token)

    command, input_text = runner.calls[0]
    assert input_text == token
    assert token not in " ".join(command)
    assert "HF_AUTH_INSTALLED" in command[-1]
    assert "chmod 0600" in command[-1]


def test_cleanup_targets_only_the_managed_run_marker() -> None:
    runner = _RecordingRemoteRunner()
    remote = Grid5000Remote("lille", runner=runner)

    remote.cleanup("a" * 20)

    command = runner.calls[0][0][-1]
    assert ".operator-managed.json" in command
    assert '"status":"(complete|failed)"' in command
    assert "REMOTE_CLEANED" in command
    assert '[ ! -L "$marker" ]' in command


def test_checkpoint_probe_is_read_only_and_marker_bound() -> None:
    runner = _RecordingRemoteRunner()
    remote = Grid5000Remote("lille", runner=runner)

    ready = remote.has_complete_checkpoint(
        "a" * 20,
        output_subdirectory=Path("models/landuse"),
        identity=IDENTITY,
    )

    assert ready is True
    command = runner.calls[0][0][-1]
    assert "checkpoint-manifest.json" in command
    assert '"identity"' in command
    assert "rm -rf" not in command
    assert "a" * 20 in command


def test_checkpoint_probe_recognizes_sharded_transformers_weights(
    tmp_path: Path,
) -> None:
    run_id = "a" * 20
    run_root = tmp_path / REMOTE_DATA_SUBDIRECTORY / "grid5000" / "runs" / run_id
    output = run_root / "models" / "landuse"
    checkpoint = output / "checkpoint-100"
    checkpoint.mkdir(parents=True)
    (run_root / ".operator-managed.json").write_text(
        json.dumps(
            {"schema_version": 1, "run_id": run_id, "status": "active"},
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    for filename in (
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "trainer_state.json",
    ):
        (checkpoint / filename).write_bytes(b"checkpoint")
    (checkpoint / "checkpoint-manifest.json").write_text(
        json.dumps(
            {"global_step": 100, "identity": IDENTITY, "schema_version": 1},
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (checkpoint / "model-00001-of-00002.safetensors").write_bytes(b"weights")

    class LocalShellRunner:
        def __call__(
            self,
            argv: Sequence[str],
            *,
            timeout: float,
            input_text: str | None = None,
        ) -> CommandResult:
            result = subprocess.run(
                ("bash", "-c", argv[-1]),
                capture_output=True,
                check=False,
                env={**os.environ, "HOME": str(tmp_path)},
                input=input_text,
                text=True,
                timeout=timeout,
            )
            return CommandResult(
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )

    ready = Grid5000Remote("lille", runner=LocalShellRunner()).has_complete_checkpoint(
        run_id,
        output_subdirectory=Path("models/landuse"),
        identity=IDENTITY,
    )

    assert ready is True


def test_checkpoint_probe_can_read_a_failed_marker_for_explicit_resume() -> None:
    runner = _RecordingRemoteRunner()
    remote = Grid5000Remote("lille", runner=runner)

    ready = remote.has_complete_checkpoint(
        "a" * 20,
        output_subdirectory=Path("models/landuse"),
        identity=IDENTITY,
        allow_failed_status=True,
    )

    assert ready is True
    command = runner.calls[0][0][-1]
    assert 'grep -Eq \'"status":"(active|failed)"\'' in command


def test_prepare_reopens_a_failed_marker_only_when_explicitly_allowed() -> None:
    runner = _RecordingRemoteRunner()
    remote = Grid5000Remote("lille", runner=runner)

    result = remote.prepare(
        run_id="a" * 20,
        source_commit="b" * 40,
        allow_failed_run=True,
    )

    assert result.run_id == "a" * 20
    command = runner.calls[0][0][-1]
    assert 'grep -Fq \'"status":"failed"\'' in command
    assert "REMOTE_PREPARED" in command
