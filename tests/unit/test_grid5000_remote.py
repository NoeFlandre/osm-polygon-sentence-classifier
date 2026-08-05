from collections.abc import Sequence

from osm_polygon_sentence_classifier.grid5000 import CommandResult
from osm_polygon_sentence_classifier.grid5000_remote import (
    REMOTE_CHECKOUT_SUBDIRECTORY,
    REMOTE_DATA_SUBDIRECTORY,
    Grid5000Remote,
)


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
