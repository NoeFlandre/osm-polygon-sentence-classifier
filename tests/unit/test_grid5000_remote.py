import json
import os
import shlex
import subprocess
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from osm_polygon_sentence_classifier import grid5000_remote
from osm_polygon_sentence_classifier.grid5000 import (
    CommandResult,
    Grid5000ExecutionError,
)
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
    def __init__(
        self,
        prepared_stdout: str = "REMOTE_PREPARED reused=false\n",
    ) -> None:
        self.calls: list[tuple[tuple[str, ...], str | None]] = []
        self.prepared_stdout = prepared_stdout

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
            return CommandResult(returncode=0, stdout=self.prepared_stdout)
        if "HF_AUTH_INSTALLED" in remote_command:
            return CommandResult(returncode=0, stdout="HF_AUTH_INSTALLED\n")
        if "REMOTE_CLEANUP_STARTED" in remote_command:
            return CommandResult(returncode=0, stdout="REMOTE_CLEANUP_STARTED\n")
        if "checkpoint-manifest.json" in remote_command:
            return CommandResult(returncode=0, stdout="CHECKPOINT_READY\n")
        return CommandResult(returncode=0, stdout='{"run_id":"a"}\n')


class _CompletionRemoteRunner:
    def __init__(self, stdout: str) -> None:
        self.stdout = stdout

    def __call__(
        self,
        argv: Sequence[str],
        *,
        timeout: float,
        input_text: str | None = None,
    ) -> CommandResult:
        del argv, timeout, input_text
        return CommandResult(returncode=0, stdout=self.stdout)


class _ResultRemoteRunner:
    def __init__(self, result: CommandResult) -> None:
        self.result = result
        self.calls: list[tuple[tuple[str, ...], float, str | None]] = []

    def __call__(
        self,
        argv: Sequence[str],
        *,
        timeout: float,
        input_text: str | None = None,
    ) -> CommandResult:
        self.calls.append((tuple(argv), timeout, input_text))
        return self.result


def test_subprocess_remote_runner_forwards_the_fixed_process_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def run(argv: tuple[str, ...], **kwargs: object) -> SimpleNamespace:
        calls.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout="out", stderr="err")

    monkeypatch.setattr(grid5000_remote.subprocess, "run", run)

    result = grid5000_remote.SubprocessRemoteRunner()(
        ["ssh", "nancy", "printf ok"],
        timeout=4.5,
        input_text="secret",
    )

    assert result == CommandResult(returncode=0, stdout="out", stderr="err")
    assert calls == [
        (
            ("ssh", "nancy", "printf ok"),
            {
                "capture_output": True,
                "check": False,
                "input": "secret",
                "text": True,
                "timeout": 4.5,
            },
        )
    ]


@pytest.mark.parametrize(
    "cause",
    [OSError("ssh unavailable"), subprocess.TimeoutExpired("ssh", 4.5)],
)
def test_subprocess_remote_runner_wraps_process_failures(
    monkeypatch: pytest.MonkeyPatch,
    cause: BaseException,
) -> None:
    def run(*_args: object, **_kwargs: object) -> None:
        raise cause

    monkeypatch.setattr(grid5000_remote.subprocess, "run", run)

    with pytest.raises(Grid5000ExecutionError) as error:
        grid5000_remote.SubprocessRemoteRunner()(["ssh", "nancy", "false"], timeout=4.5)

    assert str(error.value) == "Grid'5000 SSH command could not complete"
    assert error.value.__cause__ is cause


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
    assert "REMOTE_CLEANUP_STARTED" in command
    assert "nohup rm -rf" in command
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


def test_read_completion_accepts_an_identity_bound_manifest() -> None:
    remote = Grid5000Remote(
        "lille",
        runner=_CompletionRemoteRunner(
            json.dumps({"run_id": "a" * 20, "metrics": {"accuracy": 0.8}})
        ),
    )

    assert remote.read_completion("a" * 20) == {
        "run_id": "a" * 20,
        "metrics": {"accuracy": 0.8},
    }


@pytest.mark.parametrize(
    "stdout",
    ["not-json", json.dumps({"run_id": "b" * 20})],
)
def test_read_completion_rejects_invalid_or_mismatched_manifests(stdout: str) -> None:
    remote = Grid5000Remote("lille", runner=_CompletionRemoteRunner(stdout))

    with pytest.raises(Grid5000ExecutionError, match="completion manifest"):
        remote.read_completion("a" * 20)


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


def test_remote_validation_errors_are_stable_and_exact() -> None:
    with pytest.raises(
        ValueError,
        match="run_id must be twenty lowercase hexadecimal characters",
    ) as run_id_error:
        grid5000_remote._validate_run_id("not-a-run-id")
    with pytest.raises(
        ValueError,
        match="source_commit must be forty lowercase hexadecimal characters",
    ) as revision_error:
        grid5000_remote._validate_revision("not-a-source-commit")

    assert str(run_id_error.value) == (
        "run_id must be twenty lowercase hexadecimal characters"
    )
    assert str(revision_error.value) == (
        "source_commit must be forty lowercase hexadecimal characters"
    )


def test_checkpoint_identity_json_rejects_non_finite_values_with_exact_error() -> None:
    with pytest.raises(
        ValueError, match="checkpoint identity must be JSON-compatible"
    ) as error:
        grid5000_remote._checkpoint_identity_json({"score": float("nan")})

    assert str(error.value) == "checkpoint identity must be JSON-compatible"
    assert isinstance(error.value.__cause__, ValueError)


def test_checkpoint_identity_json_explicitly_disables_non_finite_json_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, dict[str, object]]] = []

    def dumps(value: object, **kwargs: object) -> str:
        calls.append((value, kwargs))
        return "{}"

    monkeypatch.setattr(grid5000_remote.json, "dumps", dumps)

    result = grid5000_remote._checkpoint_identity_json({"run_id": "a" * 20})

    assert result == "{}"
    assert calls == [
        (
            {"run_id": "a" * 20},
            {"allow_nan": False, "sort_keys": True},
        )
    ]


@pytest.mark.parametrize("value", ["/absolute/path", "", ".", "safe/.."])
def test_safe_output_relative_path_rejects_unsafe_paths_exactly(value: str) -> None:
    with pytest.raises(
        ValueError, match="output_subdirectory must be a safe relative path"
    ) as error:
        grid5000_remote._safe_output_relative_path(value)

    assert str(error.value) == "output_subdirectory must be a safe relative path"


@pytest.mark.parametrize("parts", [("",), (".",)])
def test_safe_output_relative_path_rejects_un_normalized_dot_parts(
    monkeypatch: pytest.MonkeyPatch,
    parts: tuple[str, ...],
) -> None:
    monkeypatch.setattr(
        grid5000_remote,
        "PurePosixPath",
        lambda _value: SimpleNamespace(
            is_absolute=lambda: False,
            parts=parts,
        ),
    )

    with pytest.raises(
        ValueError, match="output_subdirectory must be a safe relative path"
    ) as error:
        grid5000_remote._safe_output_relative_path("ignored")

    assert str(error.value) == "output_subdirectory must be a safe relative path"


def test_checkpoint_probe_command_quotes_the_run_marker_and_shell_boolean() -> None:
    run_id = "a" * 20
    relative = grid5000_remote.PurePosixPath("models/landuse")
    marker = shlex.quote(f'"run_id":"{run_id}"')

    command = grid5000_remote._checkpoint_probe_command(
        run_id,
        relative,
        '{"run_id": "a"}',
        False,
    )

    assert f'grep -Fq {marker} "$marker"' in command
    assert "if false; then" in command

    failed_status_command = grid5000_remote._checkpoint_probe_command(
        run_id,
        relative,
        '{"run_id": "a"}',
        True,
    )
    assert "if true; then" in failed_status_command


def test_parse_completion_payload_reports_exact_json_error() -> None:
    with pytest.raises(Grid5000ExecutionError) as json_error:
        grid5000_remote._parse_completion_payload("not-json", "a" * 20)
    with pytest.raises(Grid5000ExecutionError) as identity_error:
        grid5000_remote._parse_completion_payload(
            json.dumps({"run_id": "b" * 20}), "a" * 20
        )

    assert str(json_error.value) == "remote completion manifest is not valid JSON"
    assert isinstance(json_error.value.__cause__, json.JSONDecodeError)
    assert str(identity_error.value) == "remote completion manifest identity is invalid"


def test_remote_constructor_accepts_one_second_and_preserves_runner_and_timeout() -> (
    None
):
    runner = _ResultRemoteRunner(CommandResult(returncode=0, stdout="ok"))

    remote = Grid5000Remote("nancy", runner=runner, command_timeout=1)

    assert remote.site == "nancy"
    assert remote.runner is runner
    assert remote.command_timeout == 1

    with pytest.raises(ValueError, match="command_timeout must be positive") as error:
        Grid5000Remote("nancy", runner=runner, command_timeout=0)
    assert str(error.value) == "command_timeout must be positive"


@pytest.mark.parametrize(
    ("stderr", "expected"),
    [
        ("disk full\n", "remote command failed with exit code 17: disk full"),
        ("", "remote command failed with exit code 17"),
    ],
)
def test_remote_run_reports_exact_nonzero_command_errors(
    stderr: str,
    expected: str,
) -> None:
    runner = _ResultRemoteRunner(CommandResult(returncode=17, stdout="", stderr=stderr))
    remote = Grid5000Remote("nancy", runner=runner)

    with pytest.raises(Grid5000ExecutionError) as error:
        remote.run("false")

    assert str(error.value) == expected


def test_remote_raw_forwards_timeout_input_and_returns_raw_result() -> None:
    result = CommandResult(returncode=7, stdout="out", stderr="err")
    runner = _ResultRemoteRunner(result)
    remote = Grid5000Remote("nancy", runner=runner, command_timeout=3.5)

    assert remote.raw("printf ok", input_text="secret") is result

    assert runner.calls[0][1:] == (3.5, "secret")
    assert runner.calls[0][0][-1] == "printf ok"


def test_prepare_builds_the_exact_identity_bound_command_and_result() -> None:
    run_id = "a" * 20
    source_commit = "b" * 40
    runner = _RecordingRemoteRunner()
    remote = Grid5000Remote("grenoble", runner=runner)

    result = remote.prepare(run_id=run_id, source_commit=source_commit)

    marker = json.dumps(
        {"schema_version": 1, "run_id": run_id, "status": "active"},
        sort_keys=True,
        separators=(",", ":"),
    )
    command = runner.calls[0][0][-1]
    run_marker = shlex.quote(f'"run_id":"{run_id}"')
    assert f'repo="$HOME/{REMOTE_CHECKOUT_SUBDIRECTORY}"' in command
    assert f'data_root="$HOME/{REMOTE_DATA_SUBDIRECTORY}"' in command
    assert (
        f'run_root="$HOME/{REMOTE_DATA_SUBDIRECTORY}/grid5000/runs/{run_id}"' in command
    )
    assert result.site == "grenoble"
    assert result.run_id == run_id
    assert result.reused_checkout is False
    assert shlex.quote(marker) in command
    assert command.count(shlex.quote(marker)) == 2
    assert (
        command.count(f'grep -Fq {run_marker} "$run_root/.operator-managed.json"') == 1
    )
    assert (
        f"git clone --no-tags {shlex.quote(grid5000_remote.REMOTE_REPOSITORY_URL)}"
        in command
    )
    assert (
        f'git -C "$repo" cat-file -e {shlex.quote(f"{source_commit}^{{commit}}")}'
        in command
    )
    assert f'git -C "$repo" checkout --detach {shlex.quote(source_commit)}' in command
    assert '[ ! -L "$data_root/grid5000" ]' in command
    assert 'elif false && grep -Fq \'"status":"failed"\'' in command


def test_prepare_reports_reuse_and_failed_resume_boolean_exactly() -> None:
    run_id = "a" * 20
    source_commit = "b" * 40
    runner = _RecordingRemoteRunner("REMOTE_PREPARED reused=true\n")
    remote = Grid5000Remote("grenoble", runner=runner)

    result = remote.prepare(
        run_id=run_id,
        source_commit=source_commit,
        allow_failed_run=True,
    )

    assert result.reused_checkout is True
    assert 'elif true && grep -Fq \'"status":"failed"\'' in runner.calls[0][0][-1]


def test_prepare_rejects_invalid_boolean_and_missing_marker_exactly() -> None:
    remote = Grid5000Remote("grenoble", runner=_RecordingRemoteRunner())
    with pytest.raises(
        ValueError, match="allow_failed_run must be a boolean"
    ) as boolean_error:
        remote.prepare(
            run_id="a" * 20,
            source_commit="b" * 40,
            allow_failed_run=cast(Any, "yes"),
        )
    assert str(boolean_error.value) == "allow_failed_run must be a boolean"

    missing_marker_remote = Grid5000Remote(
        "grenoble", runner=_RecordingRemoteRunner("")
    )
    with pytest.raises(Grid5000ExecutionError) as marker_error:
        missing_marker_remote.prepare(
            run_id="a" * 20,
            source_commit="b" * 40,
        )
    assert str(marker_error.value) == "remote preparation marker is missing"


@pytest.mark.parametrize("token", [None, "", "   ", "a\nb"])
def test_install_hugging_face_token_rejects_invalid_values_exactly(
    token: object,
) -> None:
    remote = Grid5000Remote("nancy", runner=_RecordingRemoteRunner())

    with pytest.raises(
        ValueError,
        match="Hugging Face token must be a non-empty single-line value",
    ) as error:
        remote.install_hugging_face_token(cast(Any, token))

    assert str(error.value) == (
        "Hugging Face token must be a non-empty single-line value"
    )


def test_install_hugging_face_token_reports_a_missing_marker_exactly() -> None:
    remote = Grid5000Remote(
        "nancy",
        runner=_ResultRemoteRunner(CommandResult(returncode=0, stdout="")),
    )

    with pytest.raises(Grid5000ExecutionError) as error:
        remote.install_hugging_face_token("hf_secret")

    assert str(error.value) == "remote Hugging Face auth marker is missing"


def test_read_completion_uses_the_exact_completion_manifest_path() -> None:
    run_id = "a" * 20
    runner = _ResultRemoteRunner(
        CommandResult(returncode=0, stdout=json.dumps({"run_id": run_id}))
    )
    remote = Grid5000Remote("lille", runner=runner)

    remote.read_completion(run_id)

    path = f'"$HOME/{REMOTE_DATA_SUBDIRECTORY}/grid5000/runs/{run_id}/completion.json"'
    assert runner.calls[0][0][-1] == f"test -f {path} && cat {path}"


def test_checkpoint_probe_status_contract_and_errors_are_exact() -> None:
    run_id = "a" * 20
    missing_runner = _ResultRemoteRunner(
        CommandResult(returncode=0, stdout="CHECKPOINT_MISSING\n")
    )
    missing_remote = Grid5000Remote("lille", runner=missing_runner)
    assert (
        missing_remote.has_complete_checkpoint(
            run_id,
            output_subdirectory="models/landuse",
            identity=IDENTITY,
        )
        is False
    )
    assert "if false; then" in missing_runner.calls[0][0][-1]

    failed_status_runner = _ResultRemoteRunner(
        CommandResult(returncode=0, stdout="CHECKPOINT_MISSING\n")
    )
    failed_status_remote = Grid5000Remote("lille", runner=failed_status_runner)
    assert (
        failed_status_remote.has_complete_checkpoint(
            run_id,
            output_subdirectory="models/landuse",
            identity=IDENTITY,
            allow_failed_status=True,
        )
        is False
    )
    assert "if true; then" in failed_status_runner.calls[0][0][-1]

    invalid_status_remote = Grid5000Remote("lille", runner=_RecordingRemoteRunner())
    with pytest.raises(
        ValueError, match="allow_failed_status must be a boolean"
    ) as status_error:
        invalid_status_remote.has_complete_checkpoint(
            run_id,
            output_subdirectory="models/landuse",
            identity=IDENTITY,
            allow_failed_status=cast(Any, "yes"),
        )
    assert str(status_error.value) == "allow_failed_status must be a boolean"

    unknown_runner = _ResultRemoteRunner(
        CommandResult(returncode=0, stdout="UNKNOWN\n")
    )
    with pytest.raises(Grid5000ExecutionError) as marker_error:
        Grid5000Remote("lille", runner=unknown_runner).has_complete_checkpoint(
            run_id,
            output_subdirectory="models/landuse",
            identity=IDENTITY,
        )
    assert str(marker_error.value) == "remote checkpoint probe marker is missing"


def test_cleanup_uses_the_exact_marker_bound_root_and_reports_missing_marker() -> None:
    run_id = "a" * 20
    runner = _ResultRemoteRunner(
        CommandResult(returncode=0, stdout="REMOTE_CLEANUP_STARTED\n")
    )
    remote = Grid5000Remote("lille", runner=runner)

    remote.cleanup(run_id)

    command = runner.calls[0][0][-1]
    root = f'"$HOME/{REMOTE_DATA_SUBDIRECTORY}/grid5000/runs/{run_id}"'
    marker = shlex.quote(f'"run_id":"{run_id}"')
    assert f"root={root}" in command
    assert f'grep -Fq {marker} "$marker"' in command

    missing_marker_remote = Grid5000Remote(
        "lille",
        runner=_ResultRemoteRunner(CommandResult(returncode=0, stdout="unexpected")),
    )
    with pytest.raises(Grid5000ExecutionError) as error:
        missing_marker_remote.cleanup(run_id)
    assert str(error.value) == "remote cleanup marker is missing"


@pytest.mark.parametrize("status", ["active", "complete", "failed"])
def test_mark_status_writes_the_exact_atomic_identity_marker(status: str) -> None:
    run_id = "a" * 20
    runner = _ResultRemoteRunner(
        CommandResult(returncode=0, stdout="REMOTE_STATUS_MARKED\n")
    )
    remote = Grid5000Remote("lille", runner=runner)

    remote.mark_status(run_id, status)

    marker = json.dumps(
        {"schema_version": 1, "run_id": run_id, "status": status},
        sort_keys=True,
        separators=(",", ":"),
    )
    command = runner.calls[0][0][-1]
    root = f'"$HOME/{REMOTE_DATA_SUBDIRECTORY}/grid5000/runs/{run_id}"'
    assert f"root={root}" in command
    assert command.count(shlex.quote(marker)) == 1
    assert 'chmod 0600 "$temporary"' in command
    assert 'mv -f -- "$temporary" "$root/.operator-managed.json"' in command
    assert "REMOTE_STATUS_MARKED" in command


def test_mark_status_rejects_an_unknown_status_exactly() -> None:
    with pytest.raises(ValueError, match="remote run status is invalid") as error:
        Grid5000Remote("lille", runner=_RecordingRemoteRunner()).mark_status(
            "a" * 20, "queued"
        )

    assert str(error.value) == "remote run status is invalid"
