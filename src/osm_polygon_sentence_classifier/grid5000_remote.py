"""Secure remote checkout, credential, completion, and cleanup boundaries."""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

from .grid5000 import (
    COMMAND_TIMEOUT_SECONDS,
    CommandResult,
    Grid5000ExecutionError,
    _ssh_argv,
)

REMOTE_REPOSITORY_URL = (
    "https://github.com/NoeFlandre/osm-polygon-sentence-classifier.git"
)
REMOTE_CHECKOUT_SUBDIRECTORY = "osm-polygon-sentence-classifier"
REMOTE_DATA_SUBDIRECTORY = "osm-polygon-sentence-classifier-data"
REMOTE_RUNS_SUBDIRECTORY = "grid5000/runs"
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_RUN_ID_PATTERN = re.compile(r"[0-9a-f]{20}")


class RemoteRunner(Protocol):
    """Fixed-argv SSH runner with optional stdin for secrets."""

    def __call__(
        self,
        argv: Sequence[str],
        *,
        timeout: float,
        input_text: str | None = None,
    ) -> CommandResult: ...


class SubprocessRemoteRunner:
    """Run a constructed SSH command without a local shell."""

    def __call__(
        self,
        argv: Sequence[str],
        *,
        timeout: float,
        input_text: str | None = None,
    ) -> CommandResult:
        try:
            result = subprocess.run(
                tuple(argv),
                capture_output=True,
                check=False,
                input=input_text,
                text=True,
                timeout=timeout,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise Grid5000ExecutionError(
                "Grid'5000 SSH command could not complete"
            ) from error
        return CommandResult(
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )


@dataclass(frozen=True, slots=True)
class RemotePreparationResult:
    """Facts returned after staging one exact remote run root."""

    site: str
    run_id: str
    reused_checkout: bool
    checkout_subdirectory: str = REMOTE_CHECKOUT_SUBDIRECTORY


def _validate_run_id(run_id: str) -> None:
    if _RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError("run_id must be twenty lowercase hexadecimal characters")


def _validate_revision(revision: str) -> None:
    if _REVISION_PATTERN.fullmatch(revision) is None:
        raise ValueError("source_commit must be forty lowercase hexadecimal characters")


def _checkpoint_identity_json(identity: Mapping[str, object]) -> str:
    try:
        return json.dumps(
            dict(identity),
            allow_nan=False,
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("checkpoint identity must be JSON-compatible") from error


def _safe_output_relative_path(output_subdirectory: str | Path) -> PurePosixPath:
    relative = PurePosixPath(str(output_subdirectory))
    if relative.is_absolute():
        raise ValueError("output_subdirectory must be a safe relative path")
    if not relative.parts:
        raise ValueError("output_subdirectory must be a safe relative path")
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError("output_subdirectory must be a safe relative path")
    return relative


def _checkpoint_probe_command(
    run_id: str,
    relative: PurePosixPath,
    identity_json: str,
    allow_failed_status: bool,
) -> str:
    root = f'"$HOME/{REMOTE_DATA_SUBDIRECTORY}/{REMOTE_RUNS_SUBDIRECTORY}/{run_id}"'
    output = f'"$root/{"/".join(relative.parts)}"'
    run_marker = shlex.quote(f'"run_id":"{run_id}"')
    identity_marker = shlex.quote(f'"identity": {identity_json}')
    return f"""
set -euo pipefail
root={root}
output={output}
marker="$root/.operator-managed.json"
[ -d "$root" ] && [ ! -L "$root" ]
[ -f "$marker" ] && [ ! -L "$marker" ]
grep -Fq {run_marker} "$marker"
if {str(allow_failed_status).lower()}; then
  grep -Eq '"status":"(active|failed)"' "$marker"
else
  grep -Fq '"status":"active"' "$marker"
fi
if [ ! -d "$output" ] || [ -L "$output" ]; then
  printf 'CHECKPOINT_MISSING\n'
  exit 0
fi
ready=false
for checkpoint in "$output"/checkpoint-*; do
  [ -d "$checkpoint" ] && [ ! -L "$checkpoint" ] || continue
  [ -f "$checkpoint/checkpoint-manifest.json" ] && [ ! -L "$checkpoint/checkpoint-manifest.json" ] || continue
  grep -Fq {identity_marker} "$checkpoint/checkpoint-manifest.json" || continue
  [ -f "$checkpoint/trainer_state.json" ] && [ ! -L "$checkpoint/trainer_state.json" ] || continue
  [ -f "$checkpoint/optimizer.pt" ] && [ ! -L "$checkpoint/optimizer.pt" ] || continue
  [ -f "$checkpoint/scheduler.pt" ] && [ ! -L "$checkpoint/scheduler.pt" ] || continue
  [ -f "$checkpoint/rng_state.pth" ] && [ ! -L "$checkpoint/rng_state.pth" ] || continue
  sharded_weight=false
  for weight in \
    "$checkpoint"/model-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9].safetensors \
    "$checkpoint"/pytorch_model-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9].bin
  do
    if [ -f "$weight" ] && [ ! -L "$weight" ]; then
      sharded_weight=true
      break
    fi
  done
  if {{ [ -f "$checkpoint/model.safetensors" ] && [ ! -L "$checkpoint/model.safetensors" ]; }} \
    || {{ [ -f "$checkpoint/pytorch_model.bin" ] && [ ! -L "$checkpoint/pytorch_model.bin" ]; }} \
    || "$sharded_weight"; then
    ready=true
    break
  fi
done
if "$ready"; then
  printf 'CHECKPOINT_READY\n'
else
  printf 'CHECKPOINT_MISSING\n'
fi
""".strip()


def _parse_completion_payload(stdout: str, run_id: str) -> dict[str, object]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise Grid5000ExecutionError(
            "remote completion manifest is not valid JSON"
        ) from error
    if not isinstance(payload, Mapping) or payload.get("run_id") != run_id:
        raise Grid5000ExecutionError("remote completion manifest identity is invalid")
    return dict(payload)


class Grid5000Remote:
    """Operate on one frontend through bounded, auditable SSH commands."""

    def __init__(
        self,
        site: str,
        *,
        runner: RemoteRunner | None = None,
        command_timeout: float = COMMAND_TIMEOUT_SECONDS,
    ) -> None:
        if command_timeout <= 0:
            raise ValueError("command_timeout must be positive")
        self.site = site
        self.runner = runner or SubprocessRemoteRunner()
        self.command_timeout = command_timeout

    def run(
        self,
        remote_command: str,
        *,
        input_text: str | None = None,
    ) -> CommandResult:
        """Execute one fixed SSH command and return its raw result."""

        result = self.raw(remote_command, input_text=input_text)
        if result.returncode != 0:
            detail = result.stderr.strip()
            suffix = f": {detail}" if detail else ""
            raise Grid5000ExecutionError(
                f"remote command failed with exit code {result.returncode}{suffix}"
            )
        return result

    def raw(
        self,
        remote_command: str,
        *,
        input_text: str | None = None,
    ) -> CommandResult:
        """Execute one fixed SSH command without interpreting its exit code."""

        result = self.runner(
            _ssh_argv(self.site, remote_command),
            timeout=self.command_timeout,
            input_text=input_text,
        )
        return result

    def prepare(
        self,
        *,
        run_id: str,
        source_commit: str,
        allow_failed_run: bool = False,
    ) -> RemotePreparationResult:
        """Prepare a clean checkout and activate the managed run root.

        ``allow_failed_run`` is reserved for an explicit continuation after
        the caller has verified retained checkpoint evidence.
        """

        _validate_run_id(run_id)
        _validate_revision(source_commit)
        if not isinstance(allow_failed_run, bool):
            raise ValueError("allow_failed_run must be a boolean")
        repo = f'"$HOME/{REMOTE_CHECKOUT_SUBDIRECTORY}"'
        data_root = f'"$HOME/{REMOTE_DATA_SUBDIRECTORY}"'
        run_root = (
            f'"$HOME/{REMOTE_DATA_SUBDIRECTORY}/{REMOTE_RUNS_SUBDIRECTORY}/{run_id}"'
        )
        commit_object = shlex.quote(f"{source_commit}^{{commit}}")
        marker = json.dumps(
            {"schema_version": 1, "run_id": run_id, "status": "active"},
            sort_keys=True,
            separators=(",", ":"),
        )
        command = f"""
set -euo pipefail
umask 077
repo={repo}
data_root={data_root}
run_root={run_root}
[ ! -L "$repo" ]
[ ! -L "$data_root" ]
[ ! -L "$data_root/{REMOTE_RUNS_SUBDIRECTORY.split("/")[0]}" ]
[ ! -L "$data_root/{REMOTE_RUNS_SUBDIRECTORY}" ]
[ ! -L "$run_root" ]
reused=false
if [ ! -e "$repo" ]; then
  git clone --no-tags {shlex.quote(REMOTE_REPOSITORY_URL)} "$repo"
elif [ -d "$repo/.git" ]; then
  reused=true
else
  echo 'remote checkout path is not a git checkout' >&2
  exit 70
fi
test -z "$(git -C "$repo" status --porcelain)"
git -C "$repo" fetch --no-tags origin main
git -C "$repo" cat-file -e {commit_object}
git -C "$repo" checkout --detach {shlex.quote(source_commit)}
test -z "$(git -C "$repo" status --porcelain)"
uv_bin="$(command -v uv || true)"
[ -n "$uv_bin" ] || uv_bin="$HOME/.local/bin/uv"
test -x "$uv_bin"
mkdir -p -m 0700 "$data_root/{REMOTE_RUNS_SUBDIRECTORY}"
if [ -e "$run_root" ] && [ ! -d "$run_root" ]; then
  echo 'remote run root is not a directory' >&2
  exit 70
fi
mkdir -p -m 0700 "$run_root"
if [ -f "$run_root/.operator-managed.json" ]; then
  [ ! -L "$run_root/.operator-managed.json" ]
  grep -Fq {shlex.quote(f'"run_id":"{run_id}"')} "$run_root/.operator-managed.json"
  if grep -Fq '"status":"active"' "$run_root/.operator-managed.json"; then
    :
  elif {str(allow_failed_run).lower()} && grep -Fq '"status":"failed"' "$run_root/.operator-managed.json"; then
    temporary="$run_root/.operator-managed.json.tmp"
    printf '%s\n' {shlex.quote(marker)} >"$temporary"
    chmod 0600 "$temporary"
    mv -f -- "$temporary" "$run_root/.operator-managed.json"
  else
    echo 'remote run root is not active' >&2
    exit 70
  fi
else
  printf '%s\n' {shlex.quote(marker)} >"$run_root/.operator-managed.json"
  chmod 0600 "$run_root/.operator-managed.json"
fi
printf 'REMOTE_PREPARED reused=%s\n' "$reused"
""".strip()
        result = self.run(command)
        marker_text = "REMOTE_PREPARED reused="
        if marker_text not in result.stdout:
            raise Grid5000ExecutionError("remote preparation marker is missing")
        return RemotePreparationResult(
            site=self.site,
            run_id=run_id,
            reused_checkout="reused=true" in result.stdout,
        )

    def install_hugging_face_token(self, token: str) -> None:
        """Install a token through SSH stdin without putting it in argv or logs."""

        if not isinstance(token, str) or not token.strip() or "\n" in token:
            raise ValueError("Hugging Face token must be a non-empty single-line value")
        command = """
set -euo pipefail
umask 077
directory="$HOME/.cache/huggingface"
mkdir -p -m 0700 "$directory"
temporary="$(mktemp "$directory/.token.XXXXXX")"
trap 'rm -f -- "$temporary"' EXIT
cat >"$temporary"
test -s "$temporary"
chmod 0600 "$temporary"
mv -f -- "$temporary" "$directory/token"
trap - EXIT
printf 'HF_AUTH_INSTALLED\n'
""".strip()
        result = self.run(command, input_text=token)
        if "HF_AUTH_INSTALLED" not in result.stdout:
            raise Grid5000ExecutionError("remote Hugging Face auth marker is missing")

    def read_completion(self, run_id: str) -> dict[str, object]:
        """Read and validate the credential-free completion manifest."""

        _validate_run_id(run_id)
        path = (
            f'"$HOME/{REMOTE_DATA_SUBDIRECTORY}/{REMOTE_RUNS_SUBDIRECTORY}/'
            f'{run_id}/completion.json"'
        )
        result = self.run(f"test -f {path} && cat {path}")
        return _parse_completion_payload(result.stdout, run_id)

    def has_complete_checkpoint(
        self,
        run_id: str,
        *,
        output_subdirectory: str | Path,
        identity: Mapping[str, object],
        allow_failed_status: bool = False,
    ) -> bool:
        """Check for complete files bound to the exact run identity.

        A failed marker is accepted only for an explicit continuation whose
        caller has already verified the previous job is no longer live.
        """

        _validate_run_id(run_id)
        if not isinstance(allow_failed_status, bool):
            raise ValueError("allow_failed_status must be a boolean")
        identity_json = _checkpoint_identity_json(identity)
        relative = _safe_output_relative_path(output_subdirectory)
        command = _checkpoint_probe_command(
            run_id,
            relative,
            identity_json,
            allow_failed_status,
        )
        result = self.run(command)
        if "CHECKPOINT_READY" in result.stdout:
            return True
        if "CHECKPOINT_MISSING" in result.stdout:
            return False
        raise Grid5000ExecutionError("remote checkpoint probe marker is missing")

    def mark_status(self, run_id: str, status: str) -> None:
        """Atomically mark a managed run terminal without recording secrets."""

        _validate_run_id(run_id)
        if status not in {"active", "complete", "failed"}:
            raise ValueError("remote run status is invalid")
        path = f'"$HOME/{REMOTE_DATA_SUBDIRECTORY}/{REMOTE_RUNS_SUBDIRECTORY}/{run_id}"'
        marker = json.dumps(
            {"schema_version": 1, "run_id": run_id, "status": status},
            sort_keys=True,
            separators=(",", ":"),
        )
        command = f"""
set -euo pipefail
root={path}
[ -d "$root" ] && [ ! -L "$root" ]
temporary="$root/.operator-managed.json.tmp"
printf '%s\n' {shlex.quote(marker)} >"$temporary"
chmod 0600 "$temporary"
mv -f -- "$temporary" "$root/.operator-managed.json"
printf 'REMOTE_STATUS_MARKED\n'
""".strip()
        self.run(command)

    def cleanup(self, run_id: str) -> None:
        """Asynchronously remove only a terminal, marker-owned run root."""

        _validate_run_id(run_id)
        path = f'"$HOME/{REMOTE_DATA_SUBDIRECTORY}/{REMOTE_RUNS_SUBDIRECTORY}/{run_id}"'
        command = f"""
set -euo pipefail
root={path}
[ ! -L "$root" ]
[ -d "$root" ] || exit 0
    marker="$root/.operator-managed.json"
    [ -f "$marker" ] && [ ! -L "$marker" ]
    grep -Fq {shlex.quote(f'"run_id":"{run_id}"')} "$marker"
    grep -Eq '"status":"(complete|failed)"' "$marker"
    nohup rm -rf -- "$root" >/dev/null 2>&1 < /dev/null &
    printf 'REMOTE_CLEANUP_STARTED\n'
    """.strip()
        result = self.run(command)
        if result.stdout and "REMOTE_CLEANUP_STARTED" not in result.stdout:
            raise Grid5000ExecutionError("remote cleanup marker is missing")


__all__ = [
    "REMOTE_CHECKOUT_SUBDIRECTORY",
    "REMOTE_DATA_SUBDIRECTORY",
    "REMOTE_REPOSITORY_URL",
    "REMOTE_RUNS_SUBDIRECTORY",
    "Grid5000Remote",
    "RemotePreparationResult",
    "RemoteRunner",
    "SubprocessRemoteRunner",
]
