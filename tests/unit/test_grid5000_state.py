import json
import os
from collections.abc import Callable
from datetime import datetime as PythonDateTime
from pathlib import Path
from typing import Any, cast

import pytest

import osm_polygon_sentence_classifier.grid5000_state as state_module
from osm_polygon_sentence_classifier.grid5000_state import (
    AutonomousRunState,
    AutonomousStateStore,
    LegacyAmbiguousStateError,
    StateError,
    StateSecurityError,
    _finite_float,
)


def _state(run_id: str = "a" * 20) -> AutonomousRunState:
    return AutonomousRunState(
        run_id=run_id,
        phase="created",
        identity={"source_commit": "b" * 40},
    )


def test_state_store_round_trips_secure_state_and_events(tmp_path: Path) -> None:
    store = AutonomousStateStore(tmp_path / "runs")
    state = _state()

    store.create(state)
    store.append_event(state.run_id, "created", {"site": "grenoble"})

    loaded = store.load(state.run_id)
    assert loaded == state
    run_dir = tmp_path / "runs" / state.run_id
    assert run_dir.stat().st_mode & 0o777 == 0o700
    assert (run_dir / "state.json").stat().st_mode & 0o777 == 0o600
    assert (run_dir / "events.jsonl").stat().st_mode & 0o777 == 0o600


def test_state_store_uses_strict_unicode_and_numeric_json_options(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    observed: dict[str, object] = {}
    original_dumps = state_module.json.dumps

    def dumps(value: object, *args: Any, **kwargs: Any) -> str:
        observed.update(kwargs)
        return original_dumps(value, *args, **kwargs)

    monkeypatch.setattr(state_module.json, "dumps", dumps)

    AutonomousStateStore(tmp_path / "runs").create(_state())

    assert observed["ensure_ascii"] is False
    assert observed["allow_nan"] is False


def test_state_store_appends_the_requested_event_to_the_canonical_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(state_module, "_now", lambda: "2026-08-22T12:00:00+00:00")
    store = AutonomousStateStore(tmp_path / "runs")
    state = _state()
    store.create(state)

    store.append_event(state.run_id, "job-submitted", {"job_id": 123})

    event_path = tmp_path / "runs" / state.run_id / "events.jsonl"
    assert json.loads(event_path.read_text(encoding="utf-8")) == {
        "event": "job-submitted",
        "facts": {"job_id": 123},
        "schema_version": 1,
        "timestamp": "2026-08-22T12:00:00+00:00",
    }


def test_state_store_append_event_uses_the_canonical_event_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = AutonomousStateStore(tmp_path / "runs")
    state = _state()
    store.create(state)
    calls: list[tuple[Path, dict[str, object]]] = []

    monkeypatch.setattr(
        state_module,
        "_write_event",
        lambda path, payload: calls.append((path, dict(payload))),
    )

    store.append_event(state.run_id, "job-submitted", {"job_id": 123})

    assert calls[0][0] == tmp_path / "runs" / state.run_id / "events.jsonl"
    assert calls[0][1]["event"] == "job-submitted"


def test_state_store_validate_run_id_preserves_the_public_error(tmp_path: Path) -> None:
    store = AutonomousStateStore(tmp_path / "runs")
    assert store._validate_run_id("a" * 20) is None

    with pytest.raises(StateError) as error:
        store._validate_run_id("not-a-run")
    assert str(error.value) == "run ID is invalid"


def test_event_payload_has_the_stable_schema_and_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(state_module, "_now", lambda: "2026-08-22T12:00:00+00:00")

    assert state_module._event_payload("created", {"site": "grenoble"}) == {
        "schema_version": 1,
        "timestamp": "2026-08-22T12:00:00+00:00",
        "event": "created",
        "facts": {"site": "grenoble"},
    }


def test_validated_phase_preserves_the_public_errors_and_enum_value() -> None:
    assert (
        state_module._validated_phase("a" * 20, "created")
        is state_module.RunPhase.CREATED
    )

    with pytest.raises(StateError) as invalid_run_id:
        state_module._validated_phase("not-a-run", "created")
    assert str(invalid_run_id.value) == "run ID is invalid"

    with pytest.raises(StateError) as invalid_phase:
        state_module._validated_phase("a" * 20, "unknown")
    assert str(invalid_phase.value) == "run phase is invalid"


@pytest.mark.parametrize("site", [123, "", "   "])
def test_validate_state_site_rejects_invalid_values_with_the_exact_error(
    site: object,
) -> None:
    with pytest.raises(StateError) as error:
        state_module._validate_state_site(cast(Any, site))
    assert str(error.value) == "state site is invalid"


@pytest.mark.parametrize("job_id", [True, 0, -1, 1.5, "1"])
def test_validate_state_job_id_rejects_invalid_values_with_the_exact_error(
    job_id: object,
) -> None:
    with pytest.raises(StateError) as error:
        state_module._validate_state_job_id(cast(Any, job_id))
    assert str(error.value) == "state job ID is invalid"


@pytest.mark.parametrize("job_id", [None, 1, 2])
def test_validate_state_job_id_accepts_none_and_positive_integers(
    job_id: int | None,
) -> None:
    assert state_module._validate_state_job_id(job_id) is None


def test_required_state_fields_returns_the_required_values() -> None:
    payload = {
        "schema_version": 1,
        "run_id": "a" * 20,
        "phase": "created",
        "identity": {"source_commit": "b" * 40},
    }

    assert state_module._required_state_fields(payload) == (
        "a" * 20,
        "created",
        {"source_commit": "b" * 40},
    )


@pytest.mark.parametrize("field", ["run_id", "phase"])
def test_required_state_fields_rejects_a_non_string_identity_or_phase(
    field: str,
) -> None:
    payload: dict[str, object] = {
        "schema_version": 1,
        "run_id": "a" * 20,
        "phase": "created",
        "identity": {},
    }
    payload[field] = 1

    with pytest.raises(StateError) as error:
        state_module._required_state_fields(payload)
    assert str(error.value) == "state identity or phase is invalid"


def test_required_state_fields_rejects_a_non_mapping_identity() -> None:
    payload = {
        "schema_version": 1,
        "run_id": "a" * 20,
        "phase": "created",
        "identity": [],
    }

    with pytest.raises(StateError) as error:
        state_module._required_state_fields(payload)
    assert str(error.value) == "state identity is invalid"


def test_state_schema_and_required_keys_have_stable_errors() -> None:
    with pytest.raises(StateError) as schema_error:
        state_module._require_state_schema({"schema_version": 2})
    assert str(schema_error.value) == "state schema version is unsupported"

    for missing in ("run_id", "phase", "identity"):
        payload: dict[str, object] = {
            "run_id": "a" * 20,
            "phase": "created",
            "identity": {},
        }
        payload.pop(missing)
        with pytest.raises(StateError) as error:
            state_module._raw_required_state_fields(payload)
        assert str(error.value) == "state document is incomplete"


def test_state_field_parsers_preserve_defaults_and_reject_bad_types() -> None:
    assert state_module._state_site({}) is None
    assert state_module._state_site({"site": "nancy"}) == "nancy"
    assert state_module._state_job_id({}) is None
    assert state_module._state_job_id({"job_id": 123}) == 123
    assert state_module._state_facts({}) == {}
    assert state_module._state_facts({"facts": {"site": "nancy"}}) == {"site": "nancy"}
    assert state_module._state_timestamp(
        {"updated_at": "2026-08-22T12:00:00+00:00"}
    ) == ("2026-08-22T12:00:00+00:00")

    invalid_fields = (
        ({"site": 1}, "state site is invalid"),
        ({"job_id": True}, "state job ID is invalid"),
        ({"facts": []}, "state facts are invalid"),
        ({"updated_at": 1}, "state timestamp is invalid"),
    )
    for payload, message in invalid_fields:
        parser = {
            "site": state_module._state_site,
            "job_id": state_module._state_job_id,
            "facts": state_module._state_facts,
            "updated_at": state_module._state_timestamp,
        }[next(iter(payload))]
        with pytest.raises(StateError) as error:
            parser(payload)
        assert str(error.value) == message


def test_sanitizers_preserve_json_values_and_report_invalid_values_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert state_module._sanitize_scalar(None) is None
    assert state_module._sanitize_scalar("text") == "text"
    assert state_module._sanitize_scalar(True) is True
    assert state_module._sanitize_scalar(3) == 3
    assert state_module._sanitize_scalar(1.5) == 1.5

    with pytest.raises(StateError) as scalar_error:
        state_module._sanitize_scalar(object())
    assert str(scalar_error.value) == "state facts must be JSON-compatible values"

    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(StateError) as finite_error:
            state_module._finite_float(value)
        assert str(finite_error.value) == "state facts must contain finite JSON values"

    with pytest.raises(StateError) as key_error:
        state_module._sanitize_mapping_value({1: "value"})
    assert str(key_error.value) == "state facts must use string keys"

    assert state_module._sanitize_mapping({"nested": [1, True, None]}) == {
        "nested": [1, True, None]
    }
    with pytest.raises(StateError) as mapping_error:
        state_module._sanitize_mapping([])  # ty: ignore[invalid-argument-type]
    assert str(mapping_error.value) == "state identity/facts must be mappings"

    monkeypatch.setattr(state_module, "_sanitize", lambda _value: [])
    with pytest.raises(StateError) as result_error:
        state_module._sanitize_mapping({"value": 1})
    assert str(result_error.value) == "state identity/facts must be mappings"


def test_reject_symlinks_validates_absolute_paths_and_components(
    tmp_path: Path,
) -> None:
    with pytest.raises(StateSecurityError) as relative_error:
        state_module._reject_symlinks(Path("relative/path"))
    assert str(relative_error.value) == "state root must be absolute"

    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(StateSecurityError) as symlink_error:
        state_module._reject_symlinks(link / "child")
    assert str(symlink_error.value) == "state path cannot contain symlinks"


def test_check_mode_and_document_path_enforce_the_secure_state_document_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(StateSecurityError) as missing_error:
        state_module._check_mode(missing, 0o600, "state document")
    assert str(missing_error.value) == "state document is missing or symlinked"

    target = tmp_path / "target"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(StateSecurityError) as link_error:
        state_module._check_mode(link, 0o600, "state document")
    assert str(link_error.value) == "state document is missing or symlinked"

    directory = tmp_path / "run"
    directory.mkdir()
    state_path = directory / "state.json"
    state_path.write_text("{}", encoding="utf-8")
    state_path.chmod(0o600)
    assert state_module._state_document_path(directory) == state_path

    missing_document = tmp_path / "empty-run"
    missing_document.mkdir()
    with pytest.raises(StateError) as missing_document_error:
        state_module._state_document_path(missing_document)
    assert str(missing_document_error.value) == "state document is missing or unsafe"

    linked_document = tmp_path / "linked-run"
    linked_document.mkdir()
    linked_state = linked_document / "state.json"
    linked_state.symlink_to(target)
    with pytest.raises(StateError) as linked_document_error:
        state_module._state_document_path(linked_document)
    assert str(linked_document_error.value) == "state document is missing or unsafe"

    checks: list[tuple[Path, int, str]] = []
    monkeypatch.setattr(
        state_module,
        "_check_mode",
        lambda path, expected, label: checks.append((path, expected, label)),
    )
    assert state_module._state_document_path(directory) == state_path
    assert checks == [(state_path, 0o600, "state document")]


def test_existing_run_directory_enforces_mode_and_security_labels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    directory = tmp_path / "run"
    directory.mkdir()
    directory.chmod(0o700)
    assert state_module._require_existing_run_directory(directory) is None

    missing = tmp_path / "missing"
    with pytest.raises(StateError) as missing_error:
        state_module._require_existing_run_directory(missing)
    assert str(missing_error.value) == "run state directory is missing or unsafe"

    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(StateError) as link_error:
        state_module._require_existing_run_directory(link)
    assert str(link_error.value) == "run state directory is missing or unsafe"

    checks: list[tuple[Path, int, str]] = []
    monkeypatch.setattr(
        state_module,
        "_check_mode",
        lambda path, expected, label: checks.append((path, expected, label)),
    )
    state_module._require_existing_run_directory(directory)
    assert checks == [(directory, 0o700, "run state directory")]


@pytest.mark.parametrize("event", ["", "line\nbreak", "carriage\rreturn"])
def test_validated_event_name_rejects_empty_or_multiline_names(event: str) -> None:
    with pytest.raises(StateError) as error:
        state_module._validated_event_name(event)
    assert str(error.value) == "event name is invalid"


def test_validated_event_name_returns_a_single_line_name_unchanged() -> None:
    assert state_module._validated_event_name("job-submitted") == "job-submitted"


def test_state_store_load_rejects_a_symlinked_run_directory_exactly(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runs"
    root.mkdir()
    run_id = "a" * 20
    target = tmp_path / "outside"
    target.mkdir()
    (root / run_id).symlink_to(target, target_is_directory=True)

    with pytest.raises(StateSecurityError) as error:
        AutonomousStateStore(root).load(run_id)

    assert str(error.value) == "run state directory cannot be a symlink"


def test_state_store_load_reports_unsafe_directory_permissions_exactly(
    tmp_path: Path,
) -> None:
    store = AutonomousStateStore(tmp_path / "runs")
    state = _state()
    store.create(state)
    run_directory = tmp_path / "runs" / state.run_id
    run_directory.chmod(0o755)

    with pytest.raises(StateSecurityError) as error:
        store.load(state.run_id)

    assert str(error.value) == "run state directory has unsafe permissions"


def test_state_store_load_rejects_a_state_identity_mismatch_exactly(
    tmp_path: Path,
) -> None:
    store = AutonomousStateStore(tmp_path / "runs")
    state = _state()
    store.create(state)
    state_path = tmp_path / "runs" / state.run_id / "state.json"
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["run_id"] = "b" * 20
    state_path.write_text(json.dumps(payload), encoding="utf-8")
    state_path.chmod(0o600)

    with pytest.raises(StateError) as error:
        store.load(state.run_id)

    assert str(error.value) == "state identity does not match its directory"


def test_state_store_ensures_the_root_with_recursive_secure_creation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "nested" / "runs"
    store = AutonomousStateStore(root)
    mkdir_calls: list[tuple[Path, tuple[object, ...], dict[str, object]]] = []
    chmod_calls: list[tuple[Path, int]] = []

    monkeypatch.setattr(
        state_module.Path,
        "mkdir",
        lambda path, *args, **kwargs: mkdir_calls.append((path, args, kwargs)),
    )
    monkeypatch.setattr(
        state_module.os,
        "chmod",
        lambda path, mode: chmod_calls.append((Path(path), mode)),
    )

    store._ensure_root()

    assert mkdir_calls == [
        (root, (), {"parents": True, "exist_ok": True, "mode": 0o700})
    ]
    assert chmod_calls == [(root, 0o700)]


def test_state_store_wraps_root_creation_failures_with_the_public_error_message(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = AutonomousStateStore(tmp_path / "runs")
    cause = OSError("permission denied")

    def fail_mkdir(*_args: object, **_kwargs: object) -> None:
        raise cause

    monkeypatch.setattr(state_module.Path, "mkdir", fail_mkdir)

    with pytest.raises(StateSecurityError) as error:
        store._ensure_root()

    assert str(error.value) == "state root cannot be created securely"
    assert error.value.__cause__ is cause


def test_state_store_ensures_a_new_run_directory_with_secure_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "runs"
    directory = root / ("a" * 20)
    store = AutonomousStateStore(root)
    monkeypatch.setattr(store, "_ensure_root", lambda: None)
    monkeypatch.setattr(store, "_directory", lambda _run_id: directory)
    mkdir_calls: list[tuple[Path, tuple[object, ...], dict[str, object]]] = []
    chmod_calls: list[tuple[Path, int]] = []
    monkeypatch.setattr(
        state_module.Path,
        "mkdir",
        lambda path, *args, **kwargs: mkdir_calls.append((path, args, kwargs)),
    )
    monkeypatch.setattr(
        state_module.os,
        "chmod",
        lambda path, mode: chmod_calls.append((Path(path), mode)),
    )

    assert store._ensure_directory("a" * 20) == directory
    assert mkdir_calls == [(directory, (), {"mode": 0o700})]
    assert chmod_calls == [(directory, 0o700)]


def test_state_store_reports_a_symlinked_run_directory_exactly(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runs"
    root.mkdir()
    run_id = "a" * 20
    target = tmp_path / "outside"
    target.mkdir()
    (root / run_id).symlink_to(target, target_is_directory=True)
    store = AutonomousStateStore(root)

    with pytest.raises(StateSecurityError) as error:
        store._ensure_directory(run_id)

    assert str(error.value) == "run state directory cannot be a symlink"


def test_state_store_reports_a_duplicate_run_directory_exactly(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    root.mkdir()
    run_id = "a" * 20
    directory = root / run_id
    directory.mkdir()
    store = AutonomousStateStore(root)

    with pytest.raises(StateError) as error:
        store._ensure_directory(run_id)

    assert str(error.value) == "run already has durable state"


def test_state_store_wraps_run_directory_creation_failures_exactly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "runs"
    directory = root / ("a" * 20)
    store = AutonomousStateStore(root)
    monkeypatch.setattr(store, "_ensure_root", lambda: None)
    monkeypatch.setattr(store, "_directory", lambda _run_id: directory)

    def fail_mkdir(*_args: object, **_kwargs: object) -> None:
        raise OSError("permission denied")

    monkeypatch.setattr(state_module.Path, "mkdir", fail_mkdir)

    with pytest.raises(StateSecurityError) as error:
        store._ensure_directory("a" * 20)

    assert str(error.value) == "run state directory cannot be created"


def test_write_event_rejects_a_symlink_with_the_public_error_message(
    tmp_path: Path,
) -> None:
    target = tmp_path / "outside.jsonl"
    target.write_text("preserve", encoding="utf-8")
    events = tmp_path / "events.jsonl"
    events.symlink_to(target)

    with pytest.raises(StateSecurityError) as error:
        state_module._write_event(events, {"event": "created"})

    assert str(error.value) == "events document cannot be a symlink"
    assert target.read_text(encoding="utf-8") == "preserve"


def test_write_event_uses_explicit_utf8_and_canonical_single_line_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events = tmp_path / "events.jsonl"
    open_calls: list[tuple[Path, tuple[object, ...], dict[str, object]]] = []
    original_open = Path.open

    def recording_open(path: Path, *args: object, **kwargs: object) -> Any:
        open_calls.append((path, args, kwargs))
        return cast(Any, original_open)(path, *args, **kwargs)

    monkeypatch.setattr(state_module.Path, "open", recording_open)

    state_module._write_event(events, {"z": "cafe", "a": 1})

    assert open_calls == [(events, ("a",), {"encoding": "utf-8"})]
    assert events.read_text(encoding="utf-8") == '{"a": 1, "z": "cafe"}\n'


def test_write_event_wraps_os_errors_with_the_public_error_message(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events = tmp_path / "events.jsonl"

    def fail_open(*_args: object, **_kwargs: object) -> Any:
        raise OSError("disk full")

    monkeypatch.setattr(state_module.Path, "open", fail_open)

    with pytest.raises(StateError) as error:
        state_module._write_event(events, {"event": "failed"})

    assert str(error.value) == "event document cannot be written"


def test_read_state_document_uses_utf8_and_returns_json_objects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.json"
    path.write_text('{"schema_version": 1}', encoding="utf-8")
    read_calls: list[dict[str, object]] = []
    original_read_text = Path.read_text

    def recording_read_text(current: Path, *args: object, **kwargs: object) -> str:
        read_calls.append({"path": current, "args": args, **kwargs})
        return cast(Any, original_read_text)(current, *args, **kwargs)

    monkeypatch.setattr(state_module.Path, "read_text", recording_read_text)

    assert state_module._read_state_document(path) == {"schema_version": 1}
    assert read_calls == [{"path": path, "args": (), "encoding": "utf-8"}]


def test_read_state_document_wraps_read_failures_with_the_public_error_message(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(StateError) as error:
        state_module._read_state_document(path)

    assert str(error.value) == "state document cannot be read"
    assert isinstance(error.value.__cause__, json.JSONDecodeError)


def test_read_state_document_reports_legacy_ambiguity_exactly(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.json"
    path.write_text('{"phase": "submitting"}', encoding="utf-8")

    with pytest.raises(LegacyAmbiguousStateError) as error:
        state_module._read_state_document(path)

    assert str(error.value) == (
        "legacy Grid'5000 state is ambiguous; reconcile before starting"
    )


def test_read_state_document_rejects_non_object_json_exactly(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(StateError) as error:
        state_module._read_state_document(path)

    assert str(error.value) == "state document must be an object"


def test_read_legacy_state_uses_utf8_and_accepts_a_mapping_without_schema(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.json"
    path.write_text('{"phase": "submitting"}', encoding="utf-8")
    read_calls: list[dict[str, object]] = []
    original_read_text = Path.read_text

    def recording_read_text(current: Path, *args: object, **kwargs: object) -> str:
        read_calls.append({"path": current, "args": args, **kwargs})
        return cast(Any, original_read_text)(current, *args, **kwargs)

    monkeypatch.setattr(state_module.Path, "read_text", recording_read_text)

    assert state_module._read_legacy_state(path) is None
    assert read_calls == [{"path": path, "args": (), "encoding": "utf-8"}]


def test_read_legacy_state_wraps_read_failures_with_the_public_error_message(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(StateError) as error:
        state_module._read_legacy_state(path)

    assert str(error.value) == "legacy state document cannot be read"
    assert isinstance(error.value.__cause__, json.JSONDecodeError)


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"schema_version": 1},
    ],
)
def test_read_legacy_state_rejects_non_legacy_documents_exactly(
    tmp_path: Path, payload: object
) -> None:
    path = tmp_path / "state.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StateError) as error:
        state_module._read_legacy_state(path)

    assert str(error.value) == "state is not a legacy document"


def test_state_store_writes_canonical_utf8_json_and_replaces_state_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = AutonomousStateStore(tmp_path / "runs")
    state = AutonomousRunState(
        run_id="a" * 20,
        phase="created",
        identity={"z": "café", "a": "first", "source_commit": "b" * 40},
    )
    mkstemp_calls: list[dict[str, object]] = []
    original_mkstemp = cast(
        Callable[..., tuple[int, str]], state_module.tempfile.mkstemp
    )

    def recording_mkstemp(*args: object, **kwargs: object) -> tuple[int, str]:
        del args
        mkstemp_calls.append(kwargs)
        return original_mkstemp(**kwargs)

    replacements: list[tuple[Path, Path]] = []
    original_replace = state_module.os.replace

    def recording_replace(
        source: str | os.PathLike[str], target: str | os.PathLike[str]
    ) -> None:
        replacements.append((Path(source), Path(target)))
        original_replace(source, target)

    monkeypatch.setattr(state_module.tempfile, "mkstemp", recording_mkstemp)
    monkeypatch.setattr(state_module.os, "replace", recording_replace)

    store.create(state)

    run_directory = tmp_path / "runs" / state.run_id
    assert mkstemp_calls == [
        {"dir": run_directory, "prefix": ".state-", "suffix": ".tmp"}
    ]
    assert replacements[0][0].parent == run_directory
    assert replacements[0][1] == run_directory / "state.json"
    raw = (run_directory / "state.json").read_text(encoding="utf-8")
    assert "café" in raw
    assert "\\u00e9" not in raw
    assert list(json.loads(raw)["identity"]) == ["a", "source_commit", "z"]


def test_state_store_opens_the_temporary_file_as_utf8_text(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = AutonomousStateStore(tmp_path / "runs")
    calls: list[tuple[int, tuple[object, ...], dict[str, object]]] = []
    original_fdopen = cast(Callable[..., Any], state_module.os.fdopen)

    def recording_fdopen(descriptor: int, *args: object, **kwargs: object) -> Any:
        calls.append((descriptor, args, kwargs))
        return original_fdopen(descriptor, *args, **kwargs)

    monkeypatch.setattr(state_module.os, "fdopen", recording_fdopen)

    store.create(_state())

    assert len(calls) == 1
    assert calls[0][1] == ("w",)
    assert calls[0][2] == {"encoding": "utf-8"}


def test_state_store_wraps_serialization_failures_without_cleanup_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = AutonomousStateStore(tmp_path / "runs")

    def fail_dumps(*_args: object, **_kwargs: object) -> str:
        raise TypeError("not serializable")

    monkeypatch.setattr(state_module.json, "dumps", fail_dumps)

    with pytest.raises(StateError) as error:
        store.create(_state())

    assert str(error.value) == "state document cannot be written"


def test_state_store_rejects_non_finite_state_facts(
    tmp_path: Path,
) -> None:
    store = AutonomousStateStore(tmp_path / "runs")
    state = _state()
    object.__setattr__(
        state,
        "identity",
        {"source_commit": "b" * 40, "value": float("nan")},
    )

    with pytest.raises(StateError) as error:
        store.create(state)

    assert str(error.value) == "state document cannot be written"


def test_state_store_closes_a_temporary_descriptor_after_fdopen_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = AutonomousStateStore(tmp_path / "runs")
    run_directory = tmp_path / "runs" / ("a" * 20)
    run_directory.mkdir(parents=True)
    temporary_path = run_directory / ".state-test.tmp"
    temporary_path.write_text("partial", encoding="utf-8")
    closed: list[int] = []

    monkeypatch.setattr(
        state_module.tempfile,
        "mkstemp",
        lambda **_kwargs: (17, str(temporary_path)),
    )
    monkeypatch.setattr(
        state_module.os,
        "fdopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("cannot wrap")),
    )
    monkeypatch.setattr(state_module.os, "close", closed.append)

    with pytest.raises(StateError) as error:
        store._write_state(run_directory, _state())

    assert str(error.value) == "state document cannot be written"
    assert closed == [17]
    assert not temporary_path.exists()


def test_state_store_suppresses_descriptor_close_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = AutonomousStateStore(tmp_path / "runs")
    run_directory = tmp_path / "runs" / ("a" * 20)
    run_directory.mkdir(parents=True)
    temporary_path = run_directory / ".state-test.tmp"
    temporary_path.write_text("partial", encoding="utf-8")

    monkeypatch.setattr(
        state_module.tempfile,
        "mkstemp",
        lambda **_kwargs: (17, str(temporary_path)),
    )
    monkeypatch.setattr(
        state_module.os,
        "fdopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("cannot wrap")),
    )
    monkeypatch.setattr(
        state_module.os,
        "close",
        lambda _descriptor: (_ for _ in ()).throw(OSError("cannot close")),
    )

    with pytest.raises(StateError) as error:
        store._write_state(run_directory, _state())

    assert str(error.value) == "state document cannot be written"


def test_state_store_suppresses_temporary_unlink_failures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = AutonomousStateStore(tmp_path / "runs")
    run_directory = tmp_path / "runs" / ("a" * 20)
    run_directory.mkdir(parents=True)
    temporary_path = run_directory / ".state-test.tmp"
    temporary_path.write_text("partial", encoding="utf-8")

    monkeypatch.setattr(
        state_module.tempfile,
        "mkstemp",
        lambda **_kwargs: (17, str(temporary_path)),
    )
    monkeypatch.setattr(
        state_module.os,
        "fdopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("cannot wrap")),
    )
    monkeypatch.setattr(
        state_module.Path,
        "unlink",
        lambda _path: (_ for _ in ()).throw(OSError("cannot unlink")),
    )

    with pytest.raises(StateError) as error:
        store._write_state(run_directory, _state())

    assert str(error.value) == "state document cannot be written"


def test_state_store_save_does_not_follow_a_temporary_symlink(
    tmp_path: Path,
) -> None:
    store = AutonomousStateStore(tmp_path / "runs")
    state = _state()
    store.create(state)

    run_dir = tmp_path / "runs" / state.run_id
    outside = tmp_path / "outside.json"
    outside.write_text("do not overwrite", encoding="utf-8")
    temporary = run_dir / ".state.json.tmp"
    temporary.symlink_to(outside)

    store.save(state)

    assert outside.read_text(encoding="utf-8") == "do not overwrite"
    assert temporary.is_symlink()
    assert store.load(state.run_id) == state


def test_state_store_rejects_secret_facts(tmp_path: Path) -> None:
    store = AutonomousStateStore(tmp_path / "runs")
    store.create(_state())

    with pytest.raises(StateSecurityError, match="unsafe"):
        store.append_event("a" * 20, "failed", {"HF_TOKEN": "secret"})


def test_legacy_ambiguous_state_is_detected_without_overwrite(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    run_dir = root / ("a" * 20)
    run_dir.mkdir(parents=True)
    run_dir.chmod(0o700)
    (run_dir / "state.json").write_text(
        json.dumps({"phase": "submitting", "job_id": None}),
        encoding="utf-8",
    )
    (run_dir / "state.json").chmod(0o600)
    store = AutonomousStateStore(root)

    with pytest.raises(LegacyAmbiguousStateError):
        store.load("a" * 20)

    assert json.loads((run_dir / "state.json").read_text())["phase"] == "submitting"


def test_legacy_state_path_preserves_exact_security_labels_and_filename(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    directory = tmp_path / "run"
    directory.mkdir()
    state_path = directory / "state.json"
    state_path.write_text("{}", encoding="utf-8")
    checks: list[tuple[Path, int, str]] = []

    def record_check(path: Path, expected: int, label: str) -> None:
        checks.append((path, expected, label))

    monkeypatch.setattr(state_module, "_check_mode", record_check)

    assert state_module._legacy_state_path(directory, ()) == state_path
    assert checks == [
        (directory, 0o700, "legacy run state directory"),
        (state_path, 0o600, "legacy state document"),
    ]


def test_legacy_state_path_reports_the_exact_active_job_error(
    tmp_path: Path,
) -> None:
    with pytest.raises(LegacyAmbiguousStateError) as error:
        state_module._legacy_state_path(tmp_path, (123,))

    assert str(error.value) == (
        "legacy Grid'5000 state cannot be reconciled while jobs are active"
    )


@pytest.mark.parametrize("unsafe_kind", ["missing", "symlink"])
def test_legacy_state_path_rejects_missing_or_symlinked_documents(
    tmp_path: Path,
    unsafe_kind: str,
) -> None:
    directory = tmp_path / "run"
    directory.mkdir()
    directory.chmod(0o700)
    state_path = directory / "state.json"
    if unsafe_kind == "symlink":
        target = tmp_path / "outside.json"
        target.write_text("{}", encoding="utf-8")
        state_path.symlink_to(target)

    with pytest.raises(StateError) as error:
        state_module._legacy_state_path(directory, ())

    assert str(error.value) == "legacy state document is missing or unsafe"


def test_legacy_reconciliation_archives_only_when_no_jobs_are_active(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runs"
    run_id = "a" * 20
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    run_dir.chmod(0o700)
    (run_dir / "state.json").write_text(
        json.dumps({"phase": "submitting", "job_id": None}),
        encoding="utf-8",
    )
    (run_dir / "state.json").chmod(0o600)
    store = AutonomousStateStore(root)

    archived = store.reconcile_legacy(run_id, active_job_ids=())

    assert not run_dir.exists()
    assert archived.exists()
    assert (archived / "state.json").exists()


def test_legacy_reconciliation_passes_the_run_id_to_the_archive_operation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "runs"
    run_id = "a" * 20
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    run_dir.chmod(0o700)
    state_path = run_dir / "state.json"
    state_path.write_text(json.dumps({"phase": "submitting"}), encoding="utf-8")
    state_path.chmod(0o600)
    archived = root / "legacy-ambiguous" / f"{run_id}-archive"
    calls: list[tuple[Path, Path, str]] = []

    def record_archive(
        directory: Path, archive_root: Path, archived_run_id: str
    ) -> Path:
        calls.append((directory, archive_root, archived_run_id))
        return archived

    monkeypatch.setattr(state_module, "_archive_legacy_state", record_archive)

    assert (
        AutonomousStateStore(root).reconcile_legacy(run_id, active_job_ids=())
        == archived
    )
    assert calls == [(run_dir, root / "legacy-ambiguous", run_id)]


def test_legacy_archive_directory_validates_an_existing_secure_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "runs"
    archive = root / "legacy-ambiguous"
    archive.mkdir(parents=True)
    archive.chmod(0o700)
    checks: list[tuple[Path, int, str]] = []

    def record_check(path: Path, expected: int, label: str) -> None:
        checks.append((path, expected, label))

    monkeypatch.setattr(state_module, "_check_mode", record_check)

    assert state_module._legacy_archive_directory(root) == archive
    assert checks == [(archive, 0o700, "legacy archive directory")]


def test_legacy_archive_directory_creates_and_hardens_missing_directories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "nested" / "runs"
    mkdir_calls: list[tuple[Path, tuple[object, ...], dict[str, object]]] = []
    chmod_calls: list[tuple[Path, int]] = []
    original_mkdir = Path.mkdir

    def record_mkdir(path: Path, *args: object, **kwargs: object) -> None:
        mkdir_calls.append((path, args, kwargs))
        cast(Any, original_mkdir)(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", record_mkdir)
    monkeypatch.setattr(
        state_module.os,
        "chmod",
        lambda path, mode: chmod_calls.append((Path(path), mode)),
    )

    archive = state_module._legacy_archive_directory(root)

    assert archive == root / "legacy-ambiguous"
    archive_calls = [call for call in mkdir_calls if call[0] == archive]
    assert archive_calls[0] == (
        archive,
        (),
        {"parents": True, "exist_ok": True, "mode": 0o700},
    )
    assert chmod_calls == [(archive, 0o700)]
    assert archive.exists()


def test_legacy_reconciliation_refuses_while_any_job_is_active(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runs"
    run_id = "a" * 20
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    run_dir.chmod(0o700)
    (run_dir / "state.json").write_text(
        json.dumps({"phase": "submitting", "job_id": None}),
        encoding="utf-8",
    )
    (run_dir / "state.json").chmod(0o600)
    store = AutonomousStateStore(root)

    with pytest.raises(LegacyAmbiguousStateError, match="active"):
        store.reconcile_legacy(run_id, active_job_ids=(123,))

    assert run_dir.exists()


def test_legacy_reconciliation_rejects_an_insecure_run_directory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "runs"
    run_id = "a" * 20
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    run_dir.chmod(0o755)
    (run_dir / "state.json").write_text(
        json.dumps({"phase": "submitting", "job_id": None}),
        encoding="utf-8",
    )
    (run_dir / "state.json").chmod(0o600)
    store = AutonomousStateStore(root)

    with pytest.raises(StateSecurityError, match="permissions"):
        store.reconcile_legacy(run_id, active_job_ids=())

    assert run_dir.exists()


def test_parse_timestamp_accepts_utc_z_suffix_and_offset() -> None:
    assert state_module._parse_timestamp("2026-08-22T12:34:56Z") is None
    assert state_module._parse_timestamp("2026-08-22T12:34:56+00:00") is None


def test_parse_timestamp_normalizes_z_before_calling_the_parser(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []
    expected = PythonDateTime(2026, 8, 22, 12, 34, 56, tzinfo=state_module.UTC)

    class Parser:
        @staticmethod
        def fromisoformat(value: str) -> PythonDateTime:
            observed.append(value)
            return expected

    monkeypatch.setattr(state_module, "datetime", Parser)

    state_module._parse_timestamp("2026-08-22T12:34:56Z")

    assert observed == ["2026-08-22T12:34:56+00:00"]


def test_parse_timestamp_reports_an_invalid_iso_value_exactly() -> None:
    with pytest.raises(StateError) as error:
        state_module._parse_timestamp("not-a-timestamp")

    assert str(error.value) == "state timestamp is not ISO-8601"
    assert isinstance(error.value.__cause__, ValueError)


def test_parse_timestamp_requires_a_valid_timezone_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ParsedTimestamp:
        tzinfo = object()

        def utcoffset(self) -> None:
            return None

    class DateTimeFactory:
        @staticmethod
        def fromisoformat(value: str) -> ParsedTimestamp:
            assert value == "synthetic"
            return ParsedTimestamp()

    monkeypatch.setattr(state_module, "datetime", DateTimeFactory)

    with pytest.raises(StateError) as error:
        state_module._parse_timestamp("synthetic")

    assert str(error.value) == "state timestamp must be timezone-aware"


def test_archive_legacy_state_uses_utc_and_the_stable_timestamp_format(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    directory = tmp_path / "legacy-run"
    archive_root = tmp_path / "legacy-ambiguous"
    directory.mkdir()
    archive_root.mkdir()
    observed: dict[str, list[object]] = {"timezones": [], "formats": []}

    class FixedTimestamp:
        def strftime(self, format_string: str) -> str:
            observed["formats"].append(format_string)
            return "20260822T123456Z"

    class FixedDateTime:
        @classmethod
        def now(cls, timezone: object) -> FixedTimestamp:
            observed["timezones"].append(timezone)
            return FixedTimestamp()

    monkeypatch.setattr(state_module, "datetime", FixedDateTime)

    archived = state_module._archive_legacy_state(
        directory,
        archive_root,
        "a" * 20,
    )

    assert archived == archive_root / ("a" * 20 + "-20260822T123456Z")
    assert observed == {
        "timezones": [state_module.UTC],
        "formats": ["%Y%m%dT%H%M%SZ"],
    }
    assert archived.is_dir()
    assert not directory.exists()


def test_archive_legacy_state_rejects_an_existing_target_exactly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    directory = tmp_path / "legacy-run"
    archive_root = tmp_path / "legacy-ambiguous"
    directory.mkdir()
    archive_root.mkdir()
    (archive_root / ("a" * 20 + "-20260822T123456Z")).mkdir()

    class FixedTimestamp:
        def strftime(self, _format_string: str) -> str:
            return "20260822T123456Z"

    class FixedDateTime:
        @classmethod
        def now(cls, _timezone: object) -> FixedTimestamp:
            return FixedTimestamp()

    monkeypatch.setattr(state_module, "datetime", FixedDateTime)

    with pytest.raises(StateError) as error:
        state_module._archive_legacy_state(directory, archive_root, "a" * 20)

    assert str(error.value) == "legacy archive target already exists"


def test_archive_legacy_state_wraps_move_failures_with_the_public_message(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    directory = tmp_path / "legacy-run"
    archive_root = tmp_path / "legacy-ambiguous"
    directory.mkdir()
    archive_root.mkdir()
    underlying = OSError("move failed")

    def fail_move(*_args: object, **_kwargs: object) -> None:
        raise underlying

    monkeypatch.setattr(state_module.shutil, "move", fail_move)

    with pytest.raises(StateError) as error:
        state_module._archive_legacy_state(directory, archive_root, "a" * 20)

    assert str(error.value) == "legacy state could not be archived"
    assert error.value.__cause__ is underlying


def test_finite_float_accepts_finite_values_and_rejects_non_finite_values() -> None:
    assert _finite_float(1.25) == 1.25

    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(StateError, match="finite"):
            _finite_float(value)
