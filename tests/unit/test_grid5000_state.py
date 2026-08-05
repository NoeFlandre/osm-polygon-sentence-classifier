import json
from pathlib import Path

import pytest

from osm_polygon_sentence_classifier.grid5000_state import (
    AutonomousRunState,
    AutonomousStateStore,
    LegacyAmbiguousStateError,
    StateSecurityError,
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
