from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from osm_polygon_sentence_classifier import ablation_study
from osm_polygon_sentence_classifier.ablation_study import (
    ABLATION_STUDY_ID,
    ABLATION_TRACKING_PROJECT,
    AblationStudyError,
    AblationStudyStateStore,
    baseline_ablation_definitions,
    build_ablation_training_config,
    planned_ablation_runs,
    rank_screening_results,
    render_study_documents,
    study_specification,
    study_specification_fingerprint,
)
from osm_polygon_sentence_classifier.grid5000 import Grid5000ConfigurationError
from osm_polygon_sentence_classifier.grid5000_state import RunPhase


def test_screening_matrix_has_one_control_and_one_factor_per_variant() -> None:
    definitions = baseline_ablation_definitions()

    assert [definition.ablation_id for definition in definitions] == [
        "a00-baseline-head-256-lr3e-4",
        "a01-head-128",
        "a02-head-512",
        "a03-head-lr1e-4",
        "a04-head-lr1e-3",
        "a05-balanced-head",
        "a06-last2-256",
    ]
    assert definitions[0].max_length == 256
    assert definitions[0].learning_rate == pytest.approx(3e-4)
    assert definitions[0].trainable_layers == "head"
    assert definitions[0].class_weight_mode == "none"
    assert definitions[5].class_weight_mode == "balanced"
    assert definitions[6].trainable_layers == "last2"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"study_id": "bad/name"},
            "study_id must be a clean single-line name",
        ),
        (
            {"study_id": "bad\\name"},
            "study_id must be a clean single-line name",
        ),
        (
            {"study_id": "bad\nname"},
            "study_id must be a clean single-line name",
        ),
        ({"task_name": ""}, "study task and tracking project are required"),
        ({"tracking_project": ""}, "study task and tracking project are required"),
        ({"definitions": ()}, "at least one ablation definition is required"),
        ({"replication_seeds": ()}, "at least one replication seed is required"),
        ({"validation_fraction": -0.1}, "split fractions must be non-negative"),
        ({"test_fraction": -0.1}, "split fractions must be non-negative"),
        (
            {"validation_fraction": 0.8, "test_fraction": 0.3},
            "split fractions must sum to at most one",
        ),
        ({"max_steps": 0}, "max_steps must be positive"),
    ],
)
def test_ablation_protocol_rejects_invalid_immutable_settings(
    changes: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(AblationStudyError, match=f"^{message}$"):
        replace(ablation_study.landuse_ablation_protocol(), **changes)


def test_ablation_protocol_accepts_zero_fraction_and_positive_budget() -> None:
    protocol = replace(
        ablation_study.landuse_ablation_protocol(),
        validation_fraction=0.0,
        test_fraction=0.0,
        max_steps=1,
    )

    assert protocol.validation_fraction == 0.0
    assert protocol.test_fraction == 0.0
    assert protocol.max_steps == 1


def test_ablation_protocol_accepts_a_single_full_validation_split() -> None:
    protocol = replace(
        ablation_study.landuse_ablation_protocol(),
        study_id="Xclean",
        validation_fraction=1.0,
        test_fraction=0.0,
        max_steps=1,
    )

    assert protocol.study_id == "Xclean"
    assert protocol.validation_fraction + protocol.test_fraction == 1.0


@pytest.mark.parametrize(
    ("gpu_memory_mb", "walltime_seconds"),
    [(0, 1), (1, 0)],
)
def test_ablation_resource_limits_reject_each_non_positive_dimension(
    gpu_memory_mb: int,
    walltime_seconds: int,
) -> None:
    with pytest.raises(
        AblationStudyError,
        match="^GPU memory and walltime must be positive$",
    ):
        ablation_study._validate_resource_limits(gpu_memory_mb, walltime_seconds)


def test_ablation_resource_limits_accept_the_smallest_positive_dimensions() -> None:
    ablation_study._validate_resource_limits(1, 1)


@pytest.mark.parametrize(
    ("max_workers", "max_continuations"),
    [(0, 1), (1, 0)],
)
def test_ablation_worker_limits_reject_each_non_positive_dimension(
    max_workers: int,
    max_continuations: int,
) -> None:
    with pytest.raises(
        AblationStudyError,
        match="^worker and continuation limits must be positive$",
    ):
        ablation_study._validate_worker_limits(max_workers, max_continuations)


def test_ablation_worker_limits_accept_the_smallest_positive_dimensions() -> None:
    ablation_study._validate_worker_limits(1, 1)


def test_ablation_controller_policy_rejects_a_non_boolean_permission() -> None:
    with pytest.raises(
        AblationStudyError,
        match="^source commit update permission must be a boolean$",
    ):
        ablation_study._validate_controller_policy(cast(bool, 1), "auto")


@pytest.mark.parametrize(
    "policy_type",
    ["AUTO", "DAY", "NIGHT", "XXautoXX", "XXdayXX", "XXnightXX"],
)
def test_ablation_controller_policy_accepts_only_canonical_policy_names(
    policy_type: str,
) -> None:
    with pytest.raises(
        AblationStudyError,
        match="^policy_type must be auto, day, or night$",
    ):
        ablation_study._validate_controller_policy(
            False,
            cast(Any, policy_type),
        )


@pytest.mark.parametrize(
    ("source_commit", "model_revision", "message"),
    [
        (
            "not-pinned",
            "a" * 40,
            "source_commit must be a pinned revision",
        ),
        (
            "b" * 40,
            "not-pinned",
            "model_revision must be a pinned revision",
        ),
    ],
)
def test_ablation_revisions_require_pinned_commits(
    source_commit: str,
    model_revision: str,
    message: str,
) -> None:
    with pytest.raises(AblationStudyError, match=f"^{message}$"):
        ablation_study._validate_revisions(source_commit, model_revision)


def test_ablation_container_validation_accepts_an_immutable_image() -> None:
    image = "registry.example/model@sha256:" + "a" * 64

    ablation_study._validate_container_arguments(image, "docker")


def test_ablation_container_validation_preserves_configuration_errors() -> None:
    with pytest.raises(
        AblationStudyError,
        match="^container_runtime requires an explicit container_image$",
    ) as error:
        ablation_study._validate_container_arguments(None, "docker")

    assert isinstance(error.value.__cause__, Grid5000ConfigurationError)
    assert str(error.value.__cause__) == (
        "container_runtime requires an explicit container_image"
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, {}),
        (
            {"eval_f1": 0.8, "note": "ok", "count": 2},
            {"eval_f1": 0.8, "note": "ok", "count": 2},
        ),
        ({"eval_f1": True, 1: 0.5, "nested": []}, {}),
    ],
)
def test_ablation_metric_mapping_keeps_only_scalar_named_metrics(
    value: object,
    expected: dict[str, object],
) -> None:
    assert ablation_study._metric_mapping(value) == expected


@pytest.mark.parametrize(
    ("key", "value", "expected"),
    [
        ("eval_f1", 0.8, True),
        ("label", "yes", True),
        ("count", 2, True),
        ("eval_f1", True, False),
        (1, 0.8, False),
        ("nested", [], False),
    ],
)
def test_ablation_metric_entry_contract(
    key: object,
    value: object,
    expected: bool,
) -> None:
    assert ablation_study._is_metric_entry(key, value) is expected


@pytest.mark.parametrize(
    "run_state",
    [SimpleNamespace(), SimpleNamespace(facts=None), SimpleNamespace(facts={})],
)
def test_completion_metrics_returns_empty_for_missing_or_invalid_facts(
    run_state: object,
) -> None:
    assert ablation_study._completion_metrics(run_state) == {}


def test_ablation_canonical_json_is_sorted_and_rejects_nan() -> None:
    assert ablation_study._canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'
    with pytest.raises(
        AblationStudyError,
        match="^ablation study specification is not JSON-safe$",
    ) as error:
        ablation_study._canonical_json({"metric": float("nan")})
    assert isinstance(error.value.__cause__, ValueError)


def test_ablation_canonical_json_uses_explicit_strict_serialization_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, object] = {}
    original_dumps = ablation_study.json.dumps

    def dumps(value: object, *args: Any, **kwargs: Any) -> str:
        observed.update(kwargs)
        return original_dumps(value, *args, **kwargs)

    monkeypatch.setattr(ablation_study.json, "dumps", dumps)

    assert ablation_study._canonical_json({"label": "café"}) == '{"label":"café"}'
    assert observed["allow_nan"] is False
    assert observed["ensure_ascii"] is False
    assert observed["separators"] == (",", ":")
    assert observed["sort_keys"] is True


def test_ablation_specification_and_fingerprint_are_deterministic() -> None:
    specification = study_specification(
        source_commit="b" * 40,
        model_revision="a" * 40,
    )

    assert specification["study_id"] == ABLATION_STUDY_ID
    assert specification["source_commit"] == "b" * 40
    assert specification["model_revision"] == "a" * 40
    assert specification["definitions"]
    assert study_specification_fingerprint(specification) == (
        study_specification_fingerprint(dict(reversed(list(specification.items()))))
    )


def test_ablation_training_config_is_isolated_and_identity_friendly() -> None:
    definition = baseline_ablation_definitions()[1]

    config = build_ablation_training_config(
        definition,
        seed=7,
        model_name_or_path="custom/model",
        model_revision="a" * 40,
        publish_to_hub=True,
        sync_trackio=True,
    )

    assert config.output_subdirectory == Path(
        f"studies/{ABLATION_STUDY_ID}/{definition.ablation_id}/models/landuse"
    )
    assert config.artifact_namespace == (
        f"studies/{ABLATION_STUDY_ID}/{definition.ablation_id}"
    )
    assert config.tracking_project == ABLATION_TRACKING_PROJECT
    assert config.run_name == f"{ABLATION_STUDY_ID}|{definition.ablation_id}|seed-7"
    assert config.max_length == 128
    assert config.publish_to_hub is True
    assert config.model_name_or_path == "custom/model"
    assert config.validation_fraction == pytest.approx(0.2)
    assert config.test_fraction == pytest.approx(0.0)
    assert config.seed == 7
    assert config.learning_rate == pytest.approx(definition.learning_rate)
    assert config.eval_strategy == "steps"
    assert config.max_steps == 1_000
    assert config.model_revision == "a" * 40
    assert config.trainable_layers == definition.trainable_layers
    assert config.class_weight_mode == definition.class_weight_mode
    assert config.sync_trackio is True

    defaults = build_ablation_training_config(
        definition,
        seed=42,
        model_revision="a" * 40,
    )
    assert defaults.publish_to_hub is True
    assert defaults.sync_trackio is True


def test_ablation_training_config_preserves_non_default_definition_values() -> None:
    definition = baseline_ablation_definitions()[3]

    config = build_ablation_training_config(
        definition,
        seed=42,
        model_revision="a" * 40,
    )

    assert config.max_length == definition.max_length
    assert config.learning_rate == pytest.approx(1e-4)
    assert config.trainable_layers == definition.trainable_layers
    assert config.class_weight_mode == definition.class_weight_mode


def test_planned_runs_adds_only_two_replicated_non_control_finalists() -> None:
    definitions = baseline_ablation_definitions()
    results = {
        definition.ablation_id: {
            "eval_f1": 0.1 + index / 100,
            "eval_macro_f1": 0.2 + index / 100,
        }
        for index, definition in enumerate(definitions)
    }
    results["a01-head-128"] = {"eval_f1": 0.99, "eval_macro_f1": 0.1}
    results["a02-head-512"] = {"eval_f1": 0.98, "eval_macro_f1": 0.1}

    runs = planned_ablation_runs(screening_results=results)

    assert len(runs) == 13
    assert [(run.ablation_id, run.seed) for run in runs[:7]] == [
        (definition.ablation_id, 42) for definition in definitions
    ]
    assert [(run.ablation_id, run.seed) for run in runs[7:]] == [
        ("a00-baseline-head-256-lr3e-4", 43),
        ("a00-baseline-head-256-lr3e-4", 44),
        ("a01-head-128", 43),
        ("a01-head-128", 44),
        ("a02-head-512", 43),
        ("a02-head-512", 44),
    ]


def test_rank_screening_results_uses_positive_f1_then_macro_f1() -> None:
    ranked = rank_screening_results(
        {
            "a00-baseline-head-256-lr3e-4": {
                "eval_f1": 0.70,
                "eval_macro_f1": 0.80,
            },
            "a01-head-128": {"eval_f1": 0.70, "eval_macro_f1": 0.81},
            "a02-head-512": {"eval_f1": 0.90, "eval_macro_f1": 0.60},
            "a03-head-lr1e-4": {"eval_f1": 0.20, "eval_macro_f1": 0.90},
        }
    )

    assert ranked == [
        "a02-head-512",
        "a01-head-128",
        "a00-baseline-head-256-lr3e-4",
        "a03-head-lr1e-4",
    ]


def test_rank_screening_results_requires_a_positive_f1() -> None:
    with pytest.raises(AblationStudyError, match="eval_f1"):
        rank_screening_results({"a01-head-128": {"eval_macro_f1": 0.8}})


def test_rank_screening_results_uses_the_metric_name_for_ties() -> None:
    ranked = rank_screening_results(
        {
            "a01-head-128": {"eval_f1": 0.5, "eval_macro_f1": 0.8},
            "a02-head-512": {"eval_f1": 0.5, "eval_macro_f1": 0.7},
        }
    )

    assert ranked == ["a01-head-128", "a02-head-512"]


def test_rank_screening_results_breaks_complete_ties_by_ablation_id() -> None:
    ranked = rank_screening_results(
        {
            "z-last": {"eval_f1": 0.5, "eval_macro_f1": 0.5},
            "a-first": {"eval_f1": 0.5, "eval_macro_f1": 0.5},
        }
    )

    assert ranked == ["a-first", "z-last"]


def test_rank_screening_results_defaults_missing_tie_metric_to_zero() -> None:
    assert ablation_study._ranking_tuple(
        "a01-head-128",
        {"eval_f1": 0.5},
        selection_metric="eval_f1",
        tie_break_metric="eval_macro_f1",
    ) == ("a01-head-128", 0.5, 0.0)


def test_rank_screening_results_rejects_a_non_numeric_tie_metric() -> None:
    with pytest.raises(
        AblationStudyError,
        match="^screening result for a01-head-128 has invalid eval_macro_f1$",
    ):
        rank_screening_results(
            {"a01-head-128": {"eval_f1": 0.5, "eval_macro_f1": "bad"}}
        )


def test_planned_runs_propagates_custom_metrics_and_uses_tie_breaking() -> None:
    definitions = baseline_ablation_definitions()
    results = {
        definition.ablation_id: {
            "custom_f1": 0.5,
            "custom_macro": 0.0,
        }
        for definition in definitions
    }
    results["a05-balanced-head"]["custom_macro"] = 0.9
    results["a06-last2-256"]["custom_macro"] = 0.8

    runs = planned_ablation_runs(
        definitions=definitions,
        screening_results=results,
        selection_metric="custom_f1",
        tie_break_metric="custom_macro",
    )

    assert [(run.ablation_id, run.seed) for run in runs[7:]] == [
        ("a00-baseline-head-256-lr3e-4", 43),
        ("a00-baseline-head-256-lr3e-4", 44),
        ("a05-balanced-head", 43),
        ("a05-balanced-head", 44),
        ("a06-last2-256", 43),
        ("a06-last2-256", 44),
    ]


def test_planned_runs_uses_the_standard_tie_break_metric_by_default() -> None:
    definitions = baseline_ablation_definitions()
    results = {
        definition.ablation_id: {
            "eval_f1": 0.5,
            "eval_macro_f1": 0.0,
        }
        for definition in definitions
    }
    results["a05-balanced-head"]["eval_macro_f1"] = 0.9
    results["a06-last2-256"]["eval_macro_f1"] = 0.8

    runs = planned_ablation_runs(
        definitions=definitions,
        screening_results=results,
    )

    assert [run.ablation_id for run in runs[7::2]] == [
        "a00-baseline-head-256-lr3e-4",
        "a05-balanced-head",
        "a06-last2-256",
    ]


def test_report_screening_results_keeps_only_completed_screening_records() -> None:
    valid_metrics = {"eval_f1": 0.8}
    records = {
        "valid": {
            "phase": "completed",
            "seed": 7,
            "ablation_id": "a01-head-128",
            "metrics": valid_metrics,
        },
        "running": {
            "phase": "running",
            "seed": 7,
            "ablation_id": "a02-head-512",
            "metrics": {"eval_f1": 0.9},
        },
        "wrong-seed": {
            "phase": "completed",
            "seed": 42,
            "ablation_id": "a03-head-lr1e-4",
            "metrics": {"eval_f1": 0.7},
        },
    }

    assert ablation_study._report_screening_results(records, 7) == {
        "a01-head-128": valid_metrics
    }


def test_report_row_preserves_provenance_metrics_and_final_model_path() -> None:
    definition = baseline_ablation_definitions()[0]
    run = ablation_study.AblationRun(definition, 7)
    records = {
        "a00-baseline-head-256-lr3e-4|seed-7": {
            "phase": "completed",
            "run_id": "controller-run",
            "source_commit": "b" * 40,
            "metrics": {"eval_f1": 0.8, "not_scalar": []},
        }
    }

    row = ablation_study._report_row(run, records, "study")

    assert row == {
        "ablation_id": "a00-baseline-head-256-lr3e-4",
        "seed": 7,
        "status": "completed",
        "run_id": "controller-run",
        "source_commit": "b" * 40,
        "metrics": {"eval_f1": 0.8},
        "model_path": "studies/study/a00-baseline-head-256-lr3e-4/"
        "run-controller-run/final/",
    }


@pytest.mark.parametrize(
    "record",
    [
        {"phase": "completed", "run_id": 123},
        {"phase": "running", "run_id": "controller-run"},
    ],
)
def test_report_row_requires_a_string_run_id_and_completed_phase_for_model_path(
    record: dict[str, object],
) -> None:
    run = ablation_study.AblationRun(baseline_ablation_definitions()[0], 7)

    row = ablation_study._report_row(
        run,
        {"a00-baseline-head-256-lr3e-4|seed-7": record},
        "study",
    )

    assert row["model_path"] is None


def test_report_runs_uses_the_supplied_protocol_for_screening_and_replication() -> None:
    definitions = baseline_ablation_definitions()[:3]
    protocol = replace(
        ablation_study.landuse_ablation_protocol(),
        definitions=definitions,
        screening_seed=7,
        replication_seeds=(8,),
        selection_metric="custom_f1",
        tie_break_metric="custom_macro",
    )
    records = {
        f"{definition.ablation_id}|seed-7": {
            "phase": "completed",
            "seed": 7,
            "ablation_id": definition.ablation_id,
            "run_id": f"screening-{index}",
            "source_commit": "b" * 40,
            "metrics": {
                "custom_f1": 0.5,
                "custom_macro": 0.9 if index == 2 else 0.8 if index == 1 else 0.1,
            },
        }
        for index, definition in enumerate(definitions)
    }
    specification = study_specification(
        source_commit="b" * 40,
        model_revision="a" * 40,
        protocol=protocol,
    )
    state = {
        "specification": specification,
        "runs": records,
    }

    rows = ablation_study._report_runs(state, protocol=protocol)

    assert len(rows) == 6
    assert [(row["ablation_id"], row["seed"]) for row in rows[:3]] == [
        (definition.ablation_id, 7) for definition in definitions
    ]
    assert [(row["ablation_id"], row["seed"]) for row in rows[3:]] == [
        ("a00-baseline-head-256-lr3e-4", 8),
        ("a02-head-512", 8),
        ("a01-head-128", 8),
    ]


def test_study_state_store_rejects_a_symlinked_root(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    root = tmp_path / "state"
    root.symlink_to(target, target_is_directory=True)

    with pytest.raises(
        AblationStudyError,
        match="^ablation study state path cannot contain symlinks$",
    ):
        AblationStudyStateStore(root).save({"phase": "running"})


def test_study_state_store_uses_the_stable_state_filename(tmp_path: Path) -> None:
    store = AblationStudyStateStore(tmp_path, study_id="study")

    assert store._path() == tmp_path / "study" / "state.json"


def test_study_state_store_round_trips_utf8_state_with_private_permissions(
    tmp_path: Path,
) -> None:
    store = AblationStudyStateStore(tmp_path / "nested" / "state")
    payload = {"phase": "running", "label": "café"}

    store.save(payload)

    path = store._path()
    assert store.load() == payload
    assert path.read_text(encoding="utf-8") == (
        '{"label": "café", "phase": "running"}\n'
    )
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700

    store.save({"phase": "completed"})
    assert store.load() == {"phase": "completed"}


def test_study_state_store_rejects_a_symlinked_temporary_file(
    tmp_path: Path,
) -> None:
    store = AblationStudyStateStore(tmp_path / "state")
    path = store._path()
    path.parent.mkdir(parents=True)
    target = tmp_path / "target"
    target.write_text("untouched", encoding="utf-8")
    path.with_name(".state.json.tmp").symlink_to(target)

    with pytest.raises(
        AblationStudyError,
        match="^ablation study state cannot be a symlink$",
    ):
        store.save({"phase": "running"})

    assert target.read_text(encoding="utf-8") == "untouched"


def test_study_state_store_rechecks_a_temporary_symlink_before_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = AblationStudyStateStore(tmp_path / "state")
    path = store._path()
    path.parent.mkdir(parents=True)
    temporary = path.with_name(".state.json.tmp")
    temporary.write_text("sentinel", encoding="utf-8")
    original_is_symlink = Path.is_symlink
    temporary_checks = 0

    def is_symlink(candidate: Path) -> bool:
        nonlocal temporary_checks
        if candidate == temporary:
            temporary_checks += 1
            return temporary_checks >= 2
        return original_is_symlink(candidate)

    monkeypatch.setattr(Path, "is_symlink", is_symlink)

    with pytest.raises(
        AblationStudyError,
        match="^ablation study state cannot be saved$",
    ) as error:
        store.save({"phase": "running"})

    assert isinstance(error.value.__cause__, AblationStudyError)
    assert str(error.value.__cause__) == "ablation study state cannot be a symlink"


def test_study_state_store_sets_the_temporary_file_private_before_replacing_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = AblationStudyStateStore(tmp_path / "state")
    temporary_modes: list[int] = []
    original_chmod = ablation_study.os.chmod

    def chmod(path: Path, mode: int) -> None:
        if Path(path).name == ".state.json.tmp":
            temporary_modes.append(mode)
        original_chmod(path, mode)

    monkeypatch.setattr(ablation_study.os, "chmod", chmod)

    store.save({"phase": "running"})

    assert temporary_modes == [0o600]


def test_study_state_store_requests_a_private_directory_at_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = AblationStudyStateStore(tmp_path / "state")
    parent = store._path().parent
    requested_modes: list[int | None] = []
    original_mkdir = Path.mkdir

    def mkdir(path: Path, *args: Any, **kwargs: Any) -> None:
        if path == parent:
            requested_modes.append(kwargs.get("mode"))
        original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", mkdir)

    store.save({"phase": "running"})

    assert requested_modes
    assert requested_modes[0] == 0o700


def test_study_state_store_defends_against_a_symlink_swap_during_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = AblationStudyStateStore(tmp_path / "state")
    path = store._path()
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(ablation_study, "_reject_symlinked_path", lambda _path: None)
    monkeypatch.setattr(Path, "is_symlink", lambda candidate: candidate == path)

    with pytest.raises(
        AblationStudyError,
        match="^ablation study state cannot be a symlink$",
    ):
        store.load()


def test_study_state_store_preserves_serialization_errors_and_cleans_temporary_file(
    tmp_path: Path,
) -> None:
    store = AblationStudyStateStore(tmp_path / "state")

    with pytest.raises(
        AblationStudyError, match="^ablation study state cannot be saved$"
    ) as error:
        store.save({"not-json": object()})

    assert isinstance(error.value.__cause__, TypeError)
    assert not store._path().with_name(".state.json.tmp").exists()


def test_study_state_store_reports_malformed_json_with_a_stable_error(
    tmp_path: Path,
) -> None:
    store = AblationStudyStateStore(tmp_path / "state")
    path = store._path()
    path.parent.mkdir(parents=True)
    path.write_text("{", encoding="utf-8")

    with pytest.raises(
        AblationStudyError,
        match="^ablation study state cannot be read$",
    ) as error:
        store.load()

    assert isinstance(error.value.__cause__, ValueError)


def test_study_state_store_rejects_non_object_json_with_a_stable_error(
    tmp_path: Path,
) -> None:
    store = AblationStudyStateStore(tmp_path / "state")
    path = store._path()
    path.parent.mkdir(parents=True)
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(
        AblationStudyError,
        match="^ablation study state must be a JSON object$",
    ):
        store.load()


def test_study_state_store_uses_explicit_utf8_for_read_and_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = AblationStudyStateStore(tmp_path / "state")
    observed: dict[str, object] = {}
    original_read_text = Path.read_text
    original_write_text = Path.write_text

    def read_text(path: Path, *args: Any, **kwargs: Any) -> str:
        if path == store._path():
            observed["read_encoding"] = kwargs.get("encoding")
        return original_read_text(path, *args, **kwargs)

    def write_text(path: Path, data: str, *args: Any, **kwargs: Any) -> int:
        if path.name == ".state.json.tmp":
            observed["write_encoding"] = kwargs.get("encoding")
        return original_write_text(path, data, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setattr(Path, "write_text", write_text)

    store.save({"label": "café"})
    assert store.load() == {"label": "café"}
    assert observed == {"read_encoding": "utf-8", "write_encoding": "utf-8"}


def test_pending_plan_records_filter_completed_runs_and_preserve_pending_defaults() -> (
    None
):
    from osm_polygon_sentence_classifier.ablation_study import AblationRun

    definitions = baseline_ablation_definitions()
    runs = [
        AblationRun(definitions[0], 42),
        AblationRun(definitions[1], 43),
        AblationRun(definitions[2], 44),
    ]
    records = {
        "a00-baseline-head-256-lr3e-4|seed-42": {"phase": "completed"},
        "a01-head-128|seed-43": {"phase": "queued"},
    }

    assert ablation_study._pending_plan_records(
        runs,
        records,
        lambda run: f"{run.ablation_id}|seed-{run.seed}",
    ) == [
        {"ablation_id": "a01-head-128", "seed": 43, "status": "queued"},
        {"ablation_id": "a02-head-512", "seed": 44, "status": "pending"},
    ]


def test_study_controller_next_pending_run_skips_completed_records(
    tmp_path: Path,
) -> None:
    from osm_polygon_sentence_classifier.ablation_study import AblationStudyController

    controller = AblationStudyController(
        source_commit="b" * 40,
        model_revision="a" * 40,
        state_root=tmp_path,
    )
    planned = controller._planned_runs(None)
    first, second = planned[:2]
    state: dict[str, object] = {
        "runs": {controller._run_key(first): {"phase": "completed"}}
    }

    assert controller._next_pending_run(state) == second


def test_study_controller_preserves_runtime_settings_and_default_callbacks(
    tmp_path: Path,
) -> None:
    from osm_polygon_sentence_classifier.ablation_study import (
        AblationStudyController,
        place_relevance_v2_ablation_protocol,
    )

    protocol = place_relevance_v2_ablation_protocol()
    controller = AblationStudyController(
        source_commit="b" * 40,
        model_revision="a" * 40,
        sites=("grenoble",),
        gpu_memory_mb=1_234,
        walltime_seconds=987,
        policy_type="day",
        max_workers=2,
        max_continuations=5,
        container_image=f"ghcr.io/example/image@sha256:{'a' * 64}",
        container_runtime="docker",
        cleanup=False,
        allow_source_commit_update=True,
        state_root=tmp_path,
        protocol=protocol,
    )

    assert controller.sites == ("grenoble",)
    assert controller.gpu_memory_mb == 1_234
    assert controller.walltime_seconds == 987
    assert controller.policy_type == "day"
    assert controller.max_workers == 2
    assert controller.max_continuations == 5
    assert controller.container_image == (f"ghcr.io/example/image@sha256:{'a' * 64}")
    assert controller.container_runtime == "docker"
    assert controller.cleanup is False
    assert controller.allow_source_commit_update is True
    assert controller.state.study_id == protocol.study_id
    assert controller.run_state_root == tmp_path / "runs"
    assert controller.fingerprint == study_specification_fingerprint(
        controller.specification
    )

    default_controller = AblationStudyController(
        source_commit="b" * 40,
        model_revision="a" * 40,
        state_root=tmp_path / "defaults",
    )
    assert default_controller.gpu_memory_mb == 8_000
    assert default_controller.max_workers == 4
    assert default_controller.max_continuations == 6
    assert default_controller.cleanup is True
    assert default_controller.allow_source_commit_update is False
    assert default_controller.publish_report({}) is None
    assert default_controller.emit("message") is None


def test_study_controller_new_state_has_the_complete_persisted_identity(
    tmp_path: Path,
) -> None:
    from osm_polygon_sentence_classifier.ablation_study import AblationStudyController

    controller = AblationStudyController(
        source_commit="b" * 40,
        model_revision="a" * 40,
        state_root=tmp_path,
    )

    state = controller._new_state()

    assert set(state) == {
        "schema_version",
        "study_id",
        "fingerprint",
        "specification",
        "phase",
        "runs",
    }
    assert state["schema_version"] == 1
    assert state["study_id"] == controller.protocol.study_id
    assert state["fingerprint"] == controller.fingerprint
    assert state["specification"] == controller.specification
    assert state["phase"] == "running"
    assert state["runs"] == {}


def test_study_controller_adopts_only_when_permission_and_compatibility_both_hold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_classifier.ablation_study import AblationStudyController

    state: dict[str, object] = {"sentinel": True}
    controller = AblationStudyController(
        source_commit="b" * 40,
        model_revision="a" * 40,
        state_root=tmp_path,
        allow_source_commit_update=False,
    )
    monkeypatch.setattr(controller, "_can_adopt_source_commit", lambda _state: True)

    assert controller._adopt_source_commit_if_needed(state) is state

    controller.allow_source_commit_update = True
    monkeypatch.setattr(controller, "_can_adopt_source_commit", lambda _state: False)
    assert controller._adopt_source_commit_if_needed(state) is state


def test_adoption_specification_matching_requires_all_three_compatibility_checks() -> (
    None
):
    stored = study_specification(source_commit="b" * 40, model_revision="a" * 40)
    current = study_specification(source_commit="d" * 40, model_revision="a" * 40)
    base_state = {
        "phase": "running",
        "fingerprint": study_specification_fingerprint(stored),
    }

    assert ablation_study._adoption_specification_matches(
        base_state,
        stored,
        current_source_commit="d" * 40,
        current_specification=current,
    )
    assert not ablation_study._adoption_specification_matches(
        {**base_state, "phase": "completed"},
        stored,
        current_source_commit="d" * 40,
        current_specification=current,
    )
    assert not ablation_study._adoption_specification_matches(
        {**base_state, "fingerprint": "wrong"},
        stored,
        current_source_commit="d" * 40,
        current_specification=current,
    )
    changed_current = study_specification(
        source_commit="d" * 40,
        model_revision="c" * 40,
    )
    assert not ablation_study._adoption_specification_matches(
        base_state,
        stored,
        current_source_commit="d" * 40,
        current_specification=changed_current,
    )


def test_adoption_source_and_history_preserve_exact_provenance_contract() -> None:
    old_source = "b" * 40
    assert (
        ablation_study._adoption_source(
            {"specification": {"source_commit": old_source}}
        )
        == old_source
    )

    for state, message in (
        ({}, "ablation study specification is invalid"),
        ({"specification": {}}, "ablation study source commit is invalid"),
        (
            {"specification": {"source_commit": 1}},
            "ablation study source commit is invalid",
        ),
    ):
        with pytest.raises(AblationStudyError) as error:
            ablation_study._adoption_source(state)
        assert str(error.value) == message

    assert ablation_study._source_commit_history({}, old_source) == [old_source]
    assert ablation_study._source_commit_history(
        {"source_commit_history": ["c" * 40, old_source, 1]}, old_source
    ) == ["c" * 40, old_source]
    assert ablation_study._source_commit_history(
        cast(Mapping[str, object], {None: ["wrong"]}),
        old_source,
    ) == [old_source]

    class MappingWithDefaultObservation(dict[str, object]):
        def __init__(self) -> None:
            super().__init__()
            self.observed: tuple[object, object] | None = None

        def get(self, key: object, default: object = None) -> object:
            self.observed = (key, default)
            return super().get(key, default)

    observed = MappingWithDefaultObservation()
    assert ablation_study._source_commit_history(observed, old_source) == [old_source]
    assert observed.observed == ("source_commit_history", [])


def test_adopting_a_source_revision_updates_records_specification_and_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_classifier.ablation_study import AblationStudyController

    old_source = "b" * 40
    new_source = "d" * 40
    old_specification = study_specification(
        source_commit=old_source,
        model_revision="a" * 40,
    )
    state: dict[str, object] = {
        "schema_version": 1,
        "study_id": ABLATION_STUDY_ID,
        "fingerprint": study_specification_fingerprint(old_specification),
        "specification": old_specification,
        "phase": "running",
        "runs": {
            "a00-baseline-head-256-lr3e-4|seed-42": {
                "phase": "completed",
            }
        },
    }
    controller = AblationStudyController(
        source_commit=new_source,
        model_revision="a" * 40,
        state_root=tmp_path,
        allow_source_commit_update=True,
    )
    saved: list[dict[str, object]] = []
    monkeypatch.setattr(controller.state, "save", lambda payload: saved.append(payload))

    adopted = controller._adopt_source_commit_if_needed(state)

    assert adopted is state
    assert saved == [state]
    assert state["source_commit_history"] == [old_source]
    assert state["specification"] == controller.specification
    assert state["fingerprint"] == controller.fingerprint
    assert cast(dict[str, object], state["runs"])[
        "a00-baseline-head-256-lr3e-4|seed-42"
    ] == {"phase": "completed", "source_commit": old_source}


def test_add_source_commit_to_records_only_fills_missing_record_provenance() -> None:
    records: dict[str, dict[str, object]] = {
        "missing": {},
        "existing": {"source_commit": "e" * 40},
    }

    ablation_study._add_source_commit_to_records(records, "b" * 40)

    assert records == {
        "missing": {"source_commit": "b" * 40},
        "existing": {"source_commit": "e" * 40},
    }


def test_controller_records_validate_keys_and_values_with_stable_errors(
    tmp_path: Path,
) -> None:
    from osm_polygon_sentence_classifier.ablation_study import AblationStudyController

    controller = AblationStudyController(
        source_commit="b" * 40,
        model_revision="a" * 40,
        state_root=tmp_path,
    )
    assert controller._records({"runs": {"ok": {"phase": "running"}}}) == {
        "ok": {"phase": "running"}
    }

    for value in ([], {1: {}}, {"bad": []}):
        with pytest.raises(AblationStudyError) as error:
            controller._records({"runs": value})
        assert str(error.value) == "ablation study run state is invalid"


def test_state_identity_requires_schema_study_and_fingerprint_together() -> None:
    expected = {"schema_version": 1, "study_id": "study", "fingerprint": "fp"}
    assert ablation_study._state_identity_matches(
        expected,
        study_id="study",
        fingerprint="fp",
    )
    assert not ablation_study._state_identity_matches(
        {**expected, "schema_version": 2}, study_id="study", fingerprint="fp"
    )
    assert not ablation_study._state_identity_matches(
        {**expected, "study_id": "other"}, study_id="study", fingerprint="fp"
    )
    assert not ablation_study._state_identity_matches(
        {**expected, "fingerprint": "other"}, study_id="study", fingerprint="fp"
    )
    assert ablation_study._adoptable_state_identity(expected, "study")
    assert not ablation_study._adoptable_state_identity({"schema_version": 2}, "study")
    assert not ablation_study._adoptable_state_identity(
        {"schema_version": 1, "study_id": "other"}, "study"
    )


def test_validate_state_rejects_an_immutable_specification_mismatch_exactly(
    tmp_path: Path,
) -> None:
    from osm_polygon_sentence_classifier.ablation_study import AblationStudyController

    controller = AblationStudyController(
        source_commit="b" * 40,
        model_revision="a" * 40,
        state_root=tmp_path,
    )
    state = controller._new_state()
    state["fingerprint"] = "wrong"

    with pytest.raises(AblationStudyError) as error:
        controller._validate_state(state)

    assert str(error.value) == (
        "existing ablation state does not match the immutable study specification"
    )


def test_can_adopt_source_commit_requires_a_different_source_and_valid_specification(
    tmp_path: Path,
) -> None:
    from osm_polygon_sentence_classifier.ablation_study import AblationStudyController

    controller = AblationStudyController(
        source_commit="b" * 40,
        model_revision="a" * 40,
        state_root=tmp_path,
        allow_source_commit_update=True,
    )
    assert not controller._can_adopt_source_commit(controller._new_state())
    assert not controller._can_adopt_source_commit(
        {"schema_version": 1, "study_id": controller.protocol.study_id}
    )


def test_validate_state_requires_permission_and_adoption_compatibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_classifier.ablation_study import AblationStudyController

    controller = AblationStudyController(
        source_commit="b" * 40,
        model_revision="a" * 40,
        state_root=tmp_path,
        allow_source_commit_update=True,
    )
    state = controller._new_state()
    state["fingerprint"] = "wrong"
    monkeypatch.setattr(controller, "_can_adopt_source_commit", lambda _state: False)

    with pytest.raises(AblationStudyError) as error:
        controller._validate_state(state)

    assert str(error.value) == (
        "existing ablation state does not match the immutable study specification"
    )


def test_has_running_records_checks_the_phase_field_exactly() -> None:
    assert ablation_study._has_running_records({"run": {"phase": "running"}})
    assert not ablation_study._has_running_records({"run": {"phase": "completed"}})
    assert not ablation_study._has_running_records({"run": {"status": "running"}})


def test_completion_metrics_extracts_only_scalar_metrics_from_completion_facts() -> (
    None
):
    run_state = SimpleNamespace(
        facts={
            "completion": {
                "metrics": {
                    "eval_f1": 0.7,
                    "eval_loss": 0.3,
                    "label": "valid",
                    "flag": True,
                    "nested": {"value": 1},
                }
            }
        }
    )

    assert ablation_study._completion_metrics(run_state) == {
        "eval_f1": 0.7,
        "eval_loss": 0.3,
        "label": "valid",
    }


@pytest.mark.parametrize("state", [{"runs": []}, {"runs": "invalid"}])
def test_validate_run_records_reports_the_stable_error(
    state: dict[str, object],
) -> None:
    with pytest.raises(AblationStudyError) as error:
        ablation_study._validate_run_records(state)

    assert str(error.value) == "ablation study run state is invalid"


def test_validate_report_specification_reports_the_stable_error() -> None:
    with pytest.raises(AblationStudyError) as error:
        ablation_study._validate_report_specification({"specification": []})

    assert str(error.value) == "ablation study specification is invalid"


def test_screening_result_requires_completed_seed_and_scalar_metrics() -> None:
    valid = {
        "phase": "completed",
        "seed": 42,
        "ablation_id": "a01-head-128",
        "metrics": {"eval_f1": 0.5},
    }
    assert ablation_study._screening_result(valid, 42) == (
        "a01-head-128",
        {"eval_f1": 0.5},
    )
    assert ablation_study._screening_result({**valid, "ablation_id": 1}, 42) is None
    assert ablation_study._screening_result({**valid, "metrics": []}, 42) is None
    assert ablation_study._screening_result({**valid, "phase": "running"}, 42) is None
    assert ablation_study._screening_result({**valid, "seed": 43}, 42) is None


def test_planned_replication_runs_reports_missing_finalists_exactly() -> None:
    definitions = baseline_ablation_definitions()
    screening_results = {
        definitions[0].ablation_id: {"eval_f1": 0.5},
        definitions[1].ablation_id: {"eval_f1": 0.4},
    }

    with pytest.raises(AblationStudyError) as error:
        ablation_study._planned_replication_runs(
            definitions,
            screening_results,
            replication_seeds=(43,),
            selection_metric="eval_f1",
            tie_break_metric="eval_macro_f1",
        )

    assert str(error.value) == "screening requires two non-control finalists"


def test_controller_limits_report_an_empty_site_list_exactly() -> None:
    with pytest.raises(AblationStudyError) as error:
        ablation_study._validate_controller_limits(
            sites=(),
            gpu_memory_mb=1,
            walltime_seconds=1,
            max_workers=1,
            max_continuations=1,
        )

    assert str(error.value) == "at least one Grid'5000 site is required"


def test_study_controller_forwards_all_autonomous_run_settings(
    tmp_path: Path,
) -> None:
    from osm_polygon_sentence_classifier.ablation_study import (
        AblationRun,
        AblationStudyController,
    )

    controller = AblationStudyController(
        source_commit="b" * 40,
        model_revision="a" * 40,
        sites=("grenoble", "nancy"),
        gpu_memory_mb=1_234,
        walltime_seconds=987,
        policy_type="day",
        max_workers=2,
        max_continuations=5,
        container_image=f"ghcr.io/example/image@sha256:{'a' * 64}",
        container_runtime="docker",
        cleanup=False,
        state_root=tmp_path,
    )

    config = controller._autonomous_config(
        AblationRun(controller.protocol.definitions[0], seed=7)
    )

    assert config.sites == ("grenoble", "nancy")
    assert config.requirements.gpu_memory_mb == 1_234
    assert config.walltime_seconds == 987
    assert config.policy_type == "day"
    assert config.max_workers == 2
    assert config.max_continuations == 5
    assert config.container_image == (f"ghcr.io/example/image@sha256:{'a' * 64}")
    assert config.container_runtime == "docker"
    assert config.cleanup is False


def test_study_controller_planned_runs_forwards_the_protocol_selection_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_classifier.ablation_study import AblationStudyController

    protocol = replace(
        ablation_study.landuse_ablation_protocol(),
        screening_seed=7,
        replication_seeds=(8, 9),
        selection_metric="custom_f1",
        tie_break_metric="custom_macro",
    )
    controller = AblationStudyController(
        source_commit="b" * 40,
        model_revision="a" * 40,
        state_root=tmp_path,
        protocol=protocol,
    )
    captured: dict[str, object] = {}
    sentinel: list[object] = []

    def planned(**kwargs: object) -> list[object]:
        captured.update(kwargs)
        return sentinel

    monkeypatch.setattr(ablation_study, "planned_ablation_runs", planned)
    screening = {"a00": {"custom_f1": 0.5}}

    assert controller._planned_runs(screening) is sentinel
    assert captured == {
        "definitions": protocol.definitions,
        "screening_results": screening,
        "screening_seed": 7,
        "replication_seeds": (8, 9),
        "selection_metric": "custom_f1",
        "tie_break_metric": "custom_macro",
    }


def test_study_controller_run_config_preserves_custom_model_identity(
    tmp_path: Path,
) -> None:
    from osm_polygon_sentence_classifier.ablation_study import (
        AblationRun,
        AblationStudyController,
    )

    controller = AblationStudyController(
        source_commit="b" * 40,
        model_revision="a" * 40,
        model_name_or_path="custom/model",
        state_root=tmp_path,
    )
    config, identity = controller._run_config(
        AblationRun(controller.protocol.definitions[0], seed=42)
    )

    assert config.model_name_or_path == "custom/model"
    assert identity.model_name_or_path == "custom/model"


def test_study_controller_run_controller_forwards_state_root_and_emitter(
    tmp_path: Path,
) -> None:
    from osm_polygon_sentence_classifier.ablation_study import AblationStudyController

    captured: dict[str, object] = {}
    sentinel = object()

    class FakeRunController:
        def __init__(self, config: object, **kwargs: object) -> None:
            captured["config"] = config
            captured.update(kwargs)

        def run(self) -> object:
            return sentinel

    controller = AblationStudyController(
        source_commit="b" * 40,
        model_revision="a" * 40,
        state_root=tmp_path,
        run_controller_factory=FakeRunController,
    )
    config = controller._autonomous_config(
        ablation_study.AblationRun(controller.protocol.definitions[0], seed=42)
    )

    assert controller._run_controller(config) is sentinel
    assert captured == {
        "config": config,
        "state_root": tmp_path / "runs",
        "emit": controller.emit,
    }


def test_study_controller_load_run_state_persists_a_new_state_before_returning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_classifier.ablation_study import AblationStudyController

    controller = AblationStudyController(
        source_commit="b" * 40,
        model_revision="a" * 40,
        state_root=tmp_path,
    )
    state = controller._new_state()
    saved: list[dict[str, object]] = []
    monkeypatch.setattr(controller.state, "load", lambda: None)
    monkeypatch.setattr(controller, "_new_state", lambda: state)
    monkeypatch.setattr(controller.state, "save", lambda payload: saved.append(payload))

    assert controller._load_run_state() is state
    assert saved == [state]


def test_study_controller_complete_study_persists_publishes_and_emits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_classifier.ablation_study import AblationStudyController

    messages: list[str] = []
    published: list[dict[str, object]] = []

    def publish_report(payload: Mapping[str, object]) -> None:
        published.append(dict(payload))

    saved: list[dict[str, object]] = []
    controller = AblationStudyController(
        source_commit="b" * 40,
        model_revision="a" * 40,
        state_root=tmp_path,
        emit=messages.append,
        publish_report=publish_report,
    )
    monkeypatch.setattr(controller.state, "save", lambda payload: saved.append(payload))
    state = controller._new_state()

    assert controller._complete_study(state) is state
    assert state["phase"] == "completed"
    assert saved == [state]
    assert published == [state]
    assert messages == [f"study {controller.protocol.study_id}: completed"]


def test_study_controller_run_returns_completed_state_without_planning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_classifier.ablation_study import AblationStudyController

    controller = AblationStudyController(
        source_commit="b" * 40,
        model_revision="a" * 40,
        state_root=tmp_path,
    )
    state: dict[str, object] = {"phase": "completed", "runs": {}}
    monkeypatch.setattr(controller, "_load_run_state", lambda: state)

    def unexpected_next_pending(_state: dict[str, object]) -> Any:
        raise AssertionError("completed study should not plan another run")

    monkeypatch.setattr(controller, "_next_pending_run", unexpected_next_pending)

    assert controller.run() is state


def test_ablation_plan_rejects_a_malformed_persisted_run_record(
    tmp_path: Path,
) -> None:
    from osm_polygon_sentence_classifier.ablation_study import AblationStudyController

    controller = AblationStudyController(
        source_commit="b" * 40,
        model_revision="a" * 40,
        state_root=tmp_path,
    )
    state = controller._new_state()
    state["runs"] = {"broken": []}
    controller.state.save(state)

    with pytest.raises(AblationStudyError, match="run state is invalid"):
        controller.plan()


def test_study_controller_runs_screening_then_replicates_finalists(
    tmp_path: Path,
) -> None:
    from osm_polygon_sentence_classifier.ablation_study import AblationStudyController

    calls: list[tuple[str, int]] = []

    class FakeRunController:
        def __init__(self, config, **kwargs) -> None:
            del kwargs
            self.config = config

        def run(self):
            training = self.config.identity.training_config
            ablation_id = str(training["run_name"]).split("|")[1]
            seed = int(str(training["run_name"]).rsplit("-", 1)[1])
            calls.append((ablation_id, seed))
            score = {
                "a01-head-128": 0.99,
                "a02-head-512": 0.98,
            }.get(ablation_id, 0.1)
            return SimpleNamespace(
                run_id=self.config.identity.run_id,
                phase=RunPhase.COMPLETED,
                facts={
                    "completion": {
                        "metrics": {
                            "eval_f1": score,
                            "eval_macro_f1": score,
                        }
                    }
                },
            )

    controller = AblationStudyController(
        source_commit="b" * 40,
        model_revision="a" * 40,
        state_root=tmp_path,
        run_controller_factory=FakeRunController,
        publish_report=lambda _state: None,
    )

    state = controller.run()

    assert state["phase"] == "completed"
    assert len(calls) == 13
    assert calls[:7] == [
        (definition.ablation_id, 42) for definition in baseline_ablation_definitions()
    ]
    assert calls[7:] == [
        ("a00-baseline-head-256-lr3e-4", 43),
        ("a00-baseline-head-256-lr3e-4", 44),
        ("a01-head-128", 43),
        ("a01-head-128", 44),
        ("a02-head-512", 43),
        ("a02-head-512", 44),
    ]


def test_study_controller_persists_run_identity_and_start_event(
    tmp_path: Path,
) -> None:
    from osm_polygon_sentence_classifier.ablation_study import (
        AblationRun,
        AblationStudyController,
    )

    messages: list[str] = []
    controller = AblationStudyController(
        source_commit="b" * 40,
        model_revision="a" * 40,
        state_root=tmp_path,
        emit=messages.append,
    )
    state = controller._new_state()
    records: dict[str, dict[str, object]] = {}
    pending = AblationRun(controller.protocol.definitions[0], seed=42)
    key = controller._run_key(pending)
    config = controller._autonomous_config(pending)

    controller._record_run_started(
        state,
        records,
        key,
        pending,
        config.identity.run_id,
    )

    assert records == {
        key: {
            "ablation_id": pending.ablation_id,
            "seed": 42,
            "source_commit": "b" * 40,
            "run_id": config.identity.run_id,
            "phase": "running",
        }
    }
    assert state["runs"] == records
    assert messages == [
        f"study {controller.protocol.study_id}: running "
        f"{pending.ablation_id} seed=42 run={config.identity.run_id}"
    ]
    assert controller.state.load() == state


def test_study_controller_persists_failure_before_raising_a_stable_error(
    tmp_path: Path,
) -> None:
    from osm_polygon_sentence_classifier.ablation_study import (
        AblationRun,
        AblationStudyController,
    )

    class FailingRunController:
        def __init__(self, config, **kwargs) -> None:
            del config, kwargs

        def run(self):
            raise RuntimeError("worker exploded")

    controller = AblationStudyController(
        source_commit="b" * 40,
        model_revision="a" * 40,
        state_root=tmp_path,
        run_controller_factory=FailingRunController,
    )
    state = controller._new_state()
    pending = AblationRun(controller.protocol.definitions[0], seed=42)

    with pytest.raises(
        AblationStudyError,
        match="^ablation a00-baseline-head-256-lr3e-4 seed 42 failed$",
    ) as error:
        controller._execute_pending_run(state, pending)

    assert isinstance(error.value.__cause__, RuntimeError)
    records = cast(dict[str, dict[str, object]], state["runs"])
    record = records[controller._run_key(pending)]
    assert record == {
        "ablation_id": pending.ablation_id,
        "seed": 42,
        "source_commit": "b" * 40,
        "run_id": controller._autonomous_config(pending).identity.run_id,
        "phase": "failed",
        "error": "worker exploded",
    }
    assert controller.state.load() == state


def test_study_controller_persists_failure_records_under_runs_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_classifier.ablation_study import AblationStudyController

    controller = AblationStudyController(
        source_commit="b" * 40,
        model_revision="a" * 40,
        state_root=tmp_path,
    )
    key = "a00-baseline-head-256-lr3e-4|seed-42"
    records: dict[str, dict[str, object]] = {key: {"phase": "running"}}
    state: dict[str, object] = {"runs": {}}
    saved: list[dict[str, object]] = []

    def save(payload: Mapping[str, object]) -> None:
        saved.append(dict(payload))

    monkeypatch.setattr(controller.state, "save", save)

    controller._record_run_failure(
        state,
        records,
        key,
        RuntimeError("worker exploded"),
    )

    assert state["runs"] is records
    assert saved == [state]


def test_study_controller_persists_phase_records_under_runs_key(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from osm_polygon_sentence_classifier.ablation_study import AblationStudyController

    controller = AblationStudyController(
        source_commit="b" * 40,
        model_revision="a" * 40,
        state_root=tmp_path,
    )
    key = "a00-baseline-head-256-lr3e-4|seed-42"
    records: dict[str, dict[str, object]] = {key: {"phase": "running"}}
    state: dict[str, object] = {"runs": {}}
    saved: list[dict[str, object]] = []

    def save(payload: Mapping[str, object]) -> None:
        saved.append(dict(payload))

    monkeypatch.setattr(controller.state, "save", save)

    controller._record_run_phase(
        state,
        records,
        key,
        SimpleNamespace(phase=RunPhase.FAILED),
    )

    assert state["runs"] is records
    assert saved == [state]


@pytest.mark.parametrize(
    "run_state",
    [SimpleNamespace(phase=RunPhase.FAILED), SimpleNamespace()],
)
def test_study_controller_persists_noncompleted_phase_before_raising(
    tmp_path: Path,
    run_state: SimpleNamespace,
) -> None:
    from osm_polygon_sentence_classifier.ablation_study import (
        AblationRun,
        AblationStudyController,
    )

    class IncompleteRunController:
        def __init__(self, config, **kwargs) -> None:
            del config, kwargs

        def run(self):
            return run_state

    controller = AblationStudyController(
        source_commit="b" * 40,
        model_revision="a" * 40,
        state_root=tmp_path,
        run_controller_factory=IncompleteRunController,
    )
    state = controller._new_state()
    pending = AblationRun(controller.protocol.definitions[0], seed=42)
    expected_phase = getattr(run_state, "phase", "unknown")

    with pytest.raises(
        AblationStudyError,
        match=f"^ablation {pending.ablation_id} seed 42 ended in {expected_phase}$",
    ):
        controller._execute_pending_run(state, pending)

    records = cast(dict[str, dict[str, object]], state["runs"])
    record = records[controller._run_key(pending)]
    assert record["phase"] == expected_phase
    assert controller.state.load() == state


def test_study_controller_publishes_the_completed_record_state(
    tmp_path: Path,
) -> None:
    from osm_polygon_sentence_classifier.ablation_study import (
        AblationRun,
        AblationStudyController,
    )

    published: list[dict[str, object] | None] = []

    class CompletedRunController:
        def __init__(self, config, **kwargs) -> None:
            self.config = config
            del kwargs

        def run(self):
            return SimpleNamespace(
                run_id=self.config.identity.run_id,
                phase=RunPhase.COMPLETED,
                facts={"completion": {"metrics": {"eval_f1": 0.5}}},
            )

    def publish(state: Mapping[str, object]) -> None:
        published.append(
            {
                **state,
                "runs": dict(cast(dict[str, object], state["runs"])),
            }
        )

    controller = AblationStudyController(
        source_commit="b" * 40,
        model_revision="a" * 40,
        state_root=tmp_path,
        run_controller_factory=CompletedRunController,
        publish_report=publish,
    )

    state = controller._new_state()
    pending = AblationRun(controller.protocol.definitions[0], seed=42)
    controller._execute_pending_run(state, pending)

    assert published
    first_publication = published[0]
    assert first_publication is not None
    assert set(first_publication) == {
        "schema_version",
        "study_id",
        "fingerprint",
        "specification",
        "phase",
        "runs",
    }
    first_records = cast(dict[str, dict[str, object]], first_publication["runs"])
    first_record = first_records["a00-baseline-head-256-lr3e-4|seed-42"]
    assert first_record["phase"] == "completed"
    assert first_record["metrics"] == {"eval_f1": 0.5}
    assert controller.state.load() == state


def test_study_controller_plan_uses_completed_screening_results_for_replications(
    tmp_path: Path,
) -> None:
    from osm_polygon_sentence_classifier.ablation_study import AblationStudyController

    controller = AblationStudyController(
        source_commit="b" * 40,
        model_revision="a" * 40,
        state_root=tmp_path,
    )
    state = controller._new_state()
    state["runs"] = {
        f"{definition.ablation_id}|seed-42": {
            "ablation_id": definition.ablation_id,
            "seed": 42,
            "run_id": "c" * 20,
            "phase": "completed",
            "metrics": {"eval_f1": 0.5, "eval_macro_f1": 0.5},
        }
        for definition in controller.protocol.definitions
    }
    controller.state.save(state)

    plan = controller.plan()

    assert plan["total_runs"] == 13
    assert plan["fingerprint"] == controller.fingerprint
    assert plan["trackio_space_id"] == ablation_study.TRACKIO_STATIC_SPACE_ID
    assert plan["trackio_bucket_id"] == ablation_study.TRACKIO_BUCKET_ID
    assert plan["next_runs"] == [
        {
            "ablation_id": "a00-baseline-head-256-lr3e-4",
            "seed": 43,
            "status": "pending",
        },
        {
            "ablation_id": "a00-baseline-head-256-lr3e-4",
            "seed": 44,
            "status": "pending",
        },
        {
            "ablation_id": "a01-head-128",
            "seed": 43,
            "status": "pending",
        },
        {
            "ablation_id": "a01-head-128",
            "seed": 44,
            "status": "pending",
        },
        {
            "ablation_id": "a02-head-512",
            "seed": 43,
            "status": "pending",
        },
        {
            "ablation_id": "a02-head-512",
            "seed": 44,
            "status": "pending",
        },
    ]


def test_study_controller_reuses_completed_state_without_resubmitting(
    tmp_path: Path,
) -> None:
    from osm_polygon_sentence_classifier.ablation_study import AblationStudyController

    calls = 0

    class FakeRunController:
        def __init__(self, config, **kwargs) -> None:
            del config, kwargs

        def run(self):
            nonlocal calls
            calls += 1
            return SimpleNamespace(
                run_id="a" * 20,
                phase=RunPhase.COMPLETED,
                facts={"completion": {"metrics": {"eval_f1": 0.5}}},
            )

    kwargs: dict[str, Any] = {
        "source_commit": "b" * 40,
        "model_revision": "a" * 40,
        "state_root": tmp_path,
        "run_controller_factory": FakeRunController,
        "publish_report": lambda _state: None,
    }
    first = AblationStudyController(**kwargs)
    first.run()
    first_call_count = calls

    second = AblationStudyController(**kwargs)
    state = second.run()

    assert state["phase"] == "completed"
    assert calls == first_call_count


def test_incomplete_study_can_explicitly_adopt_a_new_source_revision(
    tmp_path: Path,
) -> None:
    from osm_polygon_sentence_classifier.ablation_study import (
        AblationStudyController,
        study_specification,
        study_specification_fingerprint,
    )

    old_specification = study_specification(
        source_commit="b" * 40,
        model_revision="a" * 40,
    )
    completed_runs = {
        f"{definition.ablation_id}|seed-42": {
            "ablation_id": definition.ablation_id,
            "seed": 42,
            "run_id": "c" * 20,
            "phase": "completed",
            "metrics": {"eval_f1": 0.5, "eval_macro_f1": 0.5},
        }
        for definition in baseline_ablation_definitions()[:-1]
    }
    AblationStudyStateStore(tmp_path).save(
        {
            "schema_version": 1,
            "study_id": ABLATION_STUDY_ID,
            "fingerprint": study_specification_fingerprint(old_specification),
            "specification": old_specification,
            "phase": "running",
            "runs": completed_runs,
        }
    )
    observed_source_commits: list[str] = []

    class FakeRunController:
        def __init__(self, config, **kwargs) -> None:
            del kwargs
            observed_source_commits.append(config.identity.source_commit)
            self.config = config

        def run(self):
            return SimpleNamespace(
                run_id=self.config.identity.run_id,
                phase=RunPhase.COMPLETED,
                facts={
                    "completion": {
                        "metrics": {
                            "eval_f1": 0.5,
                            "eval_macro_f1": 0.5,
                        }
                    }
                },
            )

    new_source_commit = "d" * 40
    controller = AblationStudyController(
        source_commit=new_source_commit,
        model_revision="a" * 40,
        state_root=tmp_path,
        allow_source_commit_update=True,
        run_controller_factory=FakeRunController,
        publish_report=lambda _state: None,
    )

    state = controller.run()

    assert observed_source_commits
    assert observed_source_commits[0] == new_source_commit
    specification = cast(dict[str, object], state["specification"])
    assert specification["source_commit"] == new_source_commit
    assert state["source_commit_history"] == ["b" * 40]


def test_study_documents_are_public_and_include_clear_run_names() -> None:
    from osm_polygon_sentence_classifier.ablation_study import (
        study_specification,
        study_specification_fingerprint,
    )

    specification = study_specification(
        source_commit="b" * 40,
        model_revision="a" * 40,
    )
    state = {
        "study_id": ABLATION_STUDY_ID,
        "fingerprint": study_specification_fingerprint(specification),
        "specification": specification,
        "phase": "running",
        "runs": {
            "a01-head-128|seed-42": {
                "ablation_id": "a01-head-128",
                "seed": 42,
                "run_id": "c" * 20,
                "phase": "completed",
                "metrics": {"eval_f1": 0.8, "eval_macro_f1": 0.7},
            }
        },
    }

    documents = render_study_documents(state)

    assert set(documents) == {
        "README.md",
        "studies/landuse-v1/README.md",
        "studies/landuse-v1/results.json",
        "studies/landuse-v1/study.json",
    }
    assert "a01-head-128" in documents["studies/landuse-v1/README.md"]
    assert (
        "landuse-v1|a01-head-128|seed-42" in documents["studies/landuse-v1/README.md"]
    )
    assert "Validation metrics" in documents["studies/landuse-v1/README.md"]
    assert "Maximum length" in documents["studies/landuse-v1/README.md"]
    assert "eval_f1" in documents["studies/landuse-v1/results.json"]
    assert "Grid'5000" in documents["studies/landuse-v1/README.md"]


@pytest.mark.parametrize(
    "state",
    [
        {},
        {"specification": {}, "fingerprint": None},
    ],
)
def test_render_study_documents_rejects_incomplete_public_state(
    state: dict[str, object],
) -> None:
    with pytest.raises(
        AblationStudyError,
        match="^ablation study state lacks its public specification$",
    ):
        render_study_documents(state)


def test_document_options_keep_landuse_and_worldwide_v2_lanes_separate() -> None:
    assert ablation_study._document_options(
        ablation_study.landuse_ablation_protocol()
    ) == (ablation_study.TRACKIO_STATIC_SPACE_ID, None, True)
    assert ablation_study._document_options(
        ablation_study.place_relevance_v2_ablation_protocol()
    ) == (
        ablation_study.V2_TRACKIO_STATIC_SPACE_ID,
        "worldwide V2 place-relevance sentence-classification task.",
        False,
    )


def test_worldwide_v2_ablation_protocol_is_separate_and_test_aware() -> None:
    from osm_polygon_sentence_classifier.ablation_study import (
        PLACE_RELEVANCE_V2_ABLATION_STUDY_ID,
        place_relevance_v2_ablation_protocol,
    )

    protocol = place_relevance_v2_ablation_protocol()

    assert protocol.study_id == PLACE_RELEVANCE_V2_ABLATION_STUDY_ID
    assert protocol.task_name == "place-relevance-v2"
    assert protocol.tracking_project == "place-relevance-v2-ablations"
    assert protocol.validation_fraction == pytest.approx(0.1)
    assert protocol.test_fraction == pytest.approx(0.1)
    assert protocol.eval_strategy == "epoch"
    assert protocol.max_steps == 17_661
    assert len(protocol.definitions) == 7
    assert protocol.title == "Worldwide V2 place-relevance ablation study"
    assert protocol.introduction == (
        "This study measures controlled changes to the worldwide V2 "
        "place-relevance sentence classifier."
    )
    assert protocol.evaluation_note == (
        "Validation metrics select finalists; the held-out test set is "
        "evaluated once per run after training and is not used for selection."
    )


def test_worldwide_v2_ablation_config_isolated_from_the_baseline() -> None:
    from osm_polygon_sentence_classifier.ablation_study import (
        build_ablation_training_config,
        place_relevance_v2_ablation_protocol,
    )

    protocol = place_relevance_v2_ablation_protocol()
    definition = protocol.definitions[1]
    config = build_ablation_training_config(
        definition,
        seed=42,
        model_revision="a" * 40,
        protocol=protocol,
        publish_to_hub=True,
        sync_trackio=True,
    )

    assert config.run_name == ("place-relevance-v2-ablations|a01-head-128|seed-42")
    assert config.output_subdirectory.as_posix() == (
        "studies/place-relevance-v2-ablations/a01-head-128/models/place-relevance-v2"
    )
    assert config.artifact_namespace == (
        "studies/place-relevance-v2-ablations/a01-head-128"
    )
    assert config.max_length == 128
    assert config.validation_fraction == pytest.approx(protocol.validation_fraction)
    assert config.test_fraction == pytest.approx(0.1)
    assert config.eval_strategy == "epoch"
    assert config.max_steps == 17_661
    assert config.tracking_project == "place-relevance-v2-ablations"


def test_worldwide_v2_controller_keeps_the_ablation_lane_separate(
    tmp_path: Path,
) -> None:
    from osm_polygon_sentence_classifier.ablation_study import (
        AblationStudyController,
        place_relevance_v2_ablation_protocol,
    )

    protocol = place_relevance_v2_ablation_protocol()
    observed: list[tuple[str, str, str]] = []

    class FakeRunController:
        def __init__(self, config, **kwargs) -> None:
            del kwargs
            self.config = config

        def run(self):
            identity = self.config.identity
            training = identity.training_config
            observed.append(
                (
                    identity.task_name,
                    str(training["tracking_project"]),
                    str(training["run_name"]),
                )
            )
            ablation_id = str(training["run_name"]).split("|")[1]
            score = 0.99 if ablation_id == "a01-head-128" else 0.1
            return SimpleNamespace(
                phase=RunPhase.COMPLETED,
                facts={
                    "completion": {
                        "metrics": {
                            "eval_f1": score,
                            "eval_macro_f1": score,
                        }
                    }
                },
            )

    controller = AblationStudyController(
        source_commit="b" * 40,
        model_revision="a" * 40,
        protocol=protocol,
        state_root=tmp_path,
        run_controller_factory=FakeRunController,
        publish_report=lambda _state: None,
    )

    state = controller.run()

    assert state["phase"] == "completed"
    assert len(observed) == 13
    assert observed[0] == (
        "place-relevance-v2",
        "place-relevance-v2-ablations",
        "place-relevance-v2-ablations|a00-baseline-head-256-lr3e-4|seed-42",
    )


def test_worldwide_v2_report_uses_its_own_public_namespace() -> None:
    from osm_polygon_sentence_classifier.ablation_study import (
        place_relevance_v2_ablation_protocol,
        render_study_documents,
        study_specification,
        study_specification_fingerprint,
    )

    protocol = place_relevance_v2_ablation_protocol()
    specification = study_specification(
        source_commit="b" * 40,
        model_revision="a" * 40,
        protocol=protocol,
    )
    state = {
        "study_id": protocol.study_id,
        "fingerprint": study_specification_fingerprint(specification),
        "specification": specification,
        "phase": "running",
        "runs": {},
    }

    documents = render_study_documents(state, protocol=protocol)

    assert set(documents) == {
        "studies/place-relevance-v2-ablations/README.md",
        "studies/place-relevance-v2-ablations/study.json",
        "studies/place-relevance-v2-ablations/results.json",
    }
    assert (
        "Worldwide V2 place-relevance"
        in documents["studies/place-relevance-v2-ablations/README.md"]
    )
    assert (
        "held-out test set"
        in documents["studies/place-relevance-v2-ablations/README.md"]
    )
    v2_readme = documents["studies/place-relevance-v2-ablations/README.md"]
    assert (
        "https://huggingface.co/spaces/NoeFlandre/"
        "osm-polygon-sentence-classifier-v2-trackio"
    ) in v2_readme
    assert (
        "https://huggingface.co/spaces/NoeFlandre/"
        "osm-polygon-sentence-classifier-trackio"
    ) not in v2_readme


def test_render_study_documents_forwards_custom_public_protocol_text() -> None:
    from osm_polygon_sentence_classifier.ablation_study import (
        place_relevance_v2_ablation_protocol,
    )

    protocol = replace(
        place_relevance_v2_ablation_protocol(),
        title="Unique V2 title",
        introduction="Unique V2 introduction.",
        evaluation_note="Unique V2 evaluation note.",
    )
    specification = study_specification(
        source_commit="b" * 40,
        model_revision="a" * 40,
        protocol=protocol,
    )
    state = {
        "specification": specification,
        "fingerprint": study_specification_fingerprint(specification),
        "phase": "running",
        "runs": {},
    }

    documents = render_study_documents(state, protocol=protocol)
    readme = documents["studies/place-relevance-v2-ablations/README.md"]

    assert "# Unique V2 title" in readme
    assert "Unique V2 introduction." in readme
    assert "Unique V2 evaluation note." in readme


def test_render_study_documents_forwards_all_public_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = ablation_study.landuse_ablation_protocol()
    specification = study_specification(
        source_commit="b" * 40,
        model_revision="a" * 40,
        protocol=protocol,
    )
    state = {
        "specification": specification,
        "fingerprint": study_specification_fingerprint(specification),
    }
    observed: dict[str, object] = {}

    def document_options(_protocol: object) -> tuple[str, str, bool]:
        return "space", "root scope", True

    def report_runs(
        report_state: object,
        *,
        protocol: object,
    ) -> list[dict[str, object]]:
        observed["report_state"] = report_state
        observed["report_protocol"] = protocol
        return [{"row": 1}]

    def render_documents(
        render_state: object,
        **kwargs: object,
    ) -> dict[str, str]:
        observed["render_state"] = render_state
        observed.update(kwargs)
        return {"probe": "ok"}

    monkeypatch.setattr(ablation_study, "_document_options", document_options)
    monkeypatch.setattr(ablation_study, "_report_runs", report_runs)
    monkeypatch.setattr(ablation_study, "render_public_documents", render_documents)

    assert ablation_study.render_study_documents(state, protocol=protocol) == {
        "probe": "ok"
    }
    assert observed == {
        "report_state": state,
        "report_protocol": protocol,
        "render_state": state,
        "rows": [{"row": 1}],
        "study_id": protocol.study_id,
        "tracking_space_id": "space",
        "study_title": protocol.title,
        "study_introduction": protocol.introduction,
        "evaluation_note": protocol.evaluation_note,
        "root_scope": "root scope",
        "include_root_readme": True,
    }
