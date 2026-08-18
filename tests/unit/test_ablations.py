from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

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
)
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


def test_ablation_training_config_is_isolated_and_identity_friendly() -> None:
    definition = baseline_ablation_definitions()[1]

    config = build_ablation_training_config(
        definition,
        seed=42,
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
    assert config.run_name == f"{ABLATION_STUDY_ID}|{definition.ablation_id}|seed-42"
    assert config.max_length == 128
    assert config.publish_to_hub is True


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


def test_study_state_store_rejects_a_symlinked_root(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    root = tmp_path / "state"
    root.symlink_to(target, target_is_directory=True)

    with pytest.raises(AblationStudyError, match="symlink"):
        AblationStudyStateStore(root).save({"phase": "running"})


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
    assert config.test_fraction == pytest.approx(0.1)
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
