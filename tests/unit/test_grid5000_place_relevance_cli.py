import json

from osm_polygon_sentence_classifier import grid5000_cli, grid5000_place_relevance_cli
from osm_polygon_sentence_classifier.grid5000 import Grid5000RunIdentity
from osm_polygon_sentence_classifier.grid5000_state import AutonomousRunState

SOURCE_COMMIT = "a" * 40
MODEL_REVISION = "b" * 40


def test_worldwide_v2_plan_uses_the_separate_task_and_epoch_evaluation(capsys) -> None:
    exit_code = grid5000_place_relevance_cli.main(
        [
            "run",
            "--source-commit",
            SOURCE_COMMIT,
            "--model-revision",
            MODEL_REVISION,
            "--publish",
            "--sync-trackio",
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["identity"]["task_name"] == "place-relevance-v2"
    config = payload["identity"]["training_config"]
    assert config["eval_strategy"] == "epoch"
    assert config["validation_fraction"] == 0.1
    assert config["test_fraction"] == 0.1
    assert config["trainable_layers"] == "head"
    assert config["tracking_project"] == "place-relevance-v2"
    assert payload["max_continuations"] == 40


def test_worldwide_v2_plan_carries_the_task_to_the_worker(capsys) -> None:
    exit_code = grid5000_place_relevance_cli.main(
        [
            "plan",
            "--site",
            "nancy",
            "--source-commit",
            SOURCE_COMMIT,
            "--model-revision",
            MODEL_REVISION,
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert "--task-name place-relevance-v2" in payload["scheduler_command"][-1]


def test_worldwide_v2_resume_rejects_a_landuse_run(monkeypatch, capsys) -> None:
    identity = Grid5000RunIdentity(
        source_commit=SOURCE_COMMIT,
        dataset_revision="d" * 40,
        model_name_or_path="jhu-clsp/mmBERT-small",
        model_revision=MODEL_REVISION,
        training_config={
            "model_name_or_path": "jhu-clsp/mmBERT-small",
            "model_revision": MODEL_REVISION,
            "publish_to_hub": False,
            "sync_trackio": False,
        },
    )
    state = AutonomousRunState(
        run_id=identity.run_id,
        phase="failed",
        identity=identity.canonical_payload,
    )
    monkeypatch.setattr(
        grid5000_cli.AutonomousStateStore,
        "load",
        lambda _self, _run_id: state,
    )

    exit_code = grid5000_place_relevance_cli.main(
        ["resume", "--run-id", identity.run_id]
    )

    assert exit_code == 2
    assert "task does not match" in capsys.readouterr().err
