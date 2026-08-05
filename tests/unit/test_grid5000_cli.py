import json

from osm_polygon_sentence_classifier import grid5000_cli
from osm_polygon_sentence_classifier.grid5000 import Grid5000Plan, Grid5000Submission

SOURCE_COMMIT = "a" * 40
MODEL_REVISION = "b" * 40


def _arguments(*extra: str) -> list[str]:
    return [
        "--site",
        "nancy",
        "--source-commit",
        SOURCE_COMMIT,
        "--model-revision",
        MODEL_REVISION,
        *extra,
    ]


def test_plan_command_prints_a_reproducible_plan_without_an_operator(
    monkeypatch, capsys
) -> None:
    def unexpected_operator(*args: object, **kwargs: object) -> None:
        raise AssertionError("plan mode must not construct an executing operator")

    monkeypatch.setattr(grid5000_cli, "Grid5000Operator", unexpected_operator)

    exit_code = grid5000_cli.main(["plan", *_arguments()])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"]
    assert "usagepolicycheck" not in payload["submission_command"]
    assert "oarsub" in payload["submission_command"]
    assert payload["identity"]["model_name_or_path"] == "jhu-clsp/mmBERT-small"


def test_submit_without_execute_calls_only_the_plan_path(monkeypatch, capsys) -> None:
    calls: list[bool] = []

    class FakeOperator:
        def __init__(self, plan: Grid5000Plan) -> None:
            self.plan = plan

        def submit(self, *, execute: bool = False) -> Grid5000Submission:
            calls.append(execute)
            return Grid5000Submission(plan=self.plan, executed=execute)

    monkeypatch.setattr(grid5000_cli, "Grid5000Operator", FakeOperator)

    exit_code = grid5000_cli.main(["submit", *_arguments()])

    assert exit_code == 0
    assert calls == [False]
    assert json.loads(capsys.readouterr().out)["executed"] is False


def test_submit_execute_is_the_only_explicit_execution_gate(
    monkeypatch, capsys
) -> None:
    calls: list[bool] = []

    class FakeOperator:
        def __init__(self, plan: Grid5000Plan) -> None:
            self.plan = plan

        def submit(self, *, execute: bool = False) -> Grid5000Submission:
            calls.append(execute)
            return Grid5000Submission(plan=self.plan, executed=execute, job_id=7)

    monkeypatch.setattr(grid5000_cli, "Grid5000Operator", FakeOperator)

    exit_code = grid5000_cli.main(["submit", *_arguments("--execute")])

    assert exit_code == 0
    assert calls == [True]
    assert json.loads(capsys.readouterr().out)["job_id"] == 7


def test_plan_can_explicitly_enable_final_publication_and_trackio_sync(
    capsys,
) -> None:
    exit_code = grid5000_cli.main(["plan", *_arguments("--publish", "--sync-trackio")])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["identity"]["training_config"]["publish_to_hub"] is True
    assert payload["identity"]["training_config"]["sync_trackio"] is True


def test_plan_can_request_a_policy_bounded_day_allocation(capsys) -> None:
    exit_code = grid5000_cli.main(
        [
            "plan",
            *_arguments(
                "--policy-type",
                "day",
                "--walltime-seconds",
                "3600",
            ),
        ]
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["allocation"]["policy_type"] == "day"
    assert payload["allocation"]["walltime_seconds"] == 3_600


def test_day_policy_defaults_to_a_short_thirty_minute_allocation(capsys) -> None:
    exit_code = grid5000_cli.main(["plan", *_arguments("--policy-type", "day")])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["allocation"]["walltime_seconds"] == 1_800
    assert payload["allocation"]["walltime"] == "00:30:00"
