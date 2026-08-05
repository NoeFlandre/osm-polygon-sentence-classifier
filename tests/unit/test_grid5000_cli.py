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
