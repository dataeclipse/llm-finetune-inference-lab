from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from lab.config import load_config
from lab.data.prepare import prepare_dataset
from lab.data.schema import SQLExample
from lab.eval.judge import SQLJudge, build_judge_prompt, parse_verdict
from lab.eval.runner import (
    append_report,
    build_eval_messages,
    evaluate_model,
    load_test_examples,
)

CONTEXT = (
    "CREATE TABLE employees (id INT, name TEXT, salary INT); "
    "INSERT INTO employees VALUES (1, 'Alice', 120), (2, 'Bob', 80);"
)


def make_examples() -> list[SQLExample]:
    return [
        SQLExample(
            question="Who earns more than 100?",
            context=CONTEXT,
            sql="SELECT name FROM employees WHERE salary > 100;",
        ),
        SQLExample(
            question="How many employees are there?",
            context=CONTEXT,
            sql="SELECT COUNT(*) FROM employees;",
        ),
    ]


async def test_evaluate_model_perfect_predictions() -> None:
    examples = make_examples()
    answers = {example.question: example.sql for example in examples}

    async def generate(messages: list[dict[str, str]]) -> str:
        question = messages[1]["content"].split("Question: ")[1]
        return answers[question]

    report = await evaluate_model(examples, generate, model_label="gold")
    assert report.total == 2
    assert report.overall_accuracy == 1.0
    assert report.valid_sql_rate == 1.0
    assert report.execution_accuracy == 1.0


async def test_evaluate_model_garbage_predictions() -> None:
    async def generate(messages: list[dict[str, str]]) -> str:
        return "I do not know."

    report = await evaluate_model(make_examples(), generate, model_label="broken")
    assert report.overall_accuracy == 0.0
    assert report.valid_sql_rate == 0.0


def test_build_eval_messages_has_no_answer() -> None:
    messages = build_eval_messages(make_examples()[0])
    assert len(messages) == 2
    assert messages[-1]["role"] == "user"


def test_load_test_examples_and_report(tmp_path: Path) -> None:
    report_path = tmp_path / "reports" / "eval.md"
    config = load_config(
        [
            f"data.output_dir={tmp_path.as_posix()}",
            "data.target_examples=20",
            f"eval.report_path={report_path.as_posix()}",
            "eval.num_examples=3",
        ]
    )
    raw = [
        SQLExample(
            question=f"Question number {index}?",
            context=CONTEXT,
            sql="SELECT COUNT(*) FROM employees;",
        )
        for index in range(30)
    ]
    prepare_dataset(config, raw=raw)
    examples = load_test_examples(config)
    assert len(examples) == 3
    from lab.eval.runner import EvalReport

    report = EvalReport(
        model_label="base",
        total=3,
        valid_sql_rate=0.9,
        execution_accuracy=0.8,
        normalized_match_rate=0.5,
        overall_accuracy=0.8,
    )
    append_report(config, report)
    append_report(config, report)
    content = report_path.read_text(encoding="utf-8")
    assert content.count("| base |") == 2
    assert content.count("# Evaluation Report") == 1


def test_parse_verdict() -> None:
    assert parse_verdict('{"correct": true, "reason": "same"}') is True
    assert parse_verdict('prefix {"correct": false} suffix') is False
    assert parse_verdict("not json") is None
    assert parse_verdict('{"other": 1}') is None


async def test_judge_calls_client() -> None:
    client = MagicMock()
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content='{"correct": true}'))]
    client.chat.completions.create = AsyncMock(return_value=response)
    judge = SQLJudge(client, model="judge-model")
    example = make_examples()[0]
    verdict = await judge.judge(example, "SELECT name FROM employees WHERE salary > 100")
    assert verdict is True
    prompt = client.chat.completions.create.await_args.kwargs["messages"][1]["content"]
    assert example.question in prompt
    assert "Reference query" in build_judge_prompt(example, "x")


async def test_judge_unparseable_returns_none() -> None:
    client = MagicMock()
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content="garbage"))]
    client.chat.completions.create = AsyncMock(return_value=response)
    verdict = await SQLJudge(client, model="m").judge(make_examples()[0], "SELECT 1")
    assert verdict is None


@pytest.mark.parametrize("temperature", [0.0])
def test_eval_config_defaults(temperature: float) -> None:
    config = load_config()
    assert config.eval.temperature == temperature
