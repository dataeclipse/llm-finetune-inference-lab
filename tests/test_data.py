import json
from pathlib import Path

import pytest

from lab.config import load_config
from lab.data.clean import clean_examples, is_parseable_sql
from lab.data.download import row_to_example
from lab.data.format import (
    SYSTEM_PROMPT,
    build_user_prompt,
    read_jsonl,
    split_examples,
    to_messages,
    write_jsonl,
)
from lab.data.prepare import prepare_dataset
from lab.data.schema import SQLExample
from lab.data.stats import collect_split_stats, length_stats

CONTEXT = "CREATE TABLE employees (id INT, name TEXT, salary INT);"


def make_example(index: int = 0, sql: str = "SELECT name FROM employees;") -> SQLExample:
    return SQLExample(
        question=f"Which employees earn more than {index}?",
        context=CONTEXT,
        sql=sql,
        complexity="basic",
        domain="hr",
    )


def make_corpus(count: int) -> list[SQLExample]:
    return [make_example(index) for index in range(count)]


def test_is_parseable_sql() -> None:
    assert is_parseable_sql("SELECT 1")
    assert is_parseable_sql(CONTEXT)
    assert not is_parseable_sql("SELEC name FRM employees")
    assert not is_parseable_sql("")


def test_clean_drops_invalid_sql() -> None:
    config = load_config().data
    examples = [make_example(0), make_example(1, sql="NOT REALLY SQL AT ALL !!!")]
    cleaned, dropped = clean_examples(examples, config)
    assert len(cleaned) == 1
    assert dropped["invalid_sql"] == 1


def test_clean_deduplicates_questions() -> None:
    config = load_config().data
    duplicate = make_example(0)
    shouting = duplicate.model_copy(update={"question": duplicate.question.upper() + "  "})
    cleaned, dropped = clean_examples([duplicate, shouting], config)
    assert len(cleaned) == 1
    assert dropped["duplicate_question"] == 1


def test_clean_enforces_length_caps() -> None:
    config = load_config().data
    long_example = make_example(0).model_copy(update={"context": "x" * 10000})
    cleaned, dropped = clean_examples([long_example], config)
    assert cleaned == []
    assert dropped["prompt_too_long"] == 1


def test_clean_normalizes_sql_terminator() -> None:
    config = load_config().data
    cleaned, _ = clean_examples([make_example(0, sql="SELECT 1")], config)
    assert cleaned[0].sql == "SELECT 1;"


def test_to_messages_structure() -> None:
    example = make_example()
    messages = to_messages(example)
    assert [message["role"] for message in messages] == ["system", "user", "assistant"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert CONTEXT in messages[1]["content"]
    assert messages[2]["content"] == example.sql
    assert build_user_prompt(example).startswith("Database schema:")


def test_split_deterministic_and_disjoint() -> None:
    corpus = make_corpus(100)
    train_a, val_a, test_a = split_examples(corpus, 0.1, 0.1, seed=7)
    train_b, _val_b, _test_b = split_examples(corpus, 0.1, 0.1, seed=7)
    assert [e.question for e in train_a] == [e.question for e in train_b]
    assert len(val_a) == 10
    assert len(test_a) == 10
    assert len(train_a) == 80
    questions = (
        {e.question for e in train_a} | {e.question for e in val_a} | {e.question for e in test_a}
    )
    assert len(questions) == 100


def test_split_rejects_bad_fractions() -> None:
    with pytest.raises(ValueError):
        split_examples(make_corpus(10), 0.6, 0.5, seed=1)


def test_row_to_example_gretel_fields() -> None:
    row = {
        "sql_prompt": "How many employees?",
        "sql_context": CONTEXT,
        "sql": "SELECT COUNT(*) FROM employees;",
        "sql_complexity": "aggregation",
        "domain": "hr",
    }
    example = row_to_example(row)
    assert example.question == "How many employees?"
    assert example.complexity == "aggregation"


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    rows: list[dict[str, object]] = [{"a": 1}, {"b": "текст"}]
    write_jsonl(path, rows)
    assert read_jsonl(path) == rows


def test_prepare_dataset_end_to_end(tmp_path: Path) -> None:
    config = load_config([f"data.output_dir={tmp_path.as_posix()}", "data.target_examples=20"])
    summary = prepare_dataset(config, raw=make_corpus(30))
    assert summary.cleaned_examples == 30
    assert summary.train_examples + summary.val_examples + summary.test_examples == 20
    train_rows = read_jsonl(tmp_path / "train.jsonl")
    assert "messages" in train_rows[0]
    test_rows = read_jsonl(tmp_path / "test.jsonl")
    assert {"question", "context", "sql"} <= set(test_rows[0])
    summary_payload = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary_payload["train_examples"] == summary.train_examples


def test_length_stats_percentiles() -> None:
    stats = length_stats(list(range(1, 101)))
    assert stats["count"] == 100
    assert stats["p50"] == pytest.approx(50.5)
    assert stats["max"] == 100


def test_collect_split_stats(tmp_path: Path) -> None:
    config = load_config([f"data.output_dir={tmp_path.as_posix()}", "data.target_examples=20"])
    prepare_dataset(config, raw=make_corpus(30))
    stats = collect_split_stats(tmp_path / "train.jsonl")
    assert stats["prompt_words"]["count"] > 0
    assert stats["completion_words"]["mean"] > 0
