from pathlib import Path

from lab.config import load_config
from lab.data.format import read_jsonl
from lab.data.prepare import prepare_dataset
from lab.data.schema import SQLExample
from lab.training.pairs import collect_pairs, generate_pairs, load_train_examples

CONTEXT = (
    "CREATE TABLE employees (id INT, name TEXT, salary INT); "
    "INSERT INTO employees VALUES (1, 'Alice', 120), (2, 'Bob', 80);"
)


def make_corpus(count: int) -> list[SQLExample]:
    return [
        SQLExample(
            question=f"Question variant number {index}?",
            context=CONTEXT,
            sql="SELECT name FROM employees WHERE salary > 100;",
        )
        for index in range(count)
    ]


async def test_collect_pairs_keeps_only_failures() -> None:
    examples = make_corpus(4)
    responses = iter(
        [
            "SELECT name FROM employees WHERE salary > 100;",
            "SELECT name FROM employees WHERE salary < 100;",
            "not sql at all",
            "SELECT name FROM employees WHERE salary > 100;",
        ]
    )

    async def generate(messages: list[dict[str, str]]) -> str:
        return next(responses)

    pairs = await collect_pairs(examples, generate, max_pairs=10)
    assert len(pairs) == 2
    for pair in pairs:
        assert pair["chosen"][0]["content"].startswith("SELECT name")
        assert pair["rejected"][0]["content"] != pair["chosen"][0]["content"]
        assert pair["prompt"][0]["role"] == "system"


async def test_collect_pairs_respects_cap() -> None:
    async def generate(messages: list[dict[str, str]]) -> str:
        return "SELECT nothing FROM nowhere;"

    pairs = await collect_pairs(make_corpus(10), generate, max_pairs=3)
    assert len(pairs) == 3


def test_generate_pairs_end_to_end(tmp_path: Path) -> None:
    pairs_path = tmp_path / "dpo_pairs.jsonl"
    config = load_config(
        [
            f"data.output_dir={tmp_path.as_posix()}",
            "data.target_examples=20",
            f"dpo.pairs_path={pairs_path.as_posix()}",
            "dpo.max_pairs=5",
        ]
    )
    prepare_dataset(config, raw=make_corpus(30))

    async def always_wrong(messages: list[dict[str, str]]) -> str:
        return "SELECT name FROM employees WHERE salary < 0;"

    count = generate_pairs(config, base_url="unused", model="unused", generate=always_wrong)
    assert count == 5
    rows = read_jsonl(pairs_path)
    assert len(rows) == 5
    assert {"prompt", "chosen", "rejected"} <= set(rows[0])


def test_load_train_examples_roundtrip(tmp_path: Path) -> None:
    config = load_config([f"data.output_dir={tmp_path.as_posix()}", "data.target_examples=20"])
    prepare_dataset(config, raw=make_corpus(30))
    examples = load_train_examples(config)
    assert examples
    first = examples[0]
    assert first.context.startswith("CREATE TABLE employees")
    assert first.question.endswith("?")
    assert first.sql.endswith(";")
