import json
import random
from pathlib import Path

from lab.data.schema import SQLExample

SYSTEM_PROMPT = (
    "You are a text-to-SQL assistant. Given a database schema and a question, "
    "respond with a single valid SQL query and nothing else."
)


def build_user_prompt(example: SQLExample) -> str:
    return f"Database schema:\n{example.context}\n\nQuestion: {example.question}"


def to_messages(example: SQLExample) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(example)},
        {"role": "assistant", "content": example.sql},
    ]


def split_examples(
    examples: list[SQLExample],
    val_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[list[SQLExample], list[SQLExample], list[SQLExample]]:
    if val_fraction + test_fraction >= 1.0:
        raise ValueError("val and test fractions must sum to less than 1")
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    test_size = int(total * test_fraction)
    val_size = int(total * val_fraction)
    test = shuffled[:test_size]
    val = shuffled[test_size : test_size + val_size]
    train = shuffled[test_size + val_size :]
    return train, val, test


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows
