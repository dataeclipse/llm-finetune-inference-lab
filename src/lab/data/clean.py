import hashlib
from collections import Counter

import sqlglot
from sqlglot.errors import ParseError

from lab.config import DataConfig
from lab.data.schema import SQLExample


def is_parseable_sql(statement: str) -> bool:
    if not statement.strip():
        return False
    try:
        parsed = sqlglot.parse(statement)
    except ParseError:
        return False
    return all(expression is not None for expression in parsed)


def _question_key(question: str) -> str:
    normalized = " ".join(question.lower().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def clean_examples(
    examples: list[SQLExample], config: DataConfig
) -> tuple[list[SQLExample], Counter[str]]:
    dropped: Counter[str] = Counter()
    seen: set[str] = set()
    cleaned: list[SQLExample] = []
    for example in examples:
        if not example.question.strip() or not example.context.strip():
            dropped["empty_fields"] += 1
            continue
        if len(example.question) + len(example.context) > config.max_prompt_chars:
            dropped["prompt_too_long"] += 1
            continue
        if len(example.sql) > config.max_sql_chars:
            dropped["sql_too_long"] += 1
            continue
        if not is_parseable_sql(example.sql):
            dropped["invalid_sql"] += 1
            continue
        if not is_parseable_sql(example.context):
            dropped["invalid_context"] += 1
            continue
        key = _question_key(example.question)
        if key in seen:
            dropped["duplicate_question"] += 1
            continue
        seen.add(key)
        cleaned.append(
            example.model_copy(
                update={
                    "question": example.question.strip(),
                    "context": example.context.strip(),
                    "sql": example.sql.strip().rstrip(";") + ";",
                }
            )
        )
    return cleaned, dropped
