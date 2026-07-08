import re
import sqlite3
from dataclasses import dataclass, field

import sqlglot
from sqlglot.errors import SqlglotError

from lab.data.schema import SQLExample

_FENCE = re.compile(r"```(?:sql)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_SELECT_LIKE = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)


def extract_sql(raw: str) -> str:
    text = raw.strip()
    match = _FENCE.search(text)
    if match:
        text = match.group(1).strip()
    try:
        statements = sqlglot.parse(text)
    except SqlglotError:
        statements = []
    if statements and statements[0] is not None:
        return statements[0].sql() + ";"
    lines = [line for line in text.splitlines() if line.strip()]
    for index, line in enumerate(lines):
        if _SELECT_LIKE.match(line):
            candidate = " ".join(lines[index:])
            return candidate.split(";")[0].strip() + ";"
    return text.split(";")[0].strip() + ";" if text else ""


@dataclass
class ExecutionResult:
    ok: bool
    rows: list[tuple[str, ...]] = field(default_factory=list)
    error: str = ""


def _normalize_rows(rows: list[tuple[object, ...]]) -> list[tuple[str, ...]]:
    normalized = [tuple(str(value) for value in row) for row in rows]
    normalized.sort()
    return normalized


def execute_query(context: str, query: str) -> ExecutionResult:
    connection = sqlite3.connect(":memory:")
    try:
        connection.executescript(context)
        cursor = connection.execute(query)
        rows = cursor.fetchall()
        return ExecutionResult(ok=True, rows=_normalize_rows(rows))
    except sqlite3.Error as exc:
        return ExecutionResult(ok=False, error=str(exc))
    finally:
        connection.close()


def is_select_query(query: str) -> bool:
    return bool(_SELECT_LIKE.match(query.strip()))


def normalized_sql_match(gold: str, predicted: str) -> bool:
    try:
        gold_normalized = sqlglot.transpile(gold, read="sqlite", pretty=False)
        predicted_normalized = sqlglot.transpile(predicted, read="sqlite", pretty=False)
    except SqlglotError:
        return False
    return bool(gold_normalized) and [q.lower() for q in gold_normalized] == [
        q.lower() for q in predicted_normalized
    ]


@dataclass
class PredictionScore:
    valid_sql: bool
    execution_checked: bool
    execution_match: bool
    normalized_match: bool

    @property
    def correct(self) -> bool:
        if self.execution_checked:
            return self.execution_match
        return self.normalized_match


def score_prediction(example: SQLExample, raw_prediction: str) -> PredictionScore:
    predicted = extract_sql(raw_prediction)
    try:
        parsed = sqlglot.parse(predicted) if predicted else []
        valid = bool(parsed) and all(node is not None for node in parsed)
    except SqlglotError:
        valid = False
    if not valid:
        return PredictionScore(
            valid_sql=False,
            execution_checked=False,
            execution_match=False,
            normalized_match=False,
        )
    normalized = normalized_sql_match(example.sql, predicted)
    if is_select_query(example.sql):
        gold_result = execute_query(example.context, example.sql)
        predicted_result = execute_query(example.context, predicted)
        if gold_result.ok:
            return PredictionScore(
                valid_sql=True,
                execution_checked=True,
                execution_match=predicted_result.ok and predicted_result.rows == gold_result.rows,
                normalized_match=normalized,
            )
    return PredictionScore(
        valid_sql=True,
        execution_checked=False,
        execution_match=False,
        normalized_match=normalized,
    )
