from lab.data.schema import SQLExample
from lab.eval.sql_exec import (
    execute_query,
    extract_sql,
    is_select_query,
    normalized_sql_match,
    score_prediction,
)

CONTEXT = (
    "CREATE TABLE employees (id INT, name TEXT, salary INT); "
    "INSERT INTO employees VALUES (1, 'Alice', 120), (2, 'Bob', 80), (3, 'Cara', 150);"
)


def make_example(sql: str) -> SQLExample:
    return SQLExample(question="Who earns more than 100?", context=CONTEXT, sql=sql)


def test_extract_sql_from_fenced_block() -> None:
    raw = "Here is the query:\n```sql\nSELECT name FROM employees WHERE salary > 100\n```\nDone."
    assert extract_sql(raw).lower().startswith("select name from employees")


def test_extract_sql_plain_statement() -> None:
    assert extract_sql("SELECT 1").rstrip(";") == "SELECT 1"


def test_extract_sql_with_preamble() -> None:
    raw = "The answer to your question:\nSELECT id FROM employees;\nHope that helps."
    assert extract_sql(raw).lower().startswith("select id from employees")


def test_execute_query_returns_sorted_rows() -> None:
    result = execute_query(CONTEXT, "SELECT name FROM employees WHERE salary > 100")
    assert result.ok
    assert result.rows == [("Alice",), ("Cara",)]


def test_execute_query_reports_errors() -> None:
    result = execute_query(CONTEXT, "SELECT missing_column FROM employees")
    assert not result.ok
    assert "missing_column" in result.error


def test_is_select_query() -> None:
    assert is_select_query("SELECT 1")
    assert is_select_query("  WITH t AS (SELECT 1) SELECT * FROM t")
    assert not is_select_query("INSERT INTO employees VALUES (4, 'Dan', 90)")


def test_normalized_sql_match_case_insensitive() -> None:
    assert normalized_sql_match("select name from employees", "SELECT name FROM employees")
    assert not normalized_sql_match("SELECT id FROM employees", "SELECT name FROM employees")


def test_score_prediction_execution_match_different_syntax() -> None:
    example = make_example("SELECT name FROM employees WHERE salary > 100;")
    prediction = "```sql\nSELECT name FROM employees WHERE salary >= 101\n```"
    score = score_prediction(example, prediction)
    assert score.valid_sql
    assert score.execution_checked
    assert score.execution_match
    assert score.correct


def test_score_prediction_wrong_result() -> None:
    example = make_example("SELECT name FROM employees WHERE salary > 100;")
    score = score_prediction(example, "SELECT name FROM employees WHERE salary < 100")
    assert score.valid_sql
    assert score.execution_checked
    assert not score.execution_match
    assert not score.correct


def test_score_prediction_invalid_sql() -> None:
    example = make_example("SELECT name FROM employees;")
    score = score_prediction(example, "I cannot answer that question.")
    assert not score.valid_sql
    assert not score.correct


def test_score_prediction_non_select_falls_back_to_normalized() -> None:
    example = make_example("INSERT INTO employees VALUES (4, 'Dan', 90);")
    score = score_prediction(example, "insert into employees values (4, 'Dan', 90)")
    assert score.valid_sql
    assert not score.execution_checked
    assert score.normalized_match
    assert score.correct
