# ADR 0004: Execution Accuracy as the Primary Metric

## Status

Accepted

## Context

Text-to-SQL evaluation options: exact string match (brittle - `>=101` and
`>100` differ textually but agree semantically), LLM-as-judge (subjective,
costs an extra model call, and its errors correlate with the errors of the
model being judged), or execution accuracy (run both queries, compare result
sets).

## Decision

A layered scorer (`lab/eval/sql_exec.py`):

1. Extract SQL from raw model output (code fences, preamble stripping,
   sqlglot round-trip).
2. Validity: does the prediction parse at all.
3. For SELECT-family gold queries: build the schema (with inline INSERTs) in
   an in-memory SQLite database, execute both queries, compare row sets
   order-insensitively - the primary correctness signal.
4. For non-SELECT gold queries: sqlglot-normalized string equality.
5. LLM-as-judge exists as a secondary, clearly labelled metric.

## Consequences

- The headline number is objective and reproducible offline with zero model
  calls; CI asserts the scorer's behavior with fixture databases.
- Execution comparison accepts semantically different queries that happen to
  coincide on the fixture data (false positives on small tables); gretel
  contexts include seed rows, which limits but does not eliminate this.
- SQLite dialect differences from the source data are absorbed by sqlglot
  transpilation; queries that use unsupported syntax fall back to normalized
  match rather than being scored as failures of the model.
