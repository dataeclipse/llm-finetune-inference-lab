# ADR 0003: Preference Pairs Mined from Execution Failures

## Status

Accepted

## Context

DPO needs (prompt, chosen, rejected) triples. Common shortcuts — sampling
two model outputs and ranking them with an LLM judge, or using a public
generic preference set — either inherit judge noise or do not target the
task. The training set already contains verified gold SQL.

## Decision

Pairs are mined from the SFT model's own mistakes: sample the model at
temperature 0.7 over training prompts, execute each prediction against the
example's schema in SQLite, and keep only predictions that fail the
execution check. The failing generation becomes `rejected`, the gold query
becomes `chosen` (`lab train pairs`).

## Consequences

- Every pair encodes a real, verifiable error of the current policy — the
  preference signal is grounded in execution semantics, not judge opinion.
- Pair quality is deterministic and auditable; the miner is unit-tested with
  scripted generators.
- The approach only corrects mistakes the SFT model still makes, which is
  exactly DPO's role after SFT; it cannot teach new SQL constructs.
- Mining requires one inference pass over the training prompts (vLLM makes
  this a few minutes for 4k prompts on the A100).
