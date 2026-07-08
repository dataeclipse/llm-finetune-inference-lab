# ADR 0001: Qwen3-8B as Base Model

## Status

Accepted

## Context

The target task is text-to-SQL fine-tuning on a single A100 40GB (Colab
Pro+). Candidates in the 7-9B class: Qwen3-8B, Qwen2.5-7B-Instruct,
Llama-3.1-8B-Instruct. Requirements: strong code/SQL priors, permissive
license, first-class support in TRL, PEFT and vLLM, and a small sibling model
for CPU smoke tests.

## Decision

Qwen3-8B is the default (`configs/profile/colab_a100.yaml`);
Llama-3.1-8B-Instruct remains a config-level fallback. CI and local smoke
tests use SmolLM2-135M-Instruct through the `cpu_smoke` profile, exercising
the identical code path in about a minute on CPU.

## Consequences

- Qwen3's stronger coding baseline raises the pre-finetune floor, so gains
  reported for SFT/DPO are measured against a harder baseline.
- With QLoRA (4-bit NF4) the 8B model trains within ~12GB of VRAM, leaving
  headroom for batch size 4 with 2048-token sequences on the A100.
- The smoke profile guarantees trainer regressions surface in CI without a
  GPU: the same `run_sft`/`run_dpo` functions run end-to-end on 135M weights.
- Switching the base model is a one-line profile change; nothing in the code
  references the model family.
