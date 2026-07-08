# ADR 0002: QLoRA Instead of Full Fine-Tuning

## Status

Accepted

## Context

Full bf16 fine-tuning of an 8B model with Adam needs roughly 130GB of GPU
memory (16 bytes/parameter for weights, gradients and optimizer state) -
far beyond a single A100 40GB. Options: full fine-tune on rented multi-GPU,
LoRA on bf16 weights (~19GB base + adapter overhead), or QLoRA (4-bit NF4
base + bf16 LoRA adapters).

## Decision

QLoRA: base weights in 4-bit NF4 with double quantization, bf16 compute,
gradient checkpointing, LoRA rank 16 with alpha 32 on all attention and MLP
projections. The same LoRA settings feed both SFT and DPO; DPO reuses the
SFT adapter as its starting point and implicit reference.

## Consequences

- Training fits in ~12GB, leaving room for longer sequences or larger batch
  on the A100; sessions that die resume from Drive checkpoints because only
  small adapter states are saved frequently.
- Quantized base + adapters costs some quality versus full fine-tuning, but
  for a narrow structured-output task LoRA on all projections is the
  standard quality/cost sweet spot.
- `load_in_4bit` degrades gracefully: on machines without CUDA the flag is
  ignored with a warning, which is what allows CPU smoke tests to share the
  code path.
- Serving requires an explicit merge step (`lab export merge`) because vLLM
  wants a single set of weights; the merge is part of the pipeline, not an
  afterthought.
