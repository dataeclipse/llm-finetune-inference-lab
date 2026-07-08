# ADR 0005: vLLM Serving with AWQ and GGUF Export Targets

## Status

Accepted

## Context

The fine-tuned model must serve an OpenAI-compatible API with production
throughput on the same A100, and also be distributable to weaker hardware.
Serving candidates: vLLM, TGI, TensorRT-LLM, llama.cpp.

## Decision

vLLM is the primary server (PagedAttention, continuous batching, native
OpenAI API). The launcher shells out to `vllm serve` and health-checks the
endpoint. All client code — benchmarks, A/B tests, pair mining, the Gradio
UI — talks to the `InferenceClient` abstraction whose only implementation
requirement is an OpenAI-compatible endpoint, so TGI or TensorRT-LLM drop in
by changing a URL.

Two quantized export targets: AWQ 4-bit for GPU serving via vLLM, GGUF
(Q4_K_M) via llama.cpp for CPU/laptop inference. Both are produced from the
merged checkpoint and benchmarked for perplexity and throughput before use.

## Consequences

- The OpenAI-compatible seam means every consumer is testable with a stub
  client; no test needs a GPU or a running server.
- vLLM and AutoAWQ only install on Linux; the dependency markers keep
  `uv sync` working on Windows/macOS for development, with serving features
  raising actionable errors instead of import crashes.
- GGUF export depends on an external llama.cpp checkout (`LLAMA_CPP_DIR`)
  rather than vendored conversion code that would rot.
- Benchmarks (p50/p95/p99, tokens/s) are written to `reports/` by the same
  code path used in CI-mocked tests, so the numbers in the README are
  regenerable with one command.
