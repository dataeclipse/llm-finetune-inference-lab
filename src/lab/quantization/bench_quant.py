import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from lab.config import LabConfig
from lab.logging import get_logger

logger = get_logger(__name__)

MeasureFn = Callable[[str], tuple[float, float]]


@dataclass
class QuantBenchRow:
    label: str
    perplexity: float
    tokens_per_second: float


def measure_model(config: LabConfig, model_path: str) -> tuple[float, float]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from lab.eval.perplexity import compute_perplexity

    perplexity = compute_perplexity(config, model_path, split="val")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    prompt = "Write a SQL query that counts rows in a table named events."
    tokens = tokenizer(prompt, return_tensors="pt").to(device)
    generated_tokens = 0
    started = time.perf_counter()
    for _ in range(3):
        with torch.no_grad():
            output = model.generate(
                **tokens,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        generated_tokens += output.shape[-1] - tokens["input_ids"].shape[-1]
    elapsed = time.perf_counter() - started
    return perplexity, generated_tokens / elapsed if elapsed else 0.0


def run_quant_bench(
    config: LabConfig,
    model_paths: dict[str, str],
    measure: MeasureFn | None = None,
    report_path: str = "reports/quantization.md",
) -> list[QuantBenchRow]:
    measure_fn = measure or (lambda path: measure_model(config, path))
    rows: list[QuantBenchRow] = []
    for label, path in model_paths.items():
        perplexity, tokens_per_second = measure_fn(path)
        rows.append(
            QuantBenchRow(label=label, perplexity=perplexity, tokens_per_second=tokens_per_second)
        )
        logger.info("quant_bench_row", label=label, perplexity=perplexity, tps=tokens_per_second)
    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Quantization Benchmark",
        "",
        "| Model | Perplexity (val) | Tokens/sec |",
        "|-------|------------------|------------|",
    ]
    for row in rows:
        lines.append(f"| {row.label} | {row.perplexity:.3f} | {row.tokens_per_second:.1f} |")
    report_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows
