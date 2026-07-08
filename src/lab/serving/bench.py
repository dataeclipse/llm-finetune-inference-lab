import asyncio
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lab.config import LabConfig
from lab.data.format import SYSTEM_PROMPT
from lab.logging import get_logger
from lab.serving.client import InferenceClient, OpenAICompatClient

logger = get_logger(__name__)

BENCH_PROMPTS = [
    "Count the number of orders placed in the last 30 days.",
    "Find the top five customers by total revenue.",
    "List employees hired after 2020 grouped by department.",
    "Compute the average delivery time per region.",
    "Show products that were never ordered.",
    "Find the median salary per job title.",
    "List the daily active users for the past week.",
    "Compute month-over-month revenue growth.",
]


@dataclass
class BenchReport:
    total_requests: int
    concurrency: int
    error_count: int
    latency_p50: float
    latency_p95: float
    latency_p99: float
    throughput_tokens_per_second: float
    requests_per_second: float


def summarize(
    latencies: list[float],
    tokens: list[int],
    errors: int,
    concurrency: int,
    wall_seconds: float,
) -> BenchReport:
    array = np.asarray(latencies, dtype=np.float64)
    return BenchReport(
        total_requests=len(latencies) + errors,
        concurrency=concurrency,
        error_count=errors,
        latency_p50=float(np.percentile(array, 50)) if latencies else 0.0,
        latency_p95=float(np.percentile(array, 95)) if latencies else 0.0,
        latency_p99=float(np.percentile(array, 99)) if latencies else 0.0,
        throughput_tokens_per_second=sum(tokens) / wall_seconds if wall_seconds else 0.0,
        requests_per_second=len(latencies) / wall_seconds if wall_seconds else 0.0,
    )


async def run_load(
    client: InferenceClient,
    total_requests: int,
    concurrency: int,
    max_tokens: int,
) -> BenchReport:
    semaphore = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    tokens: list[int] = []
    errors = 0

    async def one_request(index: int) -> None:
        nonlocal errors
        prompt = BENCH_PROMPTS[index % len(BENCH_PROMPTS)]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Database schema:\nCREATE TABLE t (id INT);\n\nQuestion: {prompt}",
            },
        ]
        async with semaphore:
            try:
                result = await client.complete(messages, max_tokens=max_tokens)
            except Exception:
                errors += 1
                logger.exception("bench_request_failed", index=index)
                return
        latencies.append(result.latency_seconds)
        tokens.append(result.completion_tokens)

    loop = asyncio.get_running_loop()
    started = loop.time()
    await asyncio.gather(*(one_request(index) for index in range(total_requests)))
    wall_seconds = loop.time() - started
    return summarize(latencies, tokens, errors, concurrency, wall_seconds)


def write_bench_report(report: BenchReport, path: str = "reports/bench.md") -> None:
    report_file = Path(path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Serving Benchmark",
        "",
        f"Requests: {report.total_requests}, concurrency: {report.concurrency}, "
        f"errors: {report.error_count}",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Latency p50 | {report.latency_p50:.3f}s |",
        f"| Latency p95 | {report.latency_p95:.3f}s |",
        f"| Latency p99 | {report.latency_p99:.3f}s |",
        f"| Throughput | {report.throughput_tokens_per_second:.1f} tokens/s |",
        f"| Requests/s | {report.requests_per_second:.2f} |",
    ]
    report_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_benchmark(
    config: LabConfig,
    base_url: str,
    concurrency: int,
    total_requests: int,
    client: InferenceClient | None = None,
) -> BenchReport:
    active_client = client or OpenAICompatClient(
        base_url=base_url, model=config.serve.served_model_name
    )
    report = asyncio.run(
        run_load(
            active_client,
            total_requests=total_requests,
            concurrency=concurrency,
            max_tokens=config.eval.max_new_tokens,
        )
    )
    write_bench_report(report)
    logger.info(
        "bench_finished",
        p50=report.latency_p50,
        p95=report.latency_p95,
        tps=report.throughput_tokens_per_second,
        errors=report.error_count,
    )
    return report
