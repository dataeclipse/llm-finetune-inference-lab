import asyncio
from pathlib import Path

import pytest

from lab.config import LabConfig, load_config
from lab.data.prepare import prepare_dataset
from lab.data.schema import SQLExample
from lab.exceptions import ServingError
from lab.serving.ab_test import compare_models, write_ab_report
from lab.serving.bench import run_load, summarize, write_bench_report
from lab.serving.client import CompletionResult, InferenceClient
from lab.serving.vllm_server import build_vllm_command, launch_vllm

CONTEXT = (
    "CREATE TABLE employees (id INT, name TEXT, salary INT); "
    "INSERT INTO employees VALUES (1, 'Alice', 120), (2, 'Bob', 80);"
)


class StubClient(InferenceClient):
    def __init__(self, text: str, latency: float = 0.05, fail_every: int = 0) -> None:
        self._text = text
        self._latency = latency
        self._fail_every = fail_every
        self.calls = 0

    async def complete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> CompletionResult:
        self.calls += 1
        if self._fail_every and self.calls % self._fail_every == 0:
            raise RuntimeError("stub failure")
        await asyncio.sleep(0)
        return CompletionResult(
            text=self._text, latency_seconds=self._latency, completion_tokens=32
        )


def test_build_vllm_command_includes_settings() -> None:
    config = load_config(["serve.port=9000", "serve.max_model_len=2048"])
    command = build_vllm_command(config)
    assert command[:3] == ["vllm", "serve", config.serve.model_path]
    assert "--port" in command
    assert "9000" in command
    assert "2048" in command


def test_launch_vllm_requires_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("lab.serving.vllm_server.shutil.which", lambda name: None)
    config = load_config()
    with pytest.raises(ServingError, match="vllm is not installed"):
        launch_vllm(config)


async def test_run_load_collects_metrics() -> None:
    client = StubClient("SELECT 1;")
    report = await run_load(client, total_requests=16, concurrency=4, max_tokens=64)
    assert report.total_requests == 16
    assert report.error_count == 0
    assert report.latency_p50 == pytest.approx(0.05)
    assert report.throughput_tokens_per_second > 0


async def test_run_load_counts_errors() -> None:
    client = StubClient("SELECT 1;", fail_every=4)
    report = await run_load(client, total_requests=8, concurrency=2, max_tokens=64)
    assert report.error_count == 2
    assert report.total_requests == 8


def test_summarize_percentiles() -> None:
    report = summarize(
        latencies=[0.1] * 90 + [1.0] * 10,
        tokens=[10] * 100,
        errors=0,
        concurrency=8,
        wall_seconds=10.0,
    )
    assert report.latency_p50 == pytest.approx(0.1)
    assert report.latency_p95 == pytest.approx(1.0)
    assert report.throughput_tokens_per_second == pytest.approx(100.0)


def test_write_bench_report(tmp_path: Path) -> None:
    report = summarize([0.2], [50], errors=1, concurrency=2, wall_seconds=1.0)
    path = tmp_path / "bench.md"
    write_bench_report(report, path=str(path))
    content = path.read_text(encoding="utf-8")
    assert "Latency p95" in content
    assert "errors: 1" in content


def make_eval_config(tmp_path: Path) -> LabConfig:
    config = load_config(
        [
            f"data.output_dir={tmp_path.as_posix()}",
            "data.target_examples=20",
            "eval.num_examples=5",
        ]
    )
    raw = [
        SQLExample(
            question=f"Question number {index}?",
            context=CONTEXT,
            sql="SELECT name FROM employees WHERE salary > 100;",
        )
        for index in range(30)
    ]
    prepare_dataset(config, raw=raw)
    return config


async def test_compare_models_scores_both_sides(tmp_path: Path) -> None:
    config = make_eval_config(tmp_path)
    good = StubClient("SELECT name FROM employees WHERE salary > 100;")
    bad = StubClient("SELECT name FROM employees WHERE salary < 100;")
    result = await compare_models(config, good, bad, "sft", "base")
    assert result.accuracy_a == 1.0
    assert result.accuracy_b == 0.0
    assert result.wins_a == result.total
    assert result.wins_b == 0


def test_write_ab_report(tmp_path: Path) -> None:
    from lab.serving.ab_test import ABResult

    result = ABResult(
        model_a="sft",
        model_b="base",
        total=10,
        wins_a=6,
        wins_b=1,
        ties=3,
        accuracy_a=0.8,
        accuracy_b=0.3,
    )
    path = tmp_path / "ab.md"
    write_ab_report(result, path=str(path))
    content = path.read_text(encoding="utf-8")
    assert "| sft | 0.800 | 6 |" in content
    assert "| ties | | 3 |" in content
