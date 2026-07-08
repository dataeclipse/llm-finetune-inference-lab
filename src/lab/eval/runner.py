import asyncio
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from lab.config import LabConfig
from lab.data.format import SYSTEM_PROMPT, build_user_prompt, read_jsonl
from lab.data.schema import SQLExample
from lab.eval.sql_exec import score_prediction
from lab.logging import get_logger

logger = get_logger(__name__)

GenerateFn = Callable[[list[dict[str, str]]], Awaitable[str]]


@dataclass
class EvalReport:
    model_label: str
    total: int
    valid_sql_rate: float
    execution_accuracy: float
    normalized_match_rate: float
    overall_accuracy: float


def load_test_examples(config: LabConfig, limit: int | None = None) -> list[SQLExample]:
    path = Path(config.data.output_dir) / "test.jsonl"
    rows = read_jsonl(path)
    examples = [SQLExample.model_validate(row) for row in rows]
    cap = limit if limit is not None else config.eval.num_examples
    return examples[:cap]


def build_eval_messages(example: SQLExample) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(example)},
    ]


async def evaluate_model(
    examples: list[SQLExample], generate: GenerateFn, model_label: str
) -> EvalReport:
    valid = 0
    execution_correct = 0
    execution_checked = 0
    normalized = 0
    overall = 0
    for example in examples:
        prediction = await generate(build_eval_messages(example))
        score = score_prediction(example, prediction)
        valid += int(score.valid_sql)
        normalized += int(score.normalized_match)
        overall += int(score.correct)
        if score.execution_checked:
            execution_checked += 1
            execution_correct += int(score.execution_match)
    total = len(examples)
    return EvalReport(
        model_label=model_label,
        total=total,
        valid_sql_rate=valid / total if total else 0.0,
        execution_accuracy=execution_correct / execution_checked if execution_checked else 0.0,
        normalized_match_rate=normalized / total if total else 0.0,
        overall_accuracy=overall / total if total else 0.0,
    )


def append_report(config: LabConfig, report: EvalReport) -> None:
    path = Path(config.eval.report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", encoding="utf-8") as handle:
        if is_new:
            handle.write("# Evaluation Report\n\n")
            handle.write("| Model | N | Valid SQL | Exec accuracy | Normalized match | Overall |\n")
            handle.write("|-------|---|-----------|---------------|------------------|--------|\n")
        handle.write(
            f"| {report.model_label} | {report.total} | {report.valid_sql_rate:.3f} "
            f"| {report.execution_accuracy:.3f} | {report.normalized_match_rate:.3f} "
            f"| {report.overall_accuracy:.3f} |\n"
        )
    logger.info("eval_report_appended", model=report.model_label, path=str(path))


def save_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_eval(config: LabConfig, model_path: str | None = None) -> EvalReport:
    from lab.eval.local_generate import build_local_generator

    examples = load_test_examples(config)
    resolved_path = model_path or config.sft.output_dir
    generate = build_local_generator(config, resolved_path)
    report = asyncio.run(evaluate_model(examples, generate, model_label=resolved_path))
    append_report(config, report)
    return report
