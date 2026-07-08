import asyncio
from dataclasses import dataclass
from pathlib import Path

from lab.config import LabConfig
from lab.eval.runner import build_eval_messages, load_test_examples
from lab.eval.sql_exec import score_prediction
from lab.logging import get_logger
from lab.serving.client import InferenceClient, OpenAICompatClient

logger = get_logger(__name__)


@dataclass
class ABResult:
    model_a: str
    model_b: str
    total: int
    wins_a: int
    wins_b: int
    ties: int
    accuracy_a: float
    accuracy_b: float


async def compare_models(
    config: LabConfig,
    client_a: InferenceClient,
    client_b: InferenceClient,
    label_a: str,
    label_b: str,
) -> ABResult:
    examples = load_test_examples(config)
    wins_a = wins_b = ties = correct_a = correct_b = 0
    for example in examples:
        messages = build_eval_messages(example)
        result_a, result_b = await asyncio.gather(
            client_a.complete(messages, max_tokens=config.eval.max_new_tokens),
            client_b.complete(messages, max_tokens=config.eval.max_new_tokens),
        )
        score_a = score_prediction(example, result_a.text).correct
        score_b = score_prediction(example, result_b.text).correct
        correct_a += int(score_a)
        correct_b += int(score_b)
        if score_a and not score_b:
            wins_a += 1
        elif score_b and not score_a:
            wins_b += 1
        else:
            ties += 1
    total = len(examples)
    return ABResult(
        model_a=label_a,
        model_b=label_b,
        total=total,
        wins_a=wins_a,
        wins_b=wins_b,
        ties=ties,
        accuracy_a=correct_a / total if total else 0.0,
        accuracy_b=correct_b / total if total else 0.0,
    )


def write_ab_report(result: ABResult, path: str = "reports/ab_test.md") -> None:
    report_file = Path(path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# A/B Comparison",
        "",
        f"Examples: {result.total}",
        "",
        "| Model | Accuracy | Wins |",
        "|-------|----------|------|",
        f"| {result.model_a} | {result.accuracy_a:.3f} | {result.wins_a} |",
        f"| {result.model_b} | {result.accuracy_b:.3f} | {result.wins_b} |",
        f"| ties | | {result.ties} |",
    ]
    report_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_ab_test(
    config: LabConfig,
    base_url_a: str,
    base_url_b: str,
    model_a: str,
    model_b: str,
    client_a: InferenceClient | None = None,
    client_b: InferenceClient | None = None,
) -> ABResult:
    active_a = client_a or OpenAICompatClient(base_url=base_url_a, model=model_a)
    active_b = client_b or OpenAICompatClient(base_url=base_url_b, model=model_b)
    result = asyncio.run(compare_models(config, active_a, active_b, model_a, model_b))
    write_ab_report(result)
    logger.info(
        "ab_test_finished",
        accuracy_a=result.accuracy_a,
        accuracy_b=result.accuracy_b,
        wins_a=result.wins_a,
        wins_b=result.wins_b,
    )
    return result
