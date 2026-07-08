from pathlib import Path

import numpy as np

from lab.config import LabConfig
from lab.data.format import read_jsonl
from lab.logging import get_logger

logger = get_logger(__name__)

_PERCENTILES = (50, 90, 95, 99)


def length_stats(values: list[int]) -> dict[str, float]:
    if not values:
        return {"count": 0.0}
    array = np.asarray(values, dtype=np.float64)
    stats: dict[str, float] = {
        "count": float(array.size),
        "mean": float(array.mean()),
        "max": float(array.max()),
    }
    for percentile in _PERCENTILES:
        stats[f"p{percentile}"] = float(np.percentile(array, percentile))
    return stats


def _word_count(text: str) -> int:
    return len(text.split())


def collect_split_stats(path: Path) -> dict[str, dict[str, float]]:
    rows = read_jsonl(path)
    prompt_lengths: list[int] = []
    completion_lengths: list[int] = []
    for row in rows:
        messages = row.get("messages")
        if isinstance(messages, list):
            prompt_words = sum(
                _word_count(str(message.get("content", "")))
                for message in messages
                if isinstance(message, dict) and message.get("role") != "assistant"
            )
            completion_words = sum(
                _word_count(str(message.get("content", "")))
                for message in messages
                if isinstance(message, dict) and message.get("role") == "assistant"
            )
            prompt_lengths.append(prompt_words)
            completion_lengths.append(completion_words)
        else:
            prompt_lengths.append(
                _word_count(str(row.get("context", ""))) + _word_count(str(row.get("question", "")))
            )
            completion_lengths.append(_word_count(str(row.get("sql", ""))))
    return {
        "prompt_words": length_stats(prompt_lengths),
        "completion_words": length_stats(completion_lengths),
    }


def report_stats(config: LabConfig) -> str:
    output_dir = Path(config.data.output_dir)
    lines = ["# Dataset Statistics", ""]
    for split in ("train", "val", "test"):
        path = output_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        stats = collect_split_stats(path)
        lines.append(f"## {split}")
        lines.append("")
        lines.append("| Field | count | mean | p50 | p90 | p95 | p99 | max |")
        lines.append("|-------|-------|------|-----|-----|-----|-----|-----|")
        for field, values in stats.items():
            if values.get("count", 0) == 0:
                continue
            lines.append(
                f"| {field} | {values['count']:.0f} | {values['mean']:.1f} "
                f"| {values['p50']:.0f} | {values['p90']:.0f} | {values['p95']:.0f} "
                f"| {values['p99']:.0f} | {values['max']:.0f} |"
            )
        lines.append("")
    report = "\n".join(lines)
    (output_dir / "stats.md").write_text(report, encoding="utf-8")
    logger.info("stats_written", path=str(output_dir / "stats.md"))
    return report
