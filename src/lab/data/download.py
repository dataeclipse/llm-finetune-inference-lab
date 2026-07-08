from typing import Any

from lab.config import DataConfig
from lab.data.schema import SQLExample
from lab.logging import get_logger

logger = get_logger(__name__)

_FIELD_ALIASES = {
    "question": ("sql_prompt", "question", "prompt"),
    "context": ("sql_context", "context", "schema"),
    "sql": ("sql", "query", "answer"),
    "complexity": ("sql_complexity", "complexity"),
    "domain": ("domain",),
}


def _pick(row: dict[str, Any], field: str) -> str:
    for alias in _FIELD_ALIASES[field]:
        value = row.get(alias)
        if value:
            return str(value)
    return ""


def row_to_example(row: dict[str, Any]) -> SQLExample:
    return SQLExample(
        question=_pick(row, "question"),
        context=_pick(row, "context"),
        sql=_pick(row, "sql"),
        complexity=_pick(row, "complexity"),
        domain=_pick(row, "domain"),
    )


def download_raw(config: DataConfig, limit: int | None = None) -> list[SQLExample]:
    from datasets import load_dataset

    take = limit if limit is not None else config.target_examples * 3
    dataset = load_dataset(config.dataset_name, split=f"train[:{take}]")
    examples = [row_to_example(dict(row)) for row in dataset]
    logger.info("dataset_downloaded", name=config.dataset_name, rows=len(examples))
    return examples
