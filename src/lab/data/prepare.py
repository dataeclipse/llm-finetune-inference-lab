import json
from pathlib import Path

from lab.config import LabConfig
from lab.data.clean import clean_examples
from lab.data.download import download_raw
from lab.data.format import split_examples, to_messages, write_jsonl
from lab.data.schema import DatasetSummary, SQLExample
from lab.logging import get_logger

logger = get_logger(__name__)


def prepare_dataset(config: LabConfig, raw: list[SQLExample] | None = None) -> DatasetSummary:
    data_config = config.data
    if raw is None:
        raw = download_raw(data_config)
    cleaned, dropped = clean_examples(raw, data_config)
    capped = cleaned[: data_config.target_examples]
    train, val, test = split_examples(
        capped,
        val_fraction=data_config.val_fraction,
        test_fraction=data_config.test_fraction,
        seed=data_config.seed,
    )
    output_dir = Path(data_config.output_dir)
    write_jsonl(output_dir / "train.jsonl", [{"messages": to_messages(item)} for item in train])
    write_jsonl(output_dir / "val.jsonl", [{"messages": to_messages(item)} for item in val])
    write_jsonl(output_dir / "test.jsonl", [item.model_dump() for item in test])
    summary = DatasetSummary(
        raw_examples=len(raw),
        cleaned_examples=len(cleaned),
        train_examples=len(train),
        val_examples=len(val),
        test_examples=len(test),
        dropped=dict(dropped),
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary.model_dump(), indent=2), encoding="utf-8"
    )
    logger.info("dataset_prepared", **summary.model_dump(exclude={"dropped"}))
    return summary
