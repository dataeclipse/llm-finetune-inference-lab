from pathlib import Path

import pytest

from lab.config import load_config
from lab.data.prepare import prepare_dataset
from lab.data.schema import SQLExample

pytestmark = pytest.mark.smoke

CONTEXT = "CREATE TABLE t (id INT, v TEXT); INSERT INTO t VALUES (1, 'a'), (2, 'b');"


def make_corpus(count: int) -> list[SQLExample]:
    return [
        SQLExample(
            question=f"Smoke question number {index}?",
            context=CONTEXT,
            sql="SELECT v FROM t WHERE id = 1;",
        )
        for index in range(count)
    ]


def test_sft_smoke_two_steps(tmp_path: Path) -> None:
    pytest.importorskip("trl")
    from lab.training.sft import run_sft

    output_dir = tmp_path / "sft"
    config = load_config(
        [
            f"data.output_dir={(tmp_path / 'data').as_posix()}",
            f"sft.output_dir={output_dir.as_posix()}",
            "data.target_examples=16",
        ]
    )
    prepare_dataset(config, raw=make_corpus(20))
    result_dir = run_sft(config)
    assert Path(result_dir).exists()
    assert (Path(result_dir) / "adapter_model.safetensors").exists()
    assert (Path(result_dir) / "adapter_config.json").exists()


def test_dpo_smoke_two_steps(tmp_path: Path) -> None:
    pytest.importorskip("trl")
    from lab.data.format import write_jsonl
    from lab.training.dpo import run_dpo
    from lab.training.sft import run_sft

    data_dir = tmp_path / "data"
    sft_dir = tmp_path / "sft"
    dpo_dir = tmp_path / "dpo"
    pairs_path = tmp_path / "pairs.jsonl"
    config = load_config(
        [
            f"data.output_dir={data_dir.as_posix()}",
            f"sft.output_dir={sft_dir.as_posix()}",
            f"dpo.output_dir={dpo_dir.as_posix()}",
            f"dpo.pairs_path={pairs_path.as_posix()}",
            "data.target_examples=16",
        ]
    )
    prepare_dataset(config, raw=make_corpus(20))
    run_sft(config)
    pairs = [
        {
            "prompt": [{"role": "user", "content": f"Question {index}"}],
            "chosen": [{"role": "assistant", "content": "SELECT v FROM t WHERE id = 1;"}],
            "rejected": [{"role": "assistant", "content": "SELECT wrong FROM t;"}],
        }
        for index in range(8)
    ]
    write_jsonl(pairs_path, pairs)
    result_dir = run_dpo(config)
    assert (Path(result_dir) / "adapter_model.safetensors").exists()
