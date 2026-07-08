from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lab.config import load_config
from lab.exceptions import ExportError
from lab.quantization.awq import export_awq_model
from lab.quantization.bench_quant import QuantBenchRow, run_quant_bench
from lab.quantization.gguf import export_gguf_model
from lab.quantization.merge import resolve_adapter


def test_resolve_adapter_prefers_dpo(tmp_path: Path) -> None:
    sft_dir = tmp_path / "sft"
    dpo_dir = tmp_path / "dpo"
    for directory in (sft_dir, dpo_dir):
        directory.mkdir()
        (directory / "adapter_config.json").write_text("{}", encoding="utf-8")
    config = load_config(
        [f"sft.output_dir={sft_dir.as_posix()}", f"dpo.output_dir={dpo_dir.as_posix()}"]
    )
    assert resolve_adapter(config) == str(dpo_dir)


def test_resolve_adapter_falls_back_to_sft(tmp_path: Path) -> None:
    sft_dir = tmp_path / "sft"
    sft_dir.mkdir()
    (sft_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    config = load_config(
        [
            f"sft.output_dir={sft_dir.as_posix()}",
            f"dpo.output_dir={(tmp_path / 'missing').as_posix()}",
        ]
    )
    assert resolve_adapter(config) == str(sft_dir)


def test_resolve_adapter_raises_without_artifacts(tmp_path: Path) -> None:
    config = load_config(
        [
            f"sft.output_dir={(tmp_path / 'a').as_posix()}",
            f"dpo.output_dir={(tmp_path / 'b').as_posix()}",
        ]
    )
    with pytest.raises(ExportError, match="no trained adapter"):
        resolve_adapter(config)


def test_awq_requires_dependency() -> None:
    config = load_config()
    with pytest.raises(ExportError, match="autoawq is not installed"):
        export_awq_model(config, model_path="whatever")


def test_gguf_requires_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)
    config = load_config()
    with pytest.raises(ExportError, match="LLAMA_CPP_DIR"):
        export_gguf_model(config, model_path="whatever")


def test_gguf_requires_converter(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LLAMA_CPP_DIR", str(tmp_path))
    config = load_config()
    with pytest.raises(ExportError, match="converter script not found"):
        export_gguf_model(config, model_path="whatever")


def test_gguf_runs_converter_and_quantizer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    llama_dir = tmp_path / "llama.cpp"
    (llama_dir / "build" / "bin").mkdir(parents=True)
    (llama_dir / "convert_hf_to_gguf.py").write_text("", encoding="utf-8")
    (llama_dir / "build" / "bin" / "llama-quantize").write_text("", encoding="utf-8")
    monkeypatch.setenv("LLAMA_CPP_DIR", str(llama_dir))
    config = load_config()
    with patch("lab.quantization.gguf.subprocess.run", MagicMock()) as run_mock:
        result = export_gguf_model(
            config, model_path=str(tmp_path / "merged"), output_dir=str(tmp_path / "out")
        )
    assert run_mock.call_count == 2
    convert_command = run_mock.call_args_list[0].args[0]
    assert str(llama_dir / "convert_hf_to_gguf.py") in convert_command
    quant_command = run_mock.call_args_list[1].args[0]
    assert quant_command[-1] == "Q4_K_M"
    assert result.endswith("model-q4_k_m.gguf")


def test_run_quant_bench_writes_report(tmp_path: Path) -> None:
    config = load_config()
    report_path = tmp_path / "quant.md"
    measurements = {"fp16": (3.2, 40.0), "awq-4bit": (3.4, 95.0)}

    def fake_measure(path: str) -> tuple[float, float]:
        return measurements[path.split("/")[-1]]

    rows = run_quant_bench(
        config,
        model_paths={"fp16": "models/fp16", "awq-4bit": "models/awq-4bit"},
        measure=fake_measure,
        report_path=str(report_path),
    )
    assert rows == [
        QuantBenchRow(label="fp16", perplexity=3.2, tokens_per_second=40.0),
        QuantBenchRow(label="awq-4bit", perplexity=3.4, tokens_per_second=95.0),
    ]
    content = report_path.read_text(encoding="utf-8")
    assert "| fp16 | 3.200 | 40.0 |" in content
    assert "| awq-4bit | 3.400 | 95.0 |" in content
