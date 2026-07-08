import os
import subprocess
import sys
from pathlib import Path

from lab.config import LabConfig
from lab.exceptions import ExportError
from lab.logging import get_logger

logger = get_logger(__name__)


def _quantize_binary(llama_cpp_dir: Path) -> Path | None:
    for candidate in (
        llama_cpp_dir / "build" / "bin" / "llama-quantize",
        llama_cpp_dir / "build" / "bin" / "llama-quantize.exe",
        llama_cpp_dir / "llama-quantize",
    ):
        if candidate.exists():
            return candidate
    return None


def export_gguf_model(
    config: LabConfig,
    model_path: str | None = None,
    output_dir: str | None = None,
    quant_type: str = "Q4_K_M",
) -> str:
    llama_cpp_env = os.environ.get("LLAMA_CPP_DIR", "")
    if not llama_cpp_env:
        raise ExportError("LLAMA_CPP_DIR is not set, clone https://github.com/ggml-org/llama.cpp")
    llama_cpp_dir = Path(llama_cpp_env)
    converter = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not converter.exists():
        raise ExportError(f"converter script not found: {converter}")

    source = model_path or config.serve.model_path
    target = Path(output_dir) if output_dir else Path(source).parent / "gguf"
    target.mkdir(parents=True, exist_ok=True)
    f16_path = target / "model-f16.gguf"
    subprocess.run(
        [sys.executable, str(converter), source, "--outfile", str(f16_path), "--outtype", "f16"],
        check=True,
    )
    logger.info("gguf_converted", output=str(f16_path))

    quantizer = _quantize_binary(llama_cpp_dir)
    if quantizer is None:
        logger.warning("gguf_quantize_skipped", reason="llama-quantize binary not found")
        return str(f16_path)
    quant_path = target / f"model-{quant_type.lower()}.gguf"
    subprocess.run([str(quantizer), str(f16_path), str(quant_path), quant_type], check=True)
    logger.info("gguf_quantized", output=str(quant_path), quant_type=quant_type)
    return str(quant_path)
