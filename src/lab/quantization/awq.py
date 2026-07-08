from pathlib import Path

from lab.config import LabConfig
from lab.exceptions import ExportError
from lab.logging import get_logger

logger = get_logger(__name__)

AWQ_CONFIG = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}


def export_awq_model(
    config: LabConfig,
    model_path: str | None = None,
    output_dir: str | None = None,
) -> str:
    try:
        from awq import AutoAWQForCausalLM
    except ImportError as exc:
        raise ExportError(
            "autoawq is not installed, run: uv sync --extra quant (linux + cuda only)"
        ) from exc
    from transformers import AutoTokenizer

    source = model_path or config.serve.model_path
    target = output_dir or str(Path(source).parent / "awq")
    model = AutoAWQForCausalLM.from_pretrained(source, safetensors=True)
    tokenizer = AutoTokenizer.from_pretrained(source)
    model.quantize(tokenizer, quant_config=AWQ_CONFIG)
    Path(target).mkdir(parents=True, exist_ok=True)
    model.save_quantized(target)
    tokenizer.save_pretrained(target)
    logger.info("awq_exported", source=source, output=target)
    return target
