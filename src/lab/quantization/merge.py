from pathlib import Path

from lab.config import LabConfig
from lab.exceptions import ExportError
from lab.logging import get_logger

logger = get_logger(__name__)


def resolve_adapter(config: LabConfig) -> str:
    dpo_dir = Path(config.dpo.output_dir)
    if (dpo_dir / "adapter_config.json").exists():
        return str(dpo_dir)
    sft_dir = Path(config.sft.output_dir)
    if (sft_dir / "adapter_config.json").exists():
        return str(sft_dir)
    raise ExportError("no trained adapter found in dpo or sft output directories")


def merge_adapter(
    config: LabConfig,
    adapter_path: str | None = None,
    output_dir: str | None = None,
) -> str:
    import torch
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer

    source = adapter_path or resolve_adapter(config)
    target = output_dir or config.serve.model_path
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoPeftModelForCausalLM.from_pretrained(source, dtype=dtype)
    merged = model.merge_and_unload()
    Path(target).mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(target, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(source)
    tokenizer.save_pretrained(target)
    logger.info("adapter_merged", adapter=source, output=target)
    return target
