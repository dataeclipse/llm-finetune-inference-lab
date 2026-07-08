import os
from pathlib import Path
from typing import Any

from lab.config import LabConfig, LoraSettings, ModelConfig, WandbSettings
from lab.logging import get_logger

logger = get_logger(__name__)


def build_quantization_config(config: ModelConfig) -> Any | None:
    import torch
    from transformers import BitsAndBytesConfig

    if not config.load_in_4bit:
        return None
    if not torch.cuda.is_available():
        logger.warning("quantization_disabled", reason="cuda unavailable")
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_tokenizer(model_name: str) -> Any:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(config: ModelConfig) -> Any:
    import torch
    from transformers import AutoModelForCausalLM

    quantization = build_quantization_config(config)
    kwargs: dict[str, Any] = {
        "attn_implementation": config.attn_implementation,
        "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    }
    if quantization is not None:
        kwargs["quantization_config"] = quantization
        kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(config.name, **kwargs)
    logger.info("base_model_loaded", name=config.name, quantized=quantization is not None)
    return model


def build_lora_config(settings: LoraSettings) -> Any:
    from peft import LoraConfig

    return LoraConfig(
        r=settings.r,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        target_modules=list(settings.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )


def resolve_checkpoint(output_dir: str, resume: bool) -> str | None:
    if not resume or not Path(output_dir).is_dir():
        return None
    from transformers.trainer_utils import get_last_checkpoint

    checkpoint: str | None = get_last_checkpoint(output_dir)
    if checkpoint:
        logger.info("resuming_from_checkpoint", checkpoint=checkpoint)
    return checkpoint


def configure_wandb(settings: WandbSettings) -> list[str]:
    if not settings.enabled:
        os.environ["WANDB_MODE"] = "disabled"
        return []
    os.environ.setdefault("WANDB_PROJECT", settings.project)
    if settings.run_name:
        os.environ.setdefault("WANDB_NAME", settings.run_name)
    return ["wandb"]


def load_split_dataset(config: LabConfig, split: str) -> Any:
    from datasets import load_dataset

    path = Path(config.data.output_dir) / f"{split}.jsonl"
    return load_dataset("json", data_files=str(path), split="train")
