import math
from pathlib import Path
from typing import Any

from lab.config import LabConfig
from lab.data.format import read_jsonl
from lab.logging import get_logger

logger = get_logger(__name__)


def compute_perplexity(config: LabConfig, model_path: str, split: str = "val") -> float:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model: Any = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    rows = read_jsonl(Path(config.data.output_dir) / f"{split}.jsonl")
    total_loss = 0.0
    total_tokens = 0
    for row in rows:
        messages = row["messages"]
        encoded = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            truncation=True,
            max_length=config.model.max_seq_length,
        ).to(device)
        input_ids = encoded["input_ids"]
        with torch.no_grad():
            output = model(input_ids, labels=input_ids)
        count = input_ids.shape[-1] - 1
        total_loss += float(output.loss) * count
        total_tokens += count
    perplexity = math.exp(total_loss / total_tokens) if total_tokens else float("inf")
    logger.info("perplexity_computed", model=model_path, split=split, value=perplexity)
    return perplexity
