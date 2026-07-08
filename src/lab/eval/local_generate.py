from typing import Any

from lab.config import LabConfig
from lab.eval.runner import GenerateFn
from lab.logging import get_logger

logger = get_logger(__name__)


def build_local_generator(config: LabConfig, model_path: str) -> GenerateFn:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model: Any = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    logger.info("local_model_loaded", path=model_path, device=device)

    async def generate(messages: list[dict[str, str]]) -> str:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            output = model.generate(
                inputs,
                max_new_tokens=config.eval.max_new_tokens,
                do_sample=config.eval.temperature > 0,
                temperature=config.eval.temperature or None,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        completion = output[0][inputs.shape[-1] :]
        return str(tokenizer.decode(completion, skip_special_tokens=True))

    return generate
