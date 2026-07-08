from pathlib import Path
from typing import Any

from lab.config import LabConfig
from lab.eval.runner import GenerateFn
from lab.logging import get_logger

logger = get_logger(__name__)


def _is_adapter(model_path: str) -> bool:
    return (Path(model_path) / "adapter_config.json").exists()


def _load_model(model_path: str, dtype: Any) -> Any:
    from transformers import AutoModelForCausalLM

    if _is_adapter(model_path):
        from peft import AutoPeftModelForCausalLM

        return AutoPeftModelForCausalLM.from_pretrained(model_path, dtype=dtype)
    return AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)


def _tokenizer_source(model_path: str) -> str:
    if _is_adapter(model_path) and not (Path(model_path) / "tokenizer_config.json").exists():
        import json

        config = json.loads((Path(model_path) / "adapter_config.json").read_text(encoding="utf-8"))
        base = config.get("base_model_name_or_path")
        if base:
            return str(base)
    return model_path


def build_local_generator(config: LabConfig, model_path: str) -> GenerateFn:
    import torch
    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(_tokenizer_source(model_path))
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model: Any = _load_model(model_path, dtype).to(device)
    model.eval()
    logger.info(
        "local_model_loaded",
        path=model_path,
        device=device,
        adapter=_is_adapter(model_path),
    )

    async def generate(messages: list[dict[str, str]]) -> str:
        encoded = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(device)
        prompt_length = encoded["input_ids"].shape[-1]
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=config.eval.max_new_tokens,
                do_sample=config.eval.temperature > 0,
                temperature=config.eval.temperature or None,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        completion = output[0][prompt_length:]
        return str(tokenizer.decode(completion, skip_special_tokens=True))

    return generate
