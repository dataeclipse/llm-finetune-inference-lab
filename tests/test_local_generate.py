from unittest.mock import MagicMock, patch

import pytest

torch = pytest.importorskip("torch")

from lab.config import load_config
from lab.eval import local_generate


class FakeEncoding(dict):
    def to(self, device: str) -> "FakeEncoding":
        return self


async def test_generate_passes_dict_kwargs_and_slices_prompt() -> None:
    config = load_config(["eval.max_new_tokens=8"])

    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = FakeEncoding(input_ids=torch.tensor([[1, 2, 3]]))
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 0
    tokenizer.decode.return_value = "SELECT 1;"

    model = MagicMock()
    model.to.return_value = model
    model.generate.return_value = torch.tensor([[1, 2, 3, 9, 9]])

    with (
        patch.object(local_generate, "_load_model", return_value=model),
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
        patch.object(torch.cuda, "is_available", return_value=False),
    ):
        generate = local_generate.build_local_generator(config, "some/model")
        result = await generate([{"role": "user", "content": "count rows"}])

    assert result == "SELECT 1;"
    assert tokenizer.apply_chat_template.call_args.kwargs["return_dict"] is True
    generate_kwargs = model.generate.call_args.kwargs
    assert "input_ids" in generate_kwargs
    decoded = tokenizer.decode.call_args.args[0]
    assert decoded.tolist() == [9, 9]
