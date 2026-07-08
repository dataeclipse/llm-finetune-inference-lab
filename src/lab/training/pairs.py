import asyncio
from pathlib import Path
from typing import cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from lab.config import LabConfig
from lab.data.format import SYSTEM_PROMPT, build_user_prompt, read_jsonl, write_jsonl
from lab.data.schema import SQLExample
from lab.eval.runner import GenerateFn
from lab.eval.sql_exec import extract_sql, score_prediction
from lab.logging import get_logger

logger = get_logger(__name__)


def _prompt_messages(example: SQLExample) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(example)},
    ]


async def collect_pairs(
    examples: list[SQLExample],
    generate: GenerateFn,
    max_pairs: int,
) -> list[dict[str, object]]:
    pairs: list[dict[str, object]] = []
    for example in examples:
        if len(pairs) >= max_pairs:
            break
        raw = await generate(_prompt_messages(example))
        score = score_prediction(example, raw)
        if score.correct:
            continue
        rejected = extract_sql(raw) or raw.strip()
        if not rejected or rejected == example.sql:
            continue
        pairs.append(
            {
                "prompt": _prompt_messages(example),
                "chosen": [{"role": "assistant", "content": example.sql}],
                "rejected": [{"role": "assistant", "content": rejected}],
            }
        )
    logger.info("dpo_pairs_collected", pairs=len(pairs), scanned=len(examples))
    return pairs


def load_train_examples(config: LabConfig) -> list[SQLExample]:
    rows = read_jsonl(Path(config.data.output_dir) / "train.jsonl")
    examples: list[SQLExample] = []
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            continue
        user_content = str(messages[1].get("content", ""))
        assistant_content = str(messages[2].get("content", ""))
        if "\n\nQuestion: " not in user_content:
            continue
        context_part, question_part = user_content.split("\n\nQuestion: ", 1)
        examples.append(
            SQLExample(
                question=question_part.strip(),
                context=context_part.removeprefix("Database schema:\n").strip(),
                sql=assistant_content.strip(),
            )
        )
    return examples


def build_api_generator(base_url: str, model: str, max_tokens: int) -> GenerateFn:
    client = AsyncOpenAI(base_url=base_url, api_key="not-required")

    async def generate(messages: list[dict[str, str]]) -> str:
        response = await client.chat.completions.create(
            model=model,
            messages=cast(list[ChatCompletionMessageParam], messages),
            temperature=0.7,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    return generate


def generate_pairs(
    config: LabConfig,
    base_url: str,
    model: str,
    generate: GenerateFn | None = None,
) -> int:
    examples = load_train_examples(config)
    generator = generate or build_api_generator(base_url, model, config.eval.max_new_tokens)
    pairs = asyncio.run(collect_pairs(examples, generator, config.dpo.max_pairs))
    write_jsonl(Path(config.dpo.pairs_path), pairs)
    return len(pairs)
