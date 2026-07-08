import json
import re

from openai import AsyncOpenAI

from lab.data.schema import SQLExample
from lab.logging import get_logger

logger = get_logger(__name__)

JUDGE_SYSTEM = (
    "You are a strict SQL reviewer. Given a database schema, a question, a reference "
    "query and a candidate query, decide whether the candidate answers the question "
    "with semantics equivalent to the reference. Respond with JSON only: "
    '{"correct": true or false, "reason": "<one sentence>"}'
)

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def build_judge_prompt(example: SQLExample, predicted: str) -> str:
    return (
        f"Schema:\n{example.context}\n\n"
        f"Question: {example.question}\n\n"
        f"Reference query:\n{example.sql}\n\n"
        f"Candidate query:\n{predicted}"
    )


def parse_verdict(raw: str) -> bool | None:
    match = _JSON_OBJECT.search(raw)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    verdict = payload.get("correct")
    return verdict if isinstance(verdict, bool) else None


class SQLJudge:
    def __init__(self, client: AsyncOpenAI, model: str) -> None:
        self._client = client
        self._model = model

    async def judge(self, example: SQLExample, predicted: str) -> bool | None:
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user", "content": build_judge_prompt(example, predicted)},
            ],
            temperature=0.0,
            max_tokens=200,
        )
        content = response.choices[0].message.content or ""
        verdict = parse_verdict(content)
        if verdict is None:
            logger.warning("judge_unparseable", question=example.question)
        return verdict
