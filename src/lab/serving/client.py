import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam


@dataclass
class CompletionResult:
    text: str
    latency_seconds: float
    completion_tokens: int


class InferenceClient(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> CompletionResult: ...


class OpenAICompatClient(InferenceClient):
    def __init__(self, base_url: str, model: str, api_key: str = "not-required") -> None:
        self._client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    async def complete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 256,
        temperature: float = 0.0,
    ) -> CompletionResult:
        started = time.perf_counter()
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=cast(list[ChatCompletionMessageParam], messages),
            max_tokens=max_tokens,
            temperature=temperature,
        )
        elapsed = time.perf_counter() - started
        usage = response.usage
        return CompletionResult(
            text=response.choices[0].message.content or "",
            latency_seconds=elapsed,
            completion_tokens=usage.completion_tokens if usage else 0,
        )
