"""Ollama chat (local, free): POST /api/chat, non-streaming.

`think: false` keeps qwen3 models from spending the output allowance on a reasoning trace.
`format` takes a JSON schema for structured output.
"""

import httpx

from rflens.config import LlmSettings
from rflens.llm.base import Message, ProviderError, RawResponse, Usage


class OllamaProvider:
    name = "ollama"
    is_paid = False
    service_tier = None

    def __init__(
        self,
        settings: LlmSettings,
        model: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        self.model = model or settings.llm_ollama_model
        self._base = settings.ollama_base_url.rstrip("/")
        self._temperature = settings.llm_temperature
        self._client = httpx.Client(timeout=settings.llm_ollama_timeout_s, transport=transport)

    def check_allowed(self) -> None:
        return None  # free and local

    def generate(
        self, messages: list[Message], json_schema: dict | None, max_output_tokens: int
    ) -> RawResponse:
        body: dict = {
            "model": self.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "think": False,
            "options": {"num_predict": max_output_tokens},
        }
        if self._temperature is not None:
            body["options"]["temperature"] = self._temperature
        if json_schema is not None:
            body["format"] = json_schema
        try:
            response = self._client.post(f"{self._base}/api/chat", json=body)
        except httpx.TimeoutException as e:
            raise ProviderError(f"Ollama request timed out ({type(e).__name__})") from e
        except httpx.TransportError as e:
            raise ProviderError(f"Ollama is not reachable ({type(e).__name__})") from e
        if response.status_code != 200:
            raise ProviderError(
                f"Ollama HTTP {response.status_code}: {response.text[:200]}",
                status=response.status_code,
            )
        data = response.json()
        return RawResponse(
            text=(data.get("message") or {}).get("content", ""),
            usage=Usage(
                input_tokens=data.get("prompt_eval_count", 0),
                output_tokens=data.get("eval_count", 0),
            ),
            model=data.get("model") or self.model,
            finish_reason=data.get("done_reason"),
        )
