"""Gemini over the REST API (httpx, no SDK): models.generateContent.

Field names were checked against the live API (invalid values return 400 and are not billed):
serviceTier ('standard' | 'flex' | 'priority'), generationConfig.thinkingConfig.thinkingLevel
('minimal' | 'low' | 'medium' | 'high') or .thinkingBudget (integer), responseMimeType,
responseJsonSchema, maxOutputTokens. Thinking tokens count toward maxOutputTokens and are billed as
output; usageMetadata.thoughtsTokenCount carries them.

The paid-call block is in `_send`, the only place a request leaves the process, so no caller can go
around it. A provider built on an httpx.MockTransport cannot reach the network and is not blocked
(that is how the request and response handling is tested).
"""

from collections.abc import Callable

import httpx

from rflens.config import LlmSettings
from rflens.llm.base import (
    Message,
    PaidCallBlocked,
    ProviderError,
    RawResponse,
    Usage,
    paid_block_reason,
)

BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
FLEX_MIN_TIMEOUT_S = 600  # Google: flex requests can queue for minutes; use 10 minutes or more


class GeminiProvider:
    name = "gemini"
    is_paid = True

    def __init__(
        self,
        settings: LlmSettings,
        model: str | None = None,
        transport: httpx.BaseTransport | None = None,
        block_reason: Callable[[], str | None] | None = None,
    ):
        self.model = model or settings.llm_model
        self.service_tier = settings.llm_service_tier
        self._key = settings.gemini_api_key
        self._thinking = settings.llm_thinking_budget.strip().lower()
        timeout = settings.llm_timeout_s
        if self.service_tier == "flex":
            timeout = max(timeout, FLEX_MIN_TIMEOUT_S)
        self._client = httpx.Client(timeout=timeout, transport=transport)
        self._offline = isinstance(transport, httpx.MockTransport)
        self._block = block_reason or (lambda: paid_block_reason(settings.llm_allow_paid))

    # --- the block --------------------------------------------------------------------------------

    def check_allowed(self) -> None:
        if self._offline:
            return
        reason = self._block()
        if reason:
            raise PaidCallBlocked(f"paid Gemini call blocked: {reason}")

    # --- request ----------------------------------------------------------------------------------

    def _thinking_config(self) -> dict:
        if self._thinking.lstrip("-").isdigit():
            return {"thinkingBudget": int(self._thinking)}
        return {"thinkingLevel": self._thinking}

    def build_body(
        self, messages: list[Message], json_schema: dict | None, max_output_tokens: int
    ) -> dict:
        system = [m.content for m in messages if m.role == "system"]
        generation: dict = {
            "maxOutputTokens": max_output_tokens,
            "thinkingConfig": self._thinking_config(),
        }
        if json_schema is not None:
            generation["responseMimeType"] = "application/json"
            generation["responseJsonSchema"] = json_schema
        body: dict = {
            "contents": [
                {
                    "role": "model" if m.role == "assistant" else "user",
                    "parts": [{"text": m.content}],
                }
                for m in messages
                if m.role != "system"
            ],
            "generationConfig": generation,
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system)}]}
        if self.service_tier != "standard":
            body["serviceTier"] = self.service_tier
        return body

    def _send(self, body: dict) -> httpx.Response:
        self.check_allowed()  # the only exit to the network
        if self._key is None:
            raise ProviderError("GEMINI_API_KEY is not set", status=401)
        try:
            return self._client.post(
                f"{BASE_URL}/models/{self.model}:generateContent",
                headers={"x-goog-api-key": self._key.get_secret_value()},
                json=body,
            )
        except httpx.TimeoutException as e:
            raise ProviderError(f"Gemini request timed out ({type(e).__name__})") from e
        except httpx.TransportError as e:
            raise ProviderError(f"Gemini connection failed ({type(e).__name__})") from e

    # --- response ---------------------------------------------------------------------------------

    def generate(
        self, messages: list[Message], json_schema: dict | None, max_output_tokens: int
    ) -> RawResponse:
        response = self._send(self.build_body(messages, json_schema, max_output_tokens))
        if response.status_code != 200:
            retry_after = response.headers.get("retry-after")
            raise ProviderError(
                f"Gemini HTTP {response.status_code}: {_error_message(response)}",
                status=response.status_code,
                retry_after=float(retry_after) if retry_after and retry_after.isdigit() else None,
            )
        data = response.json()
        candidates = data.get("candidates") or []
        if not candidates:
            reason = (data.get("promptFeedback") or {}).get("blockReason", "no candidates")
            raise ProviderError(f"Gemini returned no answer ({reason})", status=400)
        parts = (candidates[0].get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        u = data.get("usageMetadata") or {}
        return RawResponse(
            text=text,
            usage=Usage(
                input_tokens=u.get("promptTokenCount", 0),
                output_tokens=u.get("candidatesTokenCount", 0),
                thinking_tokens=u.get("thoughtsTokenCount", 0),
            ),
            model=data.get("modelVersion") or self.model,
            finish_reason=candidates[0].get("finishReason"),
        )


def _error_message(response: httpx.Response) -> str:
    try:
        return str(response.json()["error"]["message"])[:300]
    except Exception:
        return response.text[:300]
