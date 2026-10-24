"""LLM gateway types: messages, usage, results, errors, pricing, and the paid-call block."""

import math
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import yaml

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls("system", content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls("user", content)

    @classmethod
    def assistant(cls, content: str) -> "Message":
        return cls("assistant", content)


@dataclass(frozen=True)
class Usage:
    """Token counts as billed. `output_tokens` excludes `thinking_tokens`; both are output."""

    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0

    @property
    def billed_output(self) -> int:
        return self.output_tokens + self.thinking_tokens


@dataclass(frozen=True)
class RawResponse:
    """What a provider returns: text and usage, nothing about money."""

    text: str
    usage: Usage
    model: str
    finish_reason: str | None = None


@dataclass(frozen=True)
class LLMResult:
    text: str
    parsed: Any | None
    usage: Usage
    cost_usd: float
    cost_inr: float
    provider: str
    model: str
    service_tier: str | None
    latency_ms: int
    finish_reason: str | None
    run_id: str
    truncated: bool = False  # the provider stopped at max_output_tokens


class Provider(Protocol):
    name: str
    model: str
    is_paid: bool
    service_tier: str | None

    def check_allowed(self) -> None:
        """Raise PaidCallBlocked if this provider may not make a call now."""

    def generate(
        self,
        messages: list[Message],
        json_schema: dict | None,
        max_output_tokens: int,
    ) -> RawResponse: ...


# --- errors ---------------------------------------------------------------------------------------


class LLMError(Exception):
    """Base class of every error the gateway raises on purpose."""


class ProviderError(LLMError):
    """A provider call failed. `status` is the HTTP status if there was one (None: timeout or
    connection failure). Only timeouts, 429 and 5xx are retried."""

    def __init__(self, message: str, status: int | None = None, retry_after: float | None = None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after

    @property
    def retryable(self) -> bool:
        return self.status is None or self.status == 429 or self.status >= 500

    def signature(self) -> str:
        return f"{self.status}:{str(self)[:80]}"


class BudgetExceeded(LLMError):
    pass


class CallCapExceeded(LLMError):
    def __init__(self, scope: Literal["run", "question"], run_id: str, message: str):
        super().__init__(message)
        self.scope = scope
        self.run_id = run_id


class LoopDetected(LLMError):
    def __init__(self, kind: str, run_id: str, message: str):
        super().__init__(message)
        self.kind = kind
        self.run_id = run_id


class PaidCallBlocked(LLMError):
    pass


class LedgerError(LLMError):
    """The ledger could not be written; for a paid call the result is withheld."""


class PricingError(LLMError):
    pass


class InputTooLarge(LLMError):
    pass


class BadStructuredOutput(LLMError):
    """The model answered, but not with JSON that parses (the call was billed and logged)."""

    def __init__(self, message: str, result: "LLMResult"):
        super().__init__(message)
        self.result = result


# --- pricing --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Rate:
    input: float  # USD per `per_tokens` tokens
    output: float


@dataclass
class Pricing:
    """USD prices from config/llm_prices.yaml, never from code. A paid provider/model/tier with no
    entry is an error (the gateway would otherwise spend blind)."""

    rates: Mapping[str, Mapping[str, Mapping[str, Rate]]]
    per_tokens: int = 1_000_000
    retrieved: str | None = None
    source_url: str | None = None
    usd_to_inr: float = 85.7

    @classmethod
    def load(cls, path: str | Path, usd_to_inr: float) -> "Pricing":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        rates = {
            provider: {
                model: {
                    tier: Rate(float(p["input"]), float(p["output"])) for tier, p in tiers.items()
                }
                for model, tiers in models.items()
            }
            for provider, models in (raw.get("providers") or {}).items()
        }
        return cls(
            rates,
            int(raw.get("per_tokens", 1_000_000)),
            str(raw.get("retrieved")),
            raw.get("source_url"),
            usd_to_inr,
        )

    def rate(self, provider: str, model: str, tier: str | None) -> Rate:
        try:
            return self.rates[provider][model][tier or "standard"]
        except KeyError:
            raise PricingError(
                f"no price for {provider}/{model} tier {tier or 'standard'!r} in the prices file; "
                "refusing to make a paid call without a price"
            ) from None

    def cost_usd(self, provider: str, model: str, tier: str | None, usage: Usage) -> float:
        r = self.rate(provider, model, tier)
        return (usage.input_tokens * r.input + usage.billed_output * r.output) / self.per_tokens

    def worst_case_usd(
        self, provider: str, model: str, tier: str | None, est_input: int, max_output: int
    ) -> float:
        """Upper bound of one call: estimated input plus the whole output allowance (thinking
        tokens count toward max_output_tokens)."""
        return self.cost_usd(provider, model, tier, Usage(est_input, max_output, 0))

    def to_inr(self, usd: float) -> float:
        return usd * self.usd_to_inr


def estimate_tokens(messages: list[Message]) -> int:
    """Conservative input estimate (3 characters per token; real text runs 3.5 to 4.5)."""
    return math.ceil(sum(len(m.content) for m in messages) / 3) + 4 * len(messages)


# --- the paid-call block --------------------------------------------------------------------------


def paid_block_reason(
    allow_paid: bool,
    env: Mapping[str, str] | None = None,
    modules: Mapping[str, Any] | None = None,
) -> str | None:
    """Why paid calls are not allowed right now, or None. CI and test runs are blocked whatever
    the settings say; elsewhere a paid call needs LLM_ALLOW_PAID=true."""
    env = os.environ if env is None else env
    modules = sys.modules if modules is None else modules
    if env.get("CI", "").lower() in ("1", "true", "yes"):
        return "CI is set: CI never makes paid calls"
    if "PYTEST_CURRENT_TEST" in env or "pytest" in modules:
        return "running under pytest: tests never make paid calls"
    if not allow_paid:
        return "paid calls are off: set LLM_ALLOW_PAID=true to allow them"
    return None
