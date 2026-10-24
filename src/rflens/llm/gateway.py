"""The LLM gateway: one entry point over the providers, with budget and loop protection.

    gateway = build_gateway()
    with gateway.run("eval-smoke") as run:          # one run = one command; prints a summary
        result = run.generate([Message.user("...")], purpose="sql-plan", question_id="q07")

What happens to a call, in order
1. input size check; repeat-prompt detector (third identical prompt in a run stops the run);
2. per attempt: paid-call block, per-run and per-question caps, rate limiter, then the ledger
   reservation (budget check, pending row with the worst-case cost);
3. the provider call; on timeout / 429 / 5xx up to LLM_MAX_RETRIES retries with exponential
   backoff (other errors are never retried); after retries are exhausted by a retryable error, one
   attempt on the first fallback provider (never back to the primary);
4. the ledger row is updated with real tokens and cost before the result is returned; if that
   fails the result is withheld;
5. a spend warning when the total crosses 50% or 80% of LLM_BUDGET_USD.
"""

import hashlib
import json
import random
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from rflens.config import LlmSettings, get_llm_settings
from rflens.llm.base import (
    BadStructuredOutput,
    InputTooLarge,
    LLMError,
    LLMResult,
    Message,
    Pricing,
    Provider,
    ProviderError,
    Usage,
    estimate_tokens,
)
from rflens.llm.guards import RateLimiter, RunGuard
from rflens.llm.ledger import Ledger

WARN_FRACTIONS = (0.5, 0.8)
MAX_BACKOFF_S = 30.0
_TRUNCATED = {"MAX_TOKENS", "length"}


def prompt_hash(messages: list[Message], json_schema: dict | None) -> str:
    payload = json.dumps(
        [[m.role, m.content] for m in messages] + [json_schema], sort_keys=True, default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_json(text: str):
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    return json.loads(cleaned)


class Gateway:
    def __init__(
        self,
        settings: LlmSettings,
        providers: dict[str, Provider],
        ledger: Ledger,
        pricing: Pricing,
        rate_limiter: RateLimiter | None = None,
        sleep: Callable[[float], None] = time.sleep,
        emit: Callable[[str], None] = lambda line: print(line, flush=True),
        backoff_base_s: float = 1.0,
    ):
        self.settings = settings
        self.providers = providers
        self.ledger = ledger
        self.pricing = pricing
        self.rate = rate_limiter or RateLimiter(settings.llm_max_calls_per_minute, sleep=sleep)
        self.sleep = sleep
        self.emit = emit
        self.backoff_base_s = backoff_base_s

    def run(self, name: str = "run") -> "LLMRun":
        return LLMRun(self, f"{name}-{uuid.uuid4().hex[:8]}")

    def chain(self, primary: str | None = None) -> list[Provider]:
        """The primary provider, then at most one fallback (the first configured one that is not
        the primary and exists). The primary never appears twice."""
        first = self.providers[primary or self.settings.llm_provider]
        for name in self.settings.fallback_providers():
            if name != first.name and name in self.providers:
                return [first, self.providers[name]]
        return [first]

    def warn_on_crossing(self, spent_before: float, spent_after: float) -> None:
        budget = self.settings.llm_budget_usd
        for fraction in WARN_FRACTIONS:
            if spent_before < fraction * budget <= spent_after:
                self.emit(
                    f"WARNING: LLM spend crossed {fraction:.0%} of LLM_BUDGET_USD: "
                    f"${spent_after:.4f} of ${budget:.2f}"
                )


@dataclass
class LLMRun:
    """One command's worth of calls: its own counters, run_id and end-of-run summary."""

    gateway: Gateway
    run_id: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    cost_usd: float = 0.0
    guard: RunGuard = field(init=False)

    def __post_init__(self) -> None:
        s = self.gateway.settings
        self.guard = RunGuard(self.run_id, s.llm_max_calls_per_run, s.llm_max_calls_per_question)

    def __enter__(self) -> "LLMRun":
        return self

    def __exit__(self, *exc) -> None:
        self.gateway.emit(self.summary())

    # --- the call ---------------------------------------------------------------------------------

    def generate(
        self,
        messages: list[Message],
        *,
        purpose: str,
        json_schema: dict | None = None,
        question_id: str | None = None,
        max_output_tokens: int | None = None,
        provider: str | None = None,
    ) -> LLMResult:
        gw, s = self.gateway, self.gateway.settings
        max_out = min(max_output_tokens or s.llm_max_output_tokens, s.llm_max_output_tokens)
        est_in = estimate_tokens(messages)
        if est_in > s.llm_max_input_tokens:
            raise InputTooLarge(
                f"prompt is about {est_in} tokens; LLM_MAX_INPUT_TOKENS is {s.llm_max_input_tokens}"
            )
        digest = prompt_hash(messages, json_schema)
        self.guard.note_prompt(digest)

        retries_left = s.llm_max_retries
        attempt = 0
        last_error: ProviderError | None = None
        for prov in gw.chain(provider):
            while True:
                prov.check_allowed()
                self.guard.count_attempt(question_id)
                gw.rate.acquire()
                attempt += 1
                paid = prov.is_paid
                worst = (
                    gw.pricing.worst_case_usd(
                        prov.name, prov.model, prov.service_tier, est_in, max_out
                    )
                    if paid
                    else 0.0
                )
                row_id = gw.ledger.reserve(
                    run_id=self.run_id,
                    question_id=question_id,
                    attempt=attempt,
                    provider=prov.name,
                    model=prov.model,
                    service_tier=prov.service_tier,
                    purpose=purpose,
                    prompt_hash=digest,
                    est_input_tokens=est_in,
                    worst_case_usd=worst,
                    paid=paid,
                )
                self.calls += 1
                started = time.perf_counter()
                try:
                    raw = prov.generate(messages, json_schema, max_out)
                except ProviderError as e:
                    gw.ledger.finish(
                        row_id,
                        status="error",
                        usage=None,
                        cost_usd=0.0,
                        latency_ms=int((time.perf_counter() - started) * 1000),
                        error=str(e),
                    )
                    last_error = e
                    if e.retryable and retries_left > 0:
                        retries_left -= 1
                        gw.sleep(self._backoff(s.llm_max_retries - retries_left - 1, e))
                        continue
                    break
                except Exception as e:  # a bug, not a provider failure: log it and let it out
                    gw.ledger.finish(
                        row_id,
                        status="error",
                        usage=None,
                        cost_usd=0.0,
                        latency_ms=int((time.perf_counter() - started) * 1000),
                        error=f"{type(e).__name__}: {e}",
                    )
                    raise
                return self._succeed(prov, raw, row_id, started, json_schema)
            if last_error is None or not last_error.retryable:
                break  # a request that was refused would be refused again: no fallback
        assert last_error is not None
        self.guard.note_failure(last_error.signature())
        raise last_error

    # --- helpers ----------------------------------------------------------------------------------

    def _backoff(self, retry_index: int, error: ProviderError) -> float:
        delay = min(self.gateway.backoff_base_s * 2**retry_index, MAX_BACKOFF_S)
        if error.retry_after:
            delay = max(delay, min(error.retry_after, MAX_BACKOFF_S))
        return delay + random.uniform(0, delay * 0.1)

    def _succeed(self, prov: Provider, raw, row_id: int, started: float, json_schema) -> LLMResult:
        gw = self.gateway
        latency_ms = int((time.perf_counter() - started) * 1000)
        usage: Usage = raw.usage
        cost = (
            gw.pricing.cost_usd(prov.name, prov.model, prov.service_tier, usage)
            if (prov.is_paid)
            else 0.0
        )
        gw.ledger.finish(  # before the result leaves: LedgerError withholds it
            row_id, status="ok", usage=usage, cost_usd=cost, latency_ms=latency_ms
        )
        self.guard.note_success()
        self.input_tokens += usage.input_tokens
        self.output_tokens += usage.output_tokens
        self.thinking_tokens += usage.thinking_tokens
        self.cost_usd += cost
        if prov.is_paid:
            after = gw.ledger.total_usd()
            gw.warn_on_crossing(after - cost, after)
        result = LLMResult(
            text=raw.text,
            parsed=None,
            usage=usage,
            cost_usd=cost,
            cost_inr=gw.pricing.to_inr(cost),
            provider=prov.name,
            model=raw.model,
            service_tier=prov.service_tier,
            latency_ms=latency_ms,
            finish_reason=raw.finish_reason,
            run_id=self.run_id,
            truncated=raw.finish_reason in _TRUNCATED,
        )
        if json_schema is not None:
            try:
                parsed = _parse_json(raw.text)
            except ValueError as e:
                raise BadStructuredOutput(
                    f"{prov.name}/{prov.model} did not return valid JSON"
                    f"{' (cut off at max_output_tokens)' if result.truncated else ''}: {e}",
                    result,
                ) from e
            result = LLMResult(**{**result.__dict__, "parsed": parsed})
        return result

    def summary(self) -> str:
        gw = self.gateway
        try:
            total = gw.ledger.total_usd()
        except Exception as e:
            return f"LLM run {self.run_id}: summary unavailable ({type(e).__name__})"
        budget = gw.settings.llm_budget_usd
        return (
            f"LLM run {self.run_id}: calls={self.calls} tokens in={self.input_tokens} "
            f"out={self.output_tokens} thinking={self.thinking_tokens} | "
            f"cost this run ${self.cost_usd:.6f} (INR {gw.pricing.to_inr(self.cost_usd):.4f}) | "
            f"total spent ${total:.6f} of ${budget:.2f} | remaining ${max(budget - total, 0):.6f}"
        )


def _repo_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else Path(__file__).resolve().parents[3] / p


_RATE_LIMITER: RateLimiter | None = None


def build_gateway(settings: LlmSettings | None = None) -> Gateway:
    """The real gateway: providers from settings, the ledger on DATABASE_URL, the process-wide
    rate limiter."""
    global _RATE_LIMITER
    from rflens.db import get_engine
    from rflens.llm.fake import FakeProvider
    from rflens.llm.gemini import GeminiProvider
    from rflens.llm.ollama import OllamaProvider

    s = settings or get_llm_settings()
    providers: dict[str, Provider] = {
        "gemini": GeminiProvider(s),
        "ollama": OllamaProvider(s),
        "fake": FakeProvider(),
    }
    if s.llm_provider not in providers:
        raise LLMError(f"unknown LLM_PROVIDER {s.llm_provider!r}; use one of {sorted(providers)}")
    if _RATE_LIMITER is None:
        _RATE_LIMITER = RateLimiter(s.llm_max_calls_per_minute)
    return Gateway(
        s,
        providers,
        Ledger(get_engine(), s.llm_budget_usd, s.llm_usd_to_inr),
        Pricing.load(_repo_path(s.llm_prices_file), s.llm_usd_to_inr),
        rate_limiter=_RATE_LIMITER,
    )
