"""LLM gateway tests. The ledger is the real Postgres test database; providers are the scripted
fake or an httpx.MockTransport (which cannot reach the network). No test makes a paid call.
"""

import json
from collections.abc import Iterator

import httpx
import pytest
from sqlalchemy import create_engine, text

from rflens.config import LlmSettings, get_sql_mcp_settings
from rflens.db import get_engine
from rflens.llm.base import (
    BadStructuredOutput,
    BudgetExceeded,
    CallCapExceeded,
    InputTooLarge,
    LedgerError,
    LoopDetected,
    Message,
    PaidCallBlocked,
    Pricing,
    PricingError,
    ProviderError,
    Rate,
    RawResponse,
    Usage,
    paid_block_reason,
)
from rflens.llm.fake import FakeProvider
from rflens.llm.gateway import Gateway
from rflens.llm.gemini import GeminiProvider
from rflens.llm.guards import RateLimiter
from rflens.llm.ledger import Ledger
from rflens.llm.ollama import OllamaProvider
from rflens.llm.spend import render

# $0.01 per token in and out, so costs in these tests are easy to read. Every fake name used below
# is priced under the model "fake-1".
PRICING = Pricing(
    {
        name: {"fake-1": {"standard": Rate(10_000, 10_000)}}
        for name in ("fake", "paid", "free", "primary", "backup", "a", "b", "c")
    },
    usd_to_inr=85.7,
)


def msg(i: int | str) -> list[Message]:
    return [Message.user(f"q{i}")]


def settings(**kw) -> LlmSettings:
    kw.setdefault("llm_fallback", "")  # no fallback unless a test asks for one
    return LlmSettings(_env_file=None, **kw)


@pytest.fixture(autouse=True)
def _empty_ledger() -> Iterator[None]:
    with get_engine().begin() as c:
        assert c.execute(text("SELECT current_database()")).scalar_one().endswith("_test")
        c.execute(text("DELETE FROM llm_calls"))
    yield
    with get_engine().begin() as c:
        c.execute(text("DELETE FROM llm_calls"))


class Rig:
    """A gateway over the real test-database ledger with recorded sleeps and warnings."""

    def __init__(self, *providers, pricing: Pricing = PRICING, rate=None, **settings_kw):
        settings_kw.setdefault("llm_provider", providers[0].name)
        self.settings = settings(**settings_kw)
        self.sleeps: list[float] = []
        self.warnings: list[str] = []
        self.ledger = Ledger(get_engine(), self.settings.llm_budget_usd, 85.7)
        self.gateway = Gateway(
            self.settings,
            {p.name: p for p in providers},
            self.ledger,
            pricing,
            rate_limiter=rate,
            sleep=self.sleeps.append,
            emit=self.warnings.append,
        )

    def run(self):
        return self.gateway.run("t")

    def rows(self) -> list:
        with get_engine().connect() as c:
            return list(
                c.execute(
                    text(
                        "SELECT provider, status, attempt, cost_usd, input_tokens, output_tokens, "
                        "thinking_tokens, purpose, question_id, error FROM llm_calls ORDER BY id"
                    )
                )
            )


def ask(run, i, **kw):
    kw.setdefault("max_output_tokens", 5)
    return run.generate(msg(i), purpose=kw.pop("purpose", "test"), **kw)


# --- budget ---------------------------------------------------------------------------------------


def test_budget_blocks_every_paid_call_once_it_would_be_exceeded() -> None:
    paid = FakeProvider(is_paid=True)  # each call costs $0.15 (10 in + 5 out at $0.01)
    rig = Rig(paid, llm_budget_usd=0.5)
    with rig.run() as run:
        for i in range(3):
            ask(run, i)
        with pytest.raises(BudgetExceeded, match="LLM_BUDGET_USD"):
            ask(run, 3)  # $0.45 spent + up to $0.10 would pass $0.50
        with pytest.raises(BudgetExceeded):
            ask(run, 4)  # and it stays blocked
    assert len(paid.calls) == 3  # the blocked calls never reached the provider
    statuses = [r.status for r in rig.rows()]
    assert statuses == ["ok", "ok", "ok", "blocked", "blocked"]
    assert rig.ledger.total_usd() == pytest.approx(0.45)


def test_budget_error_is_clear_and_free_calls_still_work_when_it_is_spent() -> None:
    paid, free = FakeProvider(name="paid", is_paid=True), FakeProvider(name="free")
    rig = Rig(paid, free, llm_budget_usd=0.2)
    with rig.run() as run:
        ask(run, 0)  # $0.15
        with pytest.raises(BudgetExceeded) as err:
            ask(run, 1)
        assert "$0.1500 spent" in str(err.value) and "Raise LLM_BUDGET_USD" in str(err.value)
        assert ask(run, 2, provider="free").provider == "free"  # zero cost: not budgeted


def test_cost_includes_thinking_tokens_and_is_recorded_with_inr() -> None:
    paid = FakeProvider(is_paid=True, usage=Usage(10, 5, 7))
    rig = Rig(paid)
    with rig.run() as run:
        result = ask(run, 0, max_output_tokens=20)
    assert result.usage.thinking_tokens == 7
    assert result.cost_usd == pytest.approx((10 + 5 + 7) * 0.01)
    assert result.cost_inr == pytest.approx(result.cost_usd * 85.7)
    row = rig.rows()[0]
    assert (row.input_tokens, row.output_tokens, row.thinking_tokens) == (10, 5, 7)
    assert float(row.cost_usd) == pytest.approx(0.22)


def test_spend_warnings_at_50_and_80_percent() -> None:
    paid = FakeProvider(is_paid=True, usage=Usage(20, 0, 0))  # $0.20 per call
    rig = Rig(paid, llm_budget_usd=1.0)
    seen = []
    with rig.run() as run:
        for i in range(5):
            ask(run, i)
            seen.append(len([w for w in rig.warnings if "WARNING" in w]))
    assert seen == [0, 0, 1, 2, 2]  # 50% after the 3rd call ($0.60), 80% after the 4th ($0.80)
    assert "50%" in rig.warnings[0] and "80%" in rig.warnings[1]
    assert "LLM_BUDGET_USD" in rig.warnings[0]


def test_unknown_price_refuses_the_paid_call() -> None:
    paid = FakeProvider(model="unpriced", is_paid=True)
    rig = Rig(paid)
    with rig.run() as run, pytest.raises(PricingError, match="no price"):
        ask(run, 0)
    assert paid.calls == [] and rig.rows() == []


def test_real_prices_file_matches_the_confirmed_prices() -> None:
    pricing = Pricing.load("config/llm_prices.yaml", 85.7)
    use = Usage(1_000_000, 1_000_000, 0)
    lite = "gemini-3.1-flash-lite"
    assert pricing.cost_usd("gemini", lite, "standard", use) == pytest.approx(1.75)
    assert pricing.cost_usd("gemini", lite, "flex", use) == pytest.approx(0.875)
    assert pricing.cost_usd("gemini", "gemini-3.5-flash-lite", "standard", use) == pytest.approx(
        2.8
    )
    # thinking tokens are billed as output
    assert pricing.cost_usd("gemini", lite, "standard", Usage(0, 0, 1_000_000)) == 1.5
    assert pricing.source_url.startswith("https://ai.google.dev/gemini-api/docs/pricing")
    assert pricing.retrieved == "2026-10-22"
    assert pricing.to_inr(1.0) == 85.7
    with pytest.raises(PricingError):
        pricing.rate("gemini", "gemini-9-imaginary", "standard")


def test_settings_defaults_match_the_design() -> None:
    s = LlmSettings(_env_file=None)
    assert (s.llm_budget_usd, s.llm_usd_to_inr) == (2.50, 85.7)
    assert (s.llm_max_calls_per_run, s.llm_max_calls_per_question) == (30, 4)
    assert (s.llm_max_calls_per_minute, s.llm_max_retries) == (20, 2)
    assert (s.llm_provider, s.llm_model) == ("gemini", "gemini-3.1-flash-lite")
    assert (s.llm_service_tier, s.llm_thinking_budget) == ("standard", "minimal")
    assert s.llm_allow_paid is False
    with pytest.raises(ValueError, match="LLM_MAX_RETRIES"):
        LlmSettings(_env_file=None, llm_max_retries=3)
    with pytest.raises(ValueError, match="LLM_BUDGET_USD"):
        LlmSettings(_env_file=None, llm_budget_usd=0)


# --- call caps ------------------------------------------------------------------------------------


def test_per_run_cap_stops_the_run_and_names_the_run_id() -> None:
    fake = FakeProvider()
    rig = Rig(fake, llm_max_calls_per_run=3)
    with rig.run() as run:
        for i in range(3):
            ask(run, i)
        with pytest.raises(CallCapExceeded) as err:
            ask(run, 3)
    assert err.value.scope == "run" and run.run_id in str(err.value)
    assert "LLM_MAX_CALLS_PER_RUN=3" in str(err.value)
    assert len(fake.calls) == 3


def test_per_question_cap_is_separate_per_question() -> None:
    fake = FakeProvider()
    rig = Rig(fake)  # default 4 per question
    with rig.run() as run:
        for i in range(4):
            ask(run, i, question_id="q07")
        with pytest.raises(CallCapExceeded) as err:
            ask(run, 4, question_id="q07")
        assert err.value.scope == "question" and "q07" in str(err.value)
        assert "LLM_MAX_CALLS_PER_QUESTION=4" in str(err.value)
        ask(run, 5, question_id="q08")  # another question is unaffected
        ask(run, 6)  # so is a call with no question


def test_retries_and_fallbacks_count_toward_both_caps() -> None:
    busy = ProviderError("busy", 503)
    flaky = FakeProvider(script=[busy, busy, "ok"])
    rig = Rig(flaky, llm_max_calls_per_question=2)
    with rig.run() as run, pytest.raises(CallCapExceeded) as err:
        ask(run, 0, question_id="q1")  # attempt 1 fails, retry 1 fails, retry 2 is over the cap
    assert err.value.scope == "question" and len(flaky.calls) == 2
    flaky2 = FakeProvider(script=[busy, busy, "ok"])
    rig2 = Rig(flaky2, llm_max_calls_per_run=2)
    with rig2.run() as run, pytest.raises(CallCapExceeded) as err:
        ask(run, 0)
    assert err.value.scope == "run" and len(flaky2.calls) == 2


# --- retries --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "error", [ProviderError("busy", 503), ProviderError("slow"), ProviderError("limit", 429)]
)
def test_timeouts_429_and_5xx_are_retried(error: ProviderError) -> None:
    fake = FakeProvider(script=[error, "fine"])
    rig = Rig(fake)
    with rig.run() as run:
        assert ask(run, 0).text == "fine"
    assert len(fake.calls) == 2 and len(rig.sleeps) == 1
    assert [(r.status, r.attempt) for r in rig.rows()] == [("error", 1), ("ok", 2)]


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_client_errors_are_never_retried_and_never_fall_back(status: int) -> None:
    primary = FakeProvider(name="primary", script=[ProviderError("refused", status)])
    backup = FakeProvider(name="backup")
    rig = Rig(primary, backup, llm_fallback="backup")
    with rig.run() as run, pytest.raises(ProviderError) as err:
        ask(run, 0)
    assert err.value.status == status
    assert len(primary.calls) == 1 and backup.calls == [] and rig.sleeps == []


def test_at_most_two_retries_with_exponential_backoff() -> None:
    fake = FakeProvider(script=[ProviderError("busy", 500)] * 5)
    rig = Rig(fake, llm_max_retries=2)
    with rig.run() as run, pytest.raises(ProviderError):
        ask(run, 0)
    assert len(fake.calls) == 3  # one call and two retries, never more
    assert len(rig.sleeps) == 2
    assert 1.0 <= rig.sleeps[0] <= 1.11 and 2.0 <= rig.sleeps[1] <= 2.21  # 1 s then 2 s


def test_retry_after_is_honoured_and_retries_do_not_trip_the_repeat_detector() -> None:
    script = [ProviderError("limit", 429, retry_after=7), ProviderError("busy", 500), "ok"]
    rig = Rig(FakeProvider(script=script))
    with rig.run() as run:
        assert ask(run, 0).text == "ok"  # the same prompt went out 3 times as attempts
    assert rig.sleeps[0] >= 7


def test_max_retries_zero_means_a_single_attempt() -> None:
    fake = FakeProvider(script=[ProviderError("busy", 503)])
    rig = Rig(fake, llm_max_retries=0)
    with rig.run() as run, pytest.raises(ProviderError):
        ask(run, 0)
    assert len(fake.calls) == 1 and rig.sleeps == []


def test_an_unexpected_exception_is_logged_and_propagates_without_retry() -> None:
    fake = FakeProvider(script=[RuntimeError("bug")])
    rig = Rig(fake)
    with rig.run() as run, pytest.raises(RuntimeError, match="bug"):
        ask(run, 0)
    assert len(fake.calls) == 1
    assert [r.status for r in rig.rows()] == ["error"] and "RuntimeError" in rig.rows()[0].error


# --- repeat detectors -----------------------------------------------------------------------------


def test_the_third_identical_prompt_stops_the_run() -> None:
    fake = FakeProvider()
    rig = Rig(fake)
    with rig.run() as run:
        run.generate(msg("same"), purpose="a", max_output_tokens=5)
        run.generate(msg("same"), purpose="b", max_output_tokens=5)  # purpose does not matter
        with pytest.raises(LoopDetected) as err:
            run.generate(msg("same"), purpose="c", max_output_tokens=5)
    assert err.value.kind == "repeat_prompt" and run.run_id in str(err.value)
    assert len(fake.calls) == 2  # the third never reached the provider
    other = rig.run()
    other.generate(msg("same"), purpose="a", max_output_tokens=5)  # a new run starts from zero


def test_the_same_error_three_calls_in_a_row_stops_the_run() -> None:
    fake = FakeProvider(script=[ProviderError("bad request", 400)] * 3)
    rig = Rig(fake)
    with rig.run() as run:
        for i in range(2):
            with pytest.raises(ProviderError):
                ask(run, i)
        with pytest.raises(LoopDetected) as err:
            ask(run, 2)
    assert err.value.kind == "repeat_error" and run.run_id in str(err.value)
    assert "400" in str(err.value)


def test_different_errors_or_a_success_reset_the_error_streak() -> None:
    def bad(label: str) -> ProviderError:
        return ProviderError(label, 400)

    fake = FakeProvider(
        script=[bad("a"), bad("a"), bad("b"), bad("a"), bad("a"), "fine", bad("a"), bad("a")]
    )
    rig = Rig(fake)
    with rig.run() as run:
        for i in range(5):
            with pytest.raises(ProviderError):
                ask(run, i)
        ask(run, 5)  # success
        for i in (6, 7):
            with pytest.raises(ProviderError):
                ask(run, i)  # streak restarted after the success: only two so far


# --- rate limit -----------------------------------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def test_rate_limiter_waits_for_a_free_slot() -> None:
    clock = Clock()
    limiter = RateLimiter(3, clock=clock, sleep=clock.sleep)
    for _ in range(3):
        limiter.acquire()
        clock.now += 1
    assert clock.slept == []
    limiter.acquire()  # the 4th within the minute waits until the first one is 60 s old
    assert len(clock.slept) == 1 and clock.slept[0] == pytest.approx(57, abs=0.01)
    assert clock.now == pytest.approx(1060, abs=0.01)


def test_gateway_attempts_are_rate_limited_across_runs() -> None:
    clock = Clock()
    limiter = RateLimiter(2, clock=clock, sleep=clock.sleep)
    fake = FakeProvider()
    rig = Rig(fake, rate=limiter)
    with rig.run() as first:
        ask(first, 0)
        ask(first, 1)
    assert clock.slept == []
    with rig.run() as second:  # another run, same process-wide limiter
        ask(second, 2)
    assert len(clock.slept) == 1 and clock.slept[0] == pytest.approx(60, abs=0.01)
    assert len(fake.calls) == 3


def test_retries_also_pass_through_the_rate_limiter() -> None:
    clock = Clock()
    limiter = RateLimiter(2, clock=clock, sleep=clock.sleep)
    busy = ProviderError("busy", 503)
    fake = FakeProvider(script=[busy, busy, "ok"])
    rig = Rig(fake, rate=limiter)
    with rig.run() as run:
        ask(run, 0)
    assert len(fake.calls) == 3 and len(clock.slept) == 1  # the 3rd attempt waited


# --- fallback -------------------------------------------------------------------------------------


def test_fallback_happens_once_after_retries_and_never_returns_to_the_primary() -> None:
    primary = FakeProvider(name="primary", script=[ProviderError("down", 503)] * 5)
    backup = FakeProvider(name="backup", script=["from backup"])
    rig = Rig(primary, backup, llm_fallback="backup")
    with rig.run() as run:
        result = ask(run, 0)
    assert (result.provider, result.text) == ("backup", "from backup")
    assert len(primary.calls) == 3 and len(backup.calls) == 1
    assert [(r.provider, r.status, r.attempt) for r in rig.rows()] == [
        ("primary", "error", 1), ("primary", "error", 2), ("primary", "error", 3),
        ("backup", "ok", 4),
    ]  # fmt: skip


def test_a_failing_fallback_is_tried_once_and_the_error_surfaces() -> None:
    primary = FakeProvider(name="primary", script=[ProviderError("down", 503)] * 5)
    backup = FakeProvider(name="backup", script=[ProviderError("also down", 503)] * 5)
    rig = Rig(primary, backup, llm_fallback="backup")
    with rig.run() as run, pytest.raises(ProviderError, match="also down"):
        ask(run, 0)
    assert len(primary.calls) == 3 and len(backup.calls) == 1  # no loop, no second round


def test_the_chain_never_lists_the_primary_twice_or_more_than_one_fallback() -> None:
    a, b, c = (FakeProvider(name=n) for n in "abc")
    rig = Rig(a, b, c, llm_fallback="a,b,c")
    assert [p.name for p in rig.gateway.chain()] == ["a", "b"]
    assert [p.name for p in rig.gateway.chain("b")] == ["b", "a"]
    solo = Rig(a, llm_fallback="a,missing")
    assert [p.name for p in solo.gateway.chain()] == ["a"]


def test_a_budget_block_does_not_fall_back_to_another_provider() -> None:
    paid = FakeProvider(name="paid", is_paid=True)
    free = FakeProvider(name="free")
    rig = Rig(paid, free, llm_fallback="free", llm_budget_usd=0.05)
    with rig.run() as run, pytest.raises(BudgetExceeded):
        ask(run, 0)
    assert free.calls == []


# --- fail closed ledger ---------------------------------------------------------------------------


def test_if_the_reservation_cannot_be_written_no_call_is_made() -> None:
    reader = create_engine(get_sql_mcp_settings().reader_url())  # SELECT-only role: INSERT fails
    try:
        fake = FakeProvider(is_paid=True)
        rig = Rig(fake)
        rig.gateway.ledger = Ledger(reader, 2.5, 85.7)
        with rig.run() as run, pytest.raises(LedgerError, match="no call was made"):
            ask(run, 0)
    finally:
        reader.dispose()
    assert fake.calls == [] and rig.rows() == []


def test_if_the_update_fails_the_paid_result_is_withheld_and_spend_still_counts() -> None:
    with get_engine().begin() as c:
        c.execute(
            text(
                """
                CREATE FUNCTION llm_calls_refuse_update() RETURNS trigger LANGUAGE plpgsql AS $$
                BEGIN RAISE EXCEPTION 'update refused by the test'; END $$
                """
            )
        )
        c.execute(
            text(
                "CREATE TRIGGER llm_calls_refuse BEFORE UPDATE ON llm_calls "
                "FOR EACH ROW EXECUTE FUNCTION llm_calls_refuse_update()"
            )
        )
    try:
        paid = FakeProvider(is_paid=True)
        rig = Rig(paid, llm_budget_usd=1.0)
        with rig.run() as run, pytest.raises(LedgerError, match="result is withheld"):
            ask(run, 0)
        assert len(paid.calls) == 1  # the provider was called ...
        (row,) = rig.rows()
        assert row.status == "pending"  # ... and the reservation stays
        assert float(row.cost_usd) == pytest.approx(0.10)  # at its worst-case cost
        assert rig.ledger.total_usd() == pytest.approx(0.10)
    finally:
        with get_engine().begin() as c:
            c.execute(text("DROP TRIGGER llm_calls_refuse ON llm_calls"))
            c.execute(text("DROP FUNCTION llm_calls_refuse_update()"))


# --- paid-call block ------------------------------------------------------------------------------


def test_paid_block_reason_covers_ci_pytest_and_the_opt_in() -> None:
    assert "CI" in paid_block_reason(True, env={"CI": "true"}, modules={})
    assert "CI" in paid_block_reason(True, env={"CI": "1"}, modules={})
    assert "pytest" in paid_block_reason(True, env={"PYTEST_CURRENT_TEST": "x"}, modules={})
    assert "pytest" in paid_block_reason(True, env={}, modules={"pytest": object()})
    assert "LLM_ALLOW_PAID" in paid_block_reason(False, env={}, modules={})
    assert paid_block_reason(True, env={}, modules={}) is None
    assert paid_block_reason(True, env={"CI": "false"}, modules={}) is None


@pytest.mark.parametrize(
    ("ci", "reason"),
    [
        (None, "pytest"),  # a developer machine: CI unset, so the pytest rule is what blocks
        ("true", "CI"),  # GitHub Actions always sets CI=true, and the CI rule is checked first
    ],
)
def test_the_real_gemini_provider_cannot_send_a_request_under_pytest_or_in_ci(
    monkeypatch: pytest.MonkeyPatch, ci: str | None, reason: str
) -> None:
    # The block must hold in both environments; which rule reports it depends on CI, so pin it.
    if ci is None:
        monkeypatch.delenv("CI", raising=False)
    else:
        monkeypatch.setenv("CI", ci)
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json={})

    s = settings(gemini_api_key="k", llm_allow_paid=True)  # even explicitly allowed
    provider = GeminiProvider(s)  # a real provider: no transport override
    provider._client = httpx.Client(transport=httpx.MockTransport(handler))  # would record a leak
    with pytest.raises(PaidCallBlocked, match=reason):
        provider.generate(msg(0), None, 10)
    assert sent == []  # nothing left the process in either environment


def test_the_gateway_blocks_a_real_gemini_provider_before_any_ledger_row() -> None:
    provider = GeminiProvider(settings(gemini_api_key="k", llm_allow_paid=True))
    rig = Rig(provider, pricing=Pricing.load("config/llm_prices.yaml", 85.7))
    with rig.run() as run, pytest.raises(PaidCallBlocked):
        ask(run, 0)
    assert rig.rows() == [] and run.calls == 0


def test_ci_blocks_even_when_pytest_is_not_involved() -> None:
    provider = GeminiProvider(settings(gemini_api_key="k", llm_allow_paid=True))
    provider._block = lambda: paid_block_reason(True, env={"CI": "true"}, modules={})
    with pytest.raises(PaidCallBlocked, match="CI"):
        provider.check_allowed()


# --- Gemini request and response handling (mock transport: cannot reach the network) --------------


def gemini(handler, **kw) -> GeminiProvider:
    s = settings(gemini_api_key="secret-key-123", llm_allow_paid=True, **kw)
    return GeminiProvider(s, transport=httpx.MockTransport(handler))


def ok_body(text_="hello", **usage) -> dict:
    usage = {"promptTokenCount": 8, "candidatesTokenCount": 2, **usage}
    return {
        "candidates": [{"content": {"parts": [{"text": text_}]}, "finishReason": "STOP"}],
        "usageMetadata": usage,
        "modelVersion": "gemini-3.1-flash-lite",
    }


def test_gemini_request_shape_and_usage_parsing() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["headers"] = str(request.url), request.headers
        seen["body"] = json.loads(request.content)
        body = ok_body(thoughtsTokenCount=11)
        body["candidates"][0]["content"]["parts"] = [
            {"text": "internal", "thought": True},
            {"text": "the answer"},
        ]
        return httpx.Response(200, json=body)

    conversation = [
        Message.system("be brief"),
        Message.user("hi"),
        Message.assistant("yo"),
        Message.user("again"),
    ]
    raw = gemini(handler).generate(conversation, {"type": "object"}, 64)
    body = seen["body"]
    assert seen["url"].endswith("/models/gemini-3.1-flash-lite:generateContent")
    assert "secret-key-123" not in seen["url"]
    assert seen["headers"]["x-goog-api-key"] == "secret-key-123"
    assert body["generationConfig"] == {
        "maxOutputTokens": 64,
        "thinkingConfig": {"thinkingLevel": "minimal"},
        "responseMimeType": "application/json",
        "responseJsonSchema": {"type": "object"},
    }
    assert body["systemInstruction"] == {"parts": [{"text": "be brief"}]}
    assert [c["role"] for c in body["contents"]] == ["user", "model", "user"]
    assert "serviceTier" not in body  # standard is the default: not sent
    assert raw.text == "the answer"  # thought parts are not part of the answer
    assert raw.usage == Usage(input_tokens=8, output_tokens=2, thinking_tokens=11)
    assert raw.finish_reason == "STOP"


def test_gemini_flex_tier_thinking_budget_and_timeout() -> None:
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=ok_body())

    flex = gemini(handler, llm_service_tier="flex", llm_thinking_budget="512", llm_timeout_s=30)
    flex.generate(msg(0), None, 10)
    assert bodies[0]["serviceTier"] == "flex"
    assert bodies[0]["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 512}
    assert flex._client.timeout.read >= 600  # flex requests can queue for minutes
    low = gemini(handler, llm_thinking_budget="low")
    low.generate(msg(0), None, 10)
    assert bodies[1]["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "low"}


@pytest.mark.parametrize(
    ("status", "retryable"),
    [(429, True), (500, True), (503, True), (400, False), (401, False), (403, False), (404, False)],
)
def test_gemini_http_errors_map_to_retryable_or_not(status: int, retryable: bool) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status, json={"error": {"message": "nope"}}, headers={"retry-after": "9"}
        )

    with pytest.raises(ProviderError) as err:
        gemini(handler).generate(msg(0), None, 10)
    assert err.value.status == status and err.value.retryable is retryable
    assert "nope" in str(err.value) and "secret-key-123" not in str(err.value)
    assert err.value.retry_after == 9


def test_gemini_timeout_connection_failure_and_empty_answer() -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    for handler in (timeout, refused):
        with pytest.raises(ProviderError) as err:
            gemini(handler).generate(msg(0), None, 10)
        assert err.value.status is None and err.value.retryable
    feedback = {"promptFeedback": {"blockReason": "SAFETY"}}
    with pytest.raises(ProviderError, match="SAFETY") as err:
        gemini(lambda r: httpx.Response(200, json=feedback)).generate(msg(0), None, 10)
    assert not err.value.retryable


def test_a_missing_api_key_is_a_clear_error_not_a_request() -> None:
    p = GeminiProvider(
        settings(llm_allow_paid=True), transport=httpx.MockTransport(lambda r: httpx.Response(200))
    )
    with pytest.raises(ProviderError, match="GEMINI_API_KEY"):
        p.generate(msg(0), None, 10)


def test_the_api_key_never_appears_in_settings_repr() -> None:
    assert "secret-key-123" not in repr(settings(gemini_api_key="secret-key-123"))


# --- Ollama ---------------------------------------------------------------------------------------


def test_ollama_request_and_usage() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"], seen["body"] = str(request.url), json.loads(request.content)
        answer = {
            "model": "qwen3:4b",
            "message": {"content": "{}"},
            "prompt_eval_count": 12,
            "eval_count": 3,
            "done_reason": "stop",
        }
        return httpx.Response(200, json=answer)

    p = OllamaProvider(settings(), transport=httpx.MockTransport(handler))
    raw = p.generate([Message.user("hi")], {"type": "object"}, 50)
    assert seen["url"].endswith("/api/chat")
    assert seen["body"]["think"] is False and seen["body"]["stream"] is False
    assert seen["body"]["options"] == {"num_predict": 50}
    assert seen["body"]["format"] == {"type": "object"}
    assert (raw.usage, raw.text, p.is_paid) == (Usage(12, 3, 0), "{}", False)


def test_ollama_down_is_a_retryable_error() -> None:
    def refused(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    p = OllamaProvider(settings(), transport=httpx.MockTransport(refused))
    with pytest.raises(ProviderError) as err:
        p.generate(msg(0), None, 5)
    assert err.value.retryable


# --- request limits, structured output, summary ---------------------------------------------------


def test_input_and_output_limits() -> None:
    fake = FakeProvider()
    rig = Rig(fake, llm_max_input_tokens=50, llm_max_output_tokens=100)
    with rig.run() as run:
        with pytest.raises(InputTooLarge):
            run.generate([Message.user("x" * 400)], purpose="t")
        assert fake.calls == []
        run.generate(msg(0), purpose="t", max_output_tokens=5000)
        run.generate(msg(1), purpose="t")
    assert [c[2] for c in fake.calls] == [100, 100]  # clamped to LLM_MAX_OUTPUT_TOKENS


def test_structured_output_is_parsed_and_bad_json_is_billed_and_reported() -> None:
    schema = {"type": "object"}
    fake = FakeProvider(is_paid=True, script=['{"a": 1}', '```json\n{"b": 2}\n```', "not json"])
    rig = Rig(fake)

    def structured(run, i):
        return run.generate(msg(i), purpose="t", json_schema=schema, max_output_tokens=5)

    with rig.run() as run:
        assert structured(run, 0).parsed == {"a": 1}
        assert structured(run, 1).parsed == {"b": 2}
        with pytest.raises(BadStructuredOutput) as err:
            structured(run, 2)
    assert err.value.result.cost_usd == pytest.approx(0.15)  # the call was paid for and logged
    assert [r.status for r in rig.rows()] == ["ok", "ok", "ok"]


def test_truncated_output_is_flagged() -> None:
    cut = RawResponse("part", Usage(5, 5, 0), "fake-1", finish_reason="MAX_TOKENS")
    rig = Rig(FakeProvider(script=[cut]))
    with rig.run() as run:
        assert ask(run, 0).truncated is True


def test_run_summary_and_spend_report() -> None:
    paid = FakeProvider(is_paid=True, usage=Usage(10, 5, 2))  # $0.17 per call
    rig = Rig(paid, llm_budget_usd=2.5)
    with rig.run() as run:
        ask(run, 0, purpose="plan-sql")
        ask(run, 1, purpose="plan-sql")
        ask(run, 2, purpose="summarise")
        summary = run.summary()
    assert f"LLM run {run.run_id}: calls=3" in summary
    assert "tokens in=30 out=15 thinking=6" in summary
    assert "cost this run $0.510000" in summary and "total spent $0.510000 of $2.50" in summary
    assert "remaining $1.990000" in summary
    assert rig.warnings[-1] == summary  # printed at the end of the run
    report = render(rig.ledger)
    assert "Total spent               $0.510000" in report
    assert "Remaining budget          $1.990000" in report
    assert "Spent today (UTC)         $0.510000" in report
    assert report.count("fake/fake-1") == 3  # the last calls
    top = report.split("Top 5 purposes by cost:")[1]
    assert top.index("plan-sql") < top.index("summarise")  # ordered by cost


# --- loop detectors are scoped to a question when the call names one ---------------------------


def test_identical_prompts_of_different_questions_do_not_add_up() -> None:
    fake = FakeProvider()
    rig = Rig(fake)
    with rig.run() as run:
        for qid in ("q1", "q2", "q3", "q4"):  # the same text for four questions: no loop
            run.generate(msg("same"), purpose="a", question_id=qid, max_output_tokens=5)
        for _ in range(2):
            run.generate(msg("again"), purpose="a", question_id="q1", max_output_tokens=5)
        with pytest.raises(LoopDetected) as err:  # the third of one question is a loop
            run.generate(msg("again"), purpose="a", question_id="q1", max_output_tokens=5)
        assert err.value.kind == "repeat_prompt" and err.value.question_id == "q1"
        assert "question 'q1'" in str(err.value) and run.run_id in str(err.value)
        # another question is not affected, and the run goes on
        run.generate(msg("again"), purpose="a", question_id="q2", max_output_tokens=5)


def test_error_streaks_are_counted_per_question() -> None:
    fake = FakeProvider(script=[ProviderError("bad", 400)] * 5)
    rig = Rig(fake)
    with rig.run() as run:
        for qid in ("q1", "q2", "q1", "q2"):  # interleaved: two each, no streak of three
            with pytest.raises(ProviderError):
                run.generate(
                    msg(qid + str(len(fake.calls))),
                    purpose="a",
                    question_id=qid,
                    max_output_tokens=5,
                )
        with pytest.raises(LoopDetected) as err:  # q1's third in a row
            run.generate(msg("q1 last"), purpose="a", question_id="q1", max_output_tokens=5)
        assert err.value.kind == "repeat_error" and err.value.question_id == "q1"
