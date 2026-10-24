"""Live smoke test of the LLM gateway: 3 paid Gemini calls and 1 Ollama call.

    $env:LLM_ALLOW_PAID = "true"; uv run python scripts/llm_smoke.py     (PowerShell)

Paid calls are off by default; this needs LLM_ALLOW_PAID=true and GEMINI_API_KEY in .env. Calls:
1. gemini plain text; 2. gemini with a JSON schema; 3. gemini asked for far more than
max_output_tokens allows (checks the cap holds); 4. ollama qwen3:4b (free).
Retries and fallback are switched off, so the script makes at most 3 paid attempts; it checks that
against the ledger. Expected cost: well under $0.01. Exits non-zero if a check fails.
"""

import sys

from sqlalchemy import text

from rflens.config import get_llm_settings
from rflens.db import get_engine
from rflens.llm.base import LLMResult, Message
from rflens.llm.gateway import LLMRun, build_gateway

SCHEMA = {
    "type": "object",
    "properties": {"unit": {"type": "string"}, "is_power_unit": {"type": "boolean"}},
    "required": ["unit", "is_power_unit"],
}
CAP = 30


def check(ok: bool, message: str) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {message}")
    if not ok:
        raise SystemExit(1)


def show(label: str, r: LLMResult) -> None:
    u = r.usage
    print(f"\n{label}")
    print(f"  {r.provider}/{r.model} tier={r.service_tier} finish={r.finish_reason}")
    print(
        f"  tokens in={u.input_tokens} out={u.output_tokens} thinking={u.thinking_tokens}  "
        f"cost ${r.cost_usd:.8f} (INR {r.cost_inr:.5f})  latency {r.latency_ms} ms"
    )
    print(f"  text: {r.text[:140]!r}")


def main() -> None:
    settings = get_llm_settings().model_copy(update={"llm_max_retries": 0, "llm_fallback": ""})
    gateway = build_gateway(settings)
    run: LLMRun
    with gateway.run("llm-smoke") as run:
        print(
            f"run {run.run_id}; model {settings.llm_model}; tier {settings.llm_service_tier}; "
            f"thinking {settings.llm_thinking_budget}"
        )

        plain = run.generate(
            [Message.user("Reply with the single word: ready")],
            purpose="smoke-text",
            max_output_tokens=20,
            provider="gemini",
        )
        show("1. gemini, plain text", plain)
        check("ready" in plain.text.lower() and plain.cost_usd > 0, "answered and priced")

        structured = run.generate(
            [Message.user("Is dBm a unit of power? Answer as JSON with the unit and a boolean.")],
            purpose="smoke-json",
            json_schema=SCHEMA,
            max_output_tokens=100,
            provider="gemini",
        )
        show("2. gemini, JSON schema", structured)
        check(
            isinstance(structured.parsed, dict)
            and set(structured.parsed) == {"unit", "is_power_unit"},
            f"parsed against the schema: {structured.parsed}",
        )

        capped = run.generate(
            [Message.user("Write the integers from 1 to 300 separated by commas, nothing else.")],
            purpose="smoke-cap",
            max_output_tokens=CAP,
            provider="gemini",
        )
        show(f"3. gemini, asked for far more than max_output_tokens={CAP}", capped)
        billed_out = capped.usage.output_tokens + capped.usage.thinking_tokens
        check(capped.truncated, f"stopped at the cap (finish reason {capped.finish_reason})")
        check(billed_out <= CAP, f"billed output {billed_out} tokens <= cap {CAP}")

        local = run.generate(
            [Message.user("Reply with the single word: ready")],
            purpose="smoke-ollama",
            max_output_tokens=20,
            provider="ollama",
        )
        show("4. ollama qwen3:4b (free)", local)
        check(local.cost_usd == 0 and local.provider == "ollama", "zero cost, recorded")

        with get_engine().connect() as conn:
            paid_attempts = conn.execute(
                text("SELECT count(*) FROM llm_calls WHERE run_id = :r AND provider = 'gemini'"),
                {"r": run.run_id},
            ).scalar_one()
        check(paid_attempts == 3, f"exactly 3 paid Gemini attempts in the ledger ({paid_attempts})")
        check(run.cost_usd < 0.01, f"run cost ${run.cost_usd:.6f} is under $0.01")
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}", file=sys.stderr)
        raise SystemExit(1) from e
