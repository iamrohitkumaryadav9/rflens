"""make eval-sql: run the SQL agent on every sql and refuse question of the golden set and score it.

    make eval-sql              # paid: needs LLM_ALLOW_PAID=true and a running sql-mcp
    make eval-sql DRY=1        # prints the plan and the worst-case cost, makes no LLM call

Before any paid call it prints the question count, the maximum number of calls (questions x 4) and
the worst-case cost, then proceeds. For this command only, the per-run call cap is questions x 4
(the per-question cap of 4 is unchanged) and fallback providers are off, so the numbers measure the
configured model. A "match" is execution accuracy against the stored reference result for sql
questions (see scorer.py), and a refusal for refuse questions.
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from rflens.agents.sql_agent import AgentResult, answer_question
from rflens.agents.sql_prompt import build_system_prompt
from rflens.agents.sql_tools import SqlMcpTools, SqlMcpUnavailable
from rflens.config import LlmSettings, get_agent_settings, get_llm_settings
from rflens.eval.golden import GoldenQuestion, load_golden, problems
from rflens.eval.scorer import ResultSet, compare
from rflens.llm.base import LLMError, Message, Pricing, estimate_tokens
from rflens.llm.gateway import _repo_path, build_gateway

CALLS_PER_QUESTION = 4  # the gateway's per-question cap
CORRECTION_OVERHEAD_TOKENS = 150  # the "SQL gateway said ..." message added per correction


@dataclass(frozen=True)
class Plan:
    questions: int
    max_calls: int
    prompt_tokens: int  # estimate, first generation of the longest question
    max_output_tokens: int
    worst_per_question_usd: float
    worst_total_usd: float


def eval_settings(settings: LlmSettings, n_questions: int) -> LlmSettings:
    """This command's settings: run cap = questions x 4, no fallback provider."""
    return settings.model_copy(
        update={"llm_max_calls_per_run": n_questions * CALLS_PER_QUESTION, "llm_fallback": ""}
    )


def build_plan(
    questions: list[GoldenQuestion],
    system_prompt: str,
    settings: LlmSettings,
    max_output_tokens: int,
    pricing: Pricing,
) -> Plan:
    """Worst case: every question uses all 4 calls, each call sends the full prompt (plus the
    growing correction history) and spends its whole output allowance, thinking included."""
    longest = max(questions, key=lambda q: len(q.question))
    prompt_tokens = estimate_tokens([Message.system(system_prompt), Message.user(longest.question)])
    per_question = 0.0
    for k in range(CALLS_PER_QUESTION):
        history = k * (max_output_tokens + CORRECTION_OVERHEAD_TOKENS)
        per_question += pricing.worst_case_usd(
            "gemini",
            settings.llm_model,
            settings.llm_service_tier,
            prompt_tokens + history,
            max_output_tokens,
        )
    return Plan(
        len(questions),
        len(questions) * CALLS_PER_QUESTION,
        prompt_tokens,
        max_output_tokens,
        per_question,
        per_question * len(questions),
    )


def score(q: GoldenQuestion, r: AgentResult) -> tuple[bool, str]:
    if q.type == "refuse":
        ok = r.status == "refused"
        return ok, "" if ok else f"expected a refusal ({q.expected_refusal}), got {r.status}"
    if r.status != "answered":
        return False, f"status {r.status}: {' | '.join(r.notes)[:200]}"
    verdict = compare(ResultSet(r.columns, r.rows), q.reference_result, q.ordered)
    if not verdict.match:
        return False, verdict.reason
    text = " ".join(r.notes).lower()
    missing = [w for w in q.must_mention if w.lower() not in text]
    if missing:
        return False, f"notes do not mention {missing} (the unit must be reported)"
    return True, ""


def _row(q: GoldenQuestion, r: AgentResult, ok: bool) -> str:
    tokens = f"{r.input_tokens}/{r.output_tokens}/{r.thinking_tokens}"
    return (
        f"{q.id:10} {q.type:6} {r.status:9} {'yes' if ok else 'NO':5} {r.calls:5} "
        f"{tokens:>16} ${r.cost_usd:.6f}"
    )


def _first_rows(rows: list, n: int = 3) -> str:
    return "\n".join(f"        {row}" for row in rows[:n]) or "        (no rows)"


def _miss_report(q: GoldenQuestion, r: AgentResult, reason: str) -> str:
    """A miss, with the agent's SQL next to the reference SQL and the first 3 rows of each."""
    lines = [f"MISS {q.id} [{r.status}]: {q.question}", f"  why: {reason}"]
    if r.notes:
        lines.append(f"  agent notes: {' | '.join(r.notes)[:300]}")
    lines.append(f"  agent sql: {r.sql or '(none)'}")
    lines.append(f"  agent rows (first 3 of {r.row_count}), columns {r.columns}:")
    lines.append(_first_rows(r.rows))
    if q.type == "sql":
        lines.append(f"  reference sql: {' '.join((q.reference_sql or '').split())}")
        ref = q.reference_result
        lines.append(f"  reference rows (first 3 of {len(ref.rows)}), columns {ref.columns}:")
        lines.append(_first_rows(ref.rows))
    else:
        lines.append(f"  expected: a refusal ({q.expected_refusal})")
    return "\n".join(lines)


def _save(run_id: str, results: list[tuple[GoldenQuestion, AgentResult, bool, str]]) -> Path:
    """Per-question results as JSON under data/ (gitignored), for later comparison."""
    path = Path(__file__).resolve().parents[3] / "data" / "eval" / f"eval_sql_{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {"match": ok, "reason": reason, "question": q.question, **asdict(r)}
        for q, r, ok, reason in results
    ]
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


HEADER = (
    f"{'id':10} {'type':6} {'status':9} {'match':5} {'calls':>5} {'tokens in/out/think':>16} cost"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the SQL agent on the golden set")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, make no LLM call")
    args = parser.parse_args(argv)

    settings = get_llm_settings()
    agent_settings = get_agent_settings()
    questions = [q for q in load_golden() if q.type in ("sql", "refuse")]
    bad = problems(questions)
    if bad:
        print("golden set problems:\n  " + "\n  ".join(bad), file=sys.stderr)
        return 2

    tools = SqlMcpTools(agent_settings.sql_mcp_url)
    try:
        system_prompt = build_system_prompt(tools.schema_context())
    except SqlMcpUnavailable as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    pricing = Pricing.load(_repo_path(settings.llm_prices_file), settings.llm_usd_to_inr)
    plan = build_plan(
        questions, system_prompt, settings, agent_settings.agent_max_output_tokens, pricing
    )
    print(
        f"eval-sql plan: {plan.questions} questions, at most {plan.max_calls} LLM calls "
        f"({plan.questions} x {CALLS_PER_QUESTION}), model {settings.llm_model} "
        f"({settings.llm_service_tier}), prompt about {plan.prompt_tokens} tokens, "
        f"max output {plan.max_output_tokens} tokens per call"
    )
    print(
        f"worst-case cost ${plan.worst_total_usd:.4f} "
        f"(INR {pricing.to_inr(plan.worst_total_usd):.2f}); every call spending its full "
        f"output allowance. Budget ${settings.llm_budget_usd:.2f}."
    )
    if args.dry_run:
        print("dry run: no LLM call made")
        return 0
    if not settings.llm_allow_paid:
        print("ERROR: eval-sql makes paid calls: set LLM_ALLOW_PAID=true", file=sys.stderr)
        return 2

    gateway = build_gateway(eval_settings(settings, plan.questions))
    ledger = gateway.ledger
    if ledger.total_usd() + plan.worst_total_usd > settings.llm_budget_usd:
        print(
            f"ERROR: spent ${ledger.total_usd():.4f} plus the worst case exceeds "
            f"LLM_BUDGET_USD ${settings.llm_budget_usd:.2f}",
            file=sys.stderr,
        )
        return 2

    results: list[tuple[GoldenQuestion, AgentResult, bool, str]] = []
    stopped: str | None = None
    print("\n" + HEADER)
    with gateway.run("eval-sql") as run:
        for q in questions:
            try:
                r = answer_question(
                    q.question,
                    run=run,
                    tools=tools,
                    question_id=q.id,
                    max_output_tokens=agent_settings.agent_max_output_tokens,
                )
            except LLMError as e:  # run cap, budget, loop detector, ledger: stop the command
                stopped = f"{q.id}: {type(e).__name__}: {e}"
                break
            ok, reason = score(q, r)
            results.append((q, r, ok, reason))
            print(_row(q, r, ok), flush=True)

    print()
    for q, r, ok, reason in results:
        if not ok:
            print(_miss_report(q, r, reason))
    saved = _save(run.run_id, results)
    print(f"full results saved to {saved}")
    if stopped:
        print(f"\nSTOPPED EARLY at {stopped}")
    for kind in ("sql", "refuse"):
        subset = [x for x in results if x[0].type == kind]
        print(f"{kind}: {sum(x[2] for x in subset)}/{len(subset)} matched")
    calls = sum(r.calls for _, r, _, _ in results)
    cost = sum(r.cost_usd for _, r, _, _ in results)
    tokens_in = sum(r.input_tokens for _, r, _, _ in results)
    tokens_out = sum(r.output_tokens + r.thinking_tokens for _, r, _, _ in results)
    total = ledger.total_usd()
    print(
        f"Spend: {calls} paid calls this run, {tokens_in} tokens in / {tokens_out} out, "
        f"cost ${cost:.6f} (INR {pricing.to_inr(cost):.4f}), total spent ${total:.6f}, "
        f"remaining ${max(settings.llm_budget_usd - total, 0):.6f}"
    )
    return 1 if stopped else 0


if __name__ == "__main__":
    raise SystemExit(main())
