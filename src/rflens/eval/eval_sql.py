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
from rflens.eval.results_log import RESULTS_PATH, append_row, git_commit
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
    """This command's settings: run cap = questions x 4, no fallback provider, temperature 0
    unless LLM_TEMPERATURE is set."""
    update: dict = {"llm_max_calls_per_run": n_questions * CALLS_PER_QUESTION, "llm_fallback": ""}
    if "llm_temperature" not in settings.model_fields_set:
        update["llm_temperature"] = 0.0
    return settings.model_copy(update=update)


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


def compare_repeats(runs: list[list[tuple[GoldenQuestion, AgentResult, bool, str]]]) -> str:
    """How many answers differ between repeated runs of the same questions: the SQL text, and the
    result set (columns and rows)."""
    first, rest = runs[0], runs[1:]
    lines = []
    sql_diff, result_diff = [], []
    for i, (q, r1, ok1, _) in enumerate(first):
        for other in rest:
            if i >= len(other):
                continue
            _, r2, ok2, _ = other[i]
            norm = lambda s: " ".join((s or "").split()).lower()  # noqa: E731
            if norm(r1.sql) != norm(r2.sql):
                sql_diff.append(q.id)
            if (r1.columns, r1.rows, r1.status) != (r2.columns, r2.rows, r2.status):
                result_diff.append(q.id)
                lines.append(
                    f"  {q.id}: run 1 {'correct' if ok1 else 'WRONG'} {r1.rows[:2]} | "
                    f"run 2 {'correct' if ok2 else 'WRONG'} {r2.rows[:2]}"
                )
    n = min(len(r) for r in runs)
    summary = (
        f"{len(set(sql_diff))} of {n} answers differ in their SQL text, "
        f"{len(set(result_diff))} of {n} differ in their result"
    )
    return "\n".join([summary, *lines])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the SQL agent on the golden set")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, make no LLM call")
    parser.add_argument("--note", default="", help="one line for the last column of RESULTS.md")
    parser.add_argument("--only", choices=["sql", "refuse"], help="run one question type only")
    parser.add_argument("--repeat", type=int, default=1, help="run the set N times and compare")
    parser.add_argument("--stop-usd", type=float, help="stop starting questions at this run cost")
    args = parser.parse_args(argv)

    settings = get_llm_settings()
    agent_settings = get_agent_settings()
    questions = [
        q for q in load_golden() if q.type in ((args.only,) if args.only else ("sql", "refuse"))
    ]
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
    reps = max(args.repeat, 1)
    shown = eval_settings(settings, plan.questions * reps)
    print(
        f"eval-sql plan: {plan.questions} questions x {reps} run(s), at most "
        f"{plan.max_calls * reps} LLM calls ({plan.questions} x {CALLS_PER_QUESTION} x {reps}), "
        f"model {settings.llm_model} ({settings.llm_service_tier}), temperature "
        f"{shown.llm_temperature}, prompt about {plan.prompt_tokens} tokens, "
        f"max output {plan.max_output_tokens} tokens per call"
    )
    print(
        f"worst-case cost ${plan.worst_total_usd * reps:.4f} "
        f"(INR {pricing.to_inr(plan.worst_total_usd * reps):.2f}); every call spending its full "
        f"output allowance."
        + (f" The runs stop at ${args.stop_usd:.2f}." if args.stop_usd else "")
        + f" Budget ${settings.llm_budget_usd:.2f}."
    )
    if args.dry_run:
        print("dry run: no LLM call made")
        return 0
    if not settings.llm_allow_paid:
        print("ERROR: eval-sql makes paid calls: set LLM_ALLOW_PAID=true", file=sys.stderr)
        return 2

    gateway = build_gateway(shown)
    ledger = gateway.ledger
    ceiling = args.stop_usd if args.stop_usd else plan.worst_total_usd * reps
    if ledger.total_usd() + ceiling > settings.llm_budget_usd:
        print(
            f"ERROR: spent ${ledger.total_usd():.4f} plus ${ceiling:.2f} exceeds "
            f"LLM_BUDGET_USD ${settings.llm_budget_usd:.2f}",
            file=sys.stderr,
        )
        return 2

    all_runs: list[list[tuple[GoldenQuestion, AgentResult, bool, str]]] = []
    stopped: str | None = None
    spent_in_runs = 0.0
    commit = git_commit()  # HEAD as the run starts
    for rep in range(reps):
        results: list[tuple[GoldenQuestion, AgentResult, bool, str]] = []
        print(f"\n--- run {rep + 1} of {reps} ---\n" + HEADER)
        with gateway.run("eval-sql") as run:
            for q in questions:
                if args.stop_usd and spent_in_runs + run.cost_usd >= args.stop_usd:
                    stopped = f"cost stop ${args.stop_usd:.2f} reached before {q.id}"
                    break
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
        spent_in_runs += run.cost_usd
        all_runs.append(results)
        print()
        for q, r, ok, reason in results:
            if not ok:
                print(_miss_report(q, r, reason))
        saved = _save(run.run_id, results)
        print(f"full results saved to {saved}")
        for kind in ("sql", "refuse"):
            subset = [x for x in results if x[0].type == kind]
            if subset:
                print(f"{kind}: {sum(x[2] for x in subset)}/{len(subset)} matched")
        calls = sum(r.calls for _, r, _, _ in results)
        cost = sum(r.cost_usd for _, r, _, _ in results)
        by_kind = {kind: [x for x in results if x[0].type == kind] for kind in ("sql", "refuse")}
        note = " ".join(
            part
            for part in (
                args.note,
                f"run {rep + 1}/{reps}, temperature {shown.llm_temperature}" if reps > 1 else "",
                f"STOPPED EARLY at {stopped}" if stopped else "",
            )
            if part
        )
        append_row(
            run_id=run.run_id,
            commit=commit,
            model=settings.llm_model,
            tier=settings.llm_service_tier,
            sql_matched=sum(x[2] for x in by_kind["sql"]),
            sql_total=len(by_kind["sql"]),
            refuse_matched=sum(x[2] for x in by_kind["refuse"]),
            refuse_total=len(by_kind["refuse"]),
            calls=calls,
            cost_usd=cost,
            note=note,
        )
        if stopped:
            break
    if stopped:
        print(f"\nSTOPPED EARLY at {stopped}")
    if len(all_runs) > 1:
        print("\n--- variance between the runs ---\n" + compare_repeats(all_runs))
    total = ledger.total_usd()
    calls_all = sum(r.calls for results in all_runs for _, r, _, _ in results)
    tokens_in = sum(r.input_tokens for results in all_runs for _, r, _, _ in results)
    tokens_out = sum(
        r.output_tokens + r.thinking_tokens for results in all_runs for _, r, _, _ in results
    )
    print(
        f"Spend: {calls_all} paid calls this command, {tokens_in} tokens in / {tokens_out} out, "
        f"cost ${spent_in_runs:.6f} (INR {pricing.to_inr(spent_in_runs):.4f}), total spent "
        f"${total:.6f}, remaining ${max(settings.llm_budget_usd - total, 0):.6f}"
    )
    print(f"rows appended to {RESULTS_PATH}")
    return 1 if stopped else 0


if __name__ == "__main__":
    raise SystemExit(main())
