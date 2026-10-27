"""make eval-e2e: every golden question through the whole graph, scored on the final answer.

    make eval-e2e            # PAID: needs LLM_ALLOW_PAID=true and make servers-up
    make eval-e2e DRY=1      # prints the plan and the worst-case cost, makes no LLM call

One thread per question (thread id = run id + question id), approval automatic, through the
checkpointer, with the orchestrated per-question cap of 10 calls. Scoring is in e2e_score.py. Before
any paid call the command prints the question count, the worst-case number of calls and cost, then
proceeds. `--stop-usd` (default 0.40) is a stop for the whole run: when the cost of the run has
reached it, the current question is finished and the command stops with the results so far. For this
command only, the run call cap is questions x 10 and fallback providers are off, so the numbers
measure the configured model. Diagnosis only: nothing here changes prompts or rules.
"""

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from rflens.agents.analytics_tools import AnalyticsMcpTools, AnalyticsMcpUnavailable
from rflens.agents.sql_prompt import build_system_prompt
from rflens.agents.sql_tools import SqlMcpTools, SqlMcpUnavailable
from rflens.config import LlmSettings, get_agent_settings, get_graph_settings, get_llm_settings
from rflens.eval.e2e_score import Score, score_question
from rflens.eval.golden import GoldenQuestion, load_golden, problems
from rflens.eval.results_log import RESULTS_PATH, append_e2e_row, git_commit
from rflens.graph.app import ask, build_graph, open_checkpointer
from rflens.graph.docs_tools import DocsMcpError, DocsMcpTools, DocsMcpUnavailable
from rflens.graph.nodes import WRITER_EST_OVERHEAD_TOKENS, GraphDeps
from rflens.llm.base import LLMError, Message, Pricing, estimate_tokens
from rflens.llm.gateway import _repo_path, build_gateway

CALLS_PER_QUESTION = 10  # ORCH_MAX_CALLS_PER_QUESTION
DEFAULT_STOP_USD = 0.40
TYPES = ("sql", "doc", "mixed", "stats", "refuse")
AUTHORS = ("human", "assistant")


def eval_settings(settings: LlmSettings, n_questions: int) -> LlmSettings:
    """This command's settings: run cap = questions x 10, no fallback provider, temperature 0
    unless LLM_TEMPERATURE is set."""
    update: dict = {
        "llm_max_calls_per_run": n_questions * CALLS_PER_QUESTION,
        "llm_fallback": "",
    }
    if "llm_temperature" not in settings.model_fields_set:
        update["llm_temperature"] = 0.0
    return settings.model_copy(update=update)


def worst_case_usd(
    questions: list[GoldenQuestion],
    system_prompt: str,
    settings: LlmSettings,
    sql_max_output: int,
    writer_max_output: int,
    evidence_chars: int,
    pricing: Pricing,
) -> tuple[float, float]:
    """(per question, total): all 10 calls of a question at the dearest kind of call, which is
    either a SQL generation (full schema prompt plus a spec context, full output allowance) or a
    writer call (full evidence, full output allowance)."""
    longest = max(questions, key=lambda q: len(q.question))
    sql_in = estimate_tokens([Message.system(system_prompt), Message.user(longest.question)])
    sql_in += evidence_chars // 3  # a mixed question adds spec excerpts to the SQL prompt
    writer_in = evidence_chars // 3 + WRITER_EST_OVERHEAD_TOKENS

    def one(est_in: int, max_out: int) -> float:
        return pricing.worst_case_usd(
            "gemini", settings.llm_model, settings.llm_service_tier, est_in, max_out
        )

    per_call = max(one(sql_in, sql_max_output), one(writer_in, writer_max_output))
    per_question = per_call * CALLS_PER_QUESTION
    return per_question, per_question * len(questions)


# --- running -----------------------------------------------------------------------------


def evaluate(
    questions: list[GoldenQuestion],
    *,
    run,
    graph,
    run_id: str,
    stop_usd: float,
    emit=print,
) -> tuple[list[Score], str | None]:
    """Run the questions one after the other; returns the scores and why it stopped early."""
    scores: list[Score] = []
    stopped: str | None = None
    for q in questions:
        if run.cost_usd >= stop_usd:
            stopped = (
                f"cost stop ${stop_usd:.2f} reached (run cost ${run.cost_usd:.4f}) before {q.id}"
            )
            break
        thread = f"{run_id}-{q.id}"
        try:
            outcome = ask(graph, q.question, thread)
        except (LLMError, SqlMcpUnavailable, DocsMcpUnavailable, AnalyticsMcpUnavailable) as e:
            stopped = f"{q.id}: {type(e).__name__}: {e}"
            break
        scored = score_question(q, outcome.state)
        scores.append(scored)
        emit(row_line(scored))
    return scores, stopped


HEADER = (
    f"{'id':10} {'type':6} {'ok':3} {'status':9} {'ver':3} {'route':6} {'calls':>5} "
    f"{'wr/sql retries':>14} {'cost':>10}  failure types"
)


def row_line(s: Score) -> str:
    retries = f"{s.writer_retries}/{s.sql_retries}"
    return (
        f"{s.qid:10} {s.type:6} {'yes' if s.correct else 'NO':3} {s.status:9} "
        f"{'yes' if s.verified else 'no':3} {str(s.route):6} {s.calls:5} {retries:>14} "
        f"${s.cost_usd:.6f}  {','.join(s.failure_types) or '-'}"
    )


# --- report ------------------------------------------------------------------------------


def fraction(scores: list[Score]) -> str:
    return f"{sum(s.correct for s in scores)}/{len(scores)}"


def summarize(scores: list[Score]) -> dict:
    """The numbers of the report, as plain data (also what the tests check)."""
    by_type = {t: [s for s in scores if s.type == t] for t in TYPES}
    by_author = {a: [s for s in scores if s.author == a] for a in AUTHORS}
    docs = [s for s in scores if s.type == "doc"]
    return {
        "type": {t: fraction(v) for t, v in by_type.items()},
        "author": {a: fraction(v) for a, v in by_author.items()},
        "doc_answer": fraction(docs),
        "doc_recall_search": f"{sum(bool(s.recall_search) for s in docs)}/{len(docs)}",
        "doc_recall_evidence": f"{sum(bool(s.recall_evidence) for s in docs)}/{len(docs)}",
        "status": dict(Counter(s.status for s in scores)),
        "writer_retries": sum(s.writer_retries for s in scores),
        "sql_retries": sum(s.sql_retries for s in scores),
        "questions_with_retries": sum(bool(s.writer_retries or s.sql_retries) for s in scores),
        "failure_types": dict(Counter(f for s in scores for f in s.failure_types)),
        "calls": {t: sum(s.calls for s in v) for t, v in by_type.items()},
        "cost": {t: round(sum(s.cost_usd for s in v), 6) for t, v in by_type.items()},
        "causes": dict(Counter(s.cause for s in scores if not s.correct)),
    }


def render_report(scores: list[Score], questions: dict[str, GoldenQuestion]) -> str:
    sm = summarize(scores)
    lines = ["=" * 78, "END-TO-END RESULTS", "=" * 78, "correct / total per type:"]
    for t in TYPES:
        lines.append(f"  {t:7} {sm['type'][t]}")
    lines.append("per author group: " + ", ".join(f"{a} {sm['author'][a]}" for a in AUTHORS))
    lines.append(
        f"doc questions: answer correct {sm['doc_answer']}  |  evidence recall: "
        f"search results {sm['doc_recall_search']}, "
        f"chunks the writer saw {sm['doc_recall_evidence']}"
    )
    mixed = [s for s in scores if s.type == "mixed"]
    if mixed:
        lines.append(
            f"mixed questions: correct {fraction(mixed)}  |  doc part "
            f"{sum(bool(s.doc_hit) for s in mixed)}/{len(mixed)}, SQL part "
            f"{sum(bool(s.sql_ok) for s in mixed)}/{len(mixed)}"
        )
    lines.append(f"status distribution: {sm['status']}")
    lines.append(
        f"retries: writer {sm['writer_retries']}, SQL {sm['sql_retries']} "
        f"({sm['questions_with_retries']} question(s) needed any)"
    )
    lines.append(f"verification failure types seen: {sm['failure_types'] or 'none'}")
    lines.append(
        "calls and cost per type: "
        + ", ".join(f"{t} {sm['calls'][t]} calls ${sm['cost'][t]:.4f}" for t in TYPES)
    )
    if sm["causes"]:
        lines.append(f"miss causes (first guess, see below): {sm['causes']}")
    for s in scores:
        if s.correct:
            continue
        q = questions[s.qid]
        lines += ["", f"MISS {s.qid} [{s.type}, {s.author}] status {s.status}, route {s.route}"]
        lines.append(f"  question: {q.question}")
        lines.append(f"  why: {s.reason}")
        lines.append(f"  cause (first guess): {s.cause}")
        lines.append(f"  answer: {' '.join(s.answer.split())[:500] or '(none)'}")
        lines.append(f"  cited sources: {s.cited or 'none'}")
        for src in s.sources.splitlines():
            lines.append(f"    {src[:260]}")
        lines.append("  reference: " + reference_text(q))
        if s.extra.get("halt"):
            lines.append(f"  halt: {s.extra['halt']}")
        if s.extra.get("failures"):
            lines.append(f"  verification failures: {[f['type'] for f in s.extra['failures']]}")
    return "\n".join(lines)


def reference_text(q: GoldenQuestion) -> str:
    if q.type == "refuse" or q.expected_refusal:
        return f"a refusal ({q.expected_refusal})"
    parts = []
    if q.reference_analysis:
        checks = (q.reference_stats or {}).get("checks") or []
        first = [(c["path"], c["value"]) for c in checks[:4]]
        parts.append(f"analysis {q.reference_analysis} checks {first}")
    if q.reference_result is not None:
        ref = q.reference_result
        parts.append(f"result {ref.columns} {ref.rows[:5]}")
    if q.type in ("doc", "mixed"):
        parts.append(
            f"target {q.spec} clauses {q.expected_clauses} tables {q.expected_tables} "
            f"chunks {q.confirmed_chunks}"
        )
    if q.must_mention:
        parts.append(f"must mention {q.must_mention}")
    return "; ".join(parts)


def result_cells(scores: list[Score]) -> dict[str, str]:
    sm = summarize(scores)
    return {
        "sql": sm["type"]["sql"],
        "doc": f"{sm['doc_answer']} / {sm['doc_recall_search']}",
        "mixed": sm["type"]["mixed"],
        "stats": sm["type"]["stats"],
        "refuse": sm["type"]["refuse"],
        "human": sm["author"]["human"],
        "assistant": sm["author"]["assistant"],
    }


def save(run_id: str, scores: list[Score]) -> Path:
    path = Path(__file__).resolve().parents[3] / "data" / "eval" / f"eval_e2e_{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(s) for s in scores], indent=2, default=str), encoding="utf-8"
    )
    return path


# --- comparison with an earlier run -----------------------------------------------------------


def compare_with_baseline(base: list[dict], new: list[Score]) -> str:
    """Per question: fixed (wrong -> correct), broke (correct -> wrong) or unchanged, and how each
    author group moved. `base` are the saved rows of the earlier run (dicts with qid, correct,
    author). Only questions present in both runs count."""
    before = {r["qid"]: r for r in base}
    fixed, broke, same_ok, same_wrong = [], [], [], []
    for s in new:
        if s.qid not in before:
            continue
        was = bool(before[s.qid]["correct"])
        if s.correct and not was:
            fixed.append(s.qid)
        elif was and not s.correct:
            broke.append(s.qid)
        else:
            (same_ok if s.correct else same_wrong).append(s.qid)
    lines = [
        f"fixed ({len(fixed)}): {', '.join(fixed) or '-'}",
        f"broke ({len(broke)}): {', '.join(broke) or '-'}",
        f"unchanged: {len(same_ok)} correct in both, {len(same_wrong)} wrong in both "
        f"({', '.join(same_wrong) or '-'})",
    ]
    for author in AUTHORS:
        group = [s for s in new if s.author == author and s.qid in before]
        was = sum(bool(before[s.qid]["correct"]) for s in group)
        now = sum(s.correct for s in group)
        lines.append(f"{author}: {was}/{len(group)} -> {now}/{len(group)} ({now - was:+d})")
    total_was = sum(bool(before[s.qid]["correct"]) for s in new if s.qid in before)
    total_now = sum(s.correct for s in new if s.qid in before)
    lines.append(f"total: {total_was} -> {total_now} ({total_now - total_was:+d})")
    return "\n".join(lines)


# --- rescoring --------------------------------------------------------------------------------


def thread_state(saver, thread: str) -> dict | None:
    """The final state of a saved thread (from its latest checkpoint), or None."""
    found = saver.get_tuple({"configurable": {"thread_id": thread}})
    return dict(found.checkpoint["channel_values"]) if found else None


def rescore_run(run_id: str, questions: list[GoldenQuestion], saver) -> list[Score]:
    """Score the saved threads of an earlier run again with the current scorer: no LLM call, no
    server. Questions the run never reached have no thread and are left out."""
    scores = []
    for q in questions:
        state = thread_state(saver, f"{run_id}-{q.id}")
        if state is not None and state.get("status"):
            scores.append(score_question(q, state))
    return scores


def render_rescore(run_id: str, old: dict[str, dict], new: list[Score]) -> str:
    lines = [f"rescore of {run_id} with the current scorer (no LLM call)", ""]
    lines.append(f"{'id':10} {'old':5} {'new':5}  reason (new)")
    changed = 0
    for s in new:
        was = old.get(s.qid, {}).get("correct")
        changed += was != s.correct
        mark = "  <- changed" if was != s.correct else ""
        lines.append(
            f"{s.qid:10} {'yes' if was else 'NO':5} {'yes' if s.correct else 'NO':5}  "
            f"{s.reason[:90]}{mark}"
        )
    was_ok = sum(bool(old.get(s.qid, {}).get("correct")) for s in new)
    lines.append("")
    lines.append(
        f"old {was_ok}/{len(new)} correct, new {sum(s.correct for s in new)}/{len(new)}, "
        f"{changed} changed"
    )
    return "\n".join(lines)


# --- command -----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="End-to-end eval of the graph on the golden set")
    parser.add_argument("--dry-run", action="store_true", help="print the plan, make no LLM call")
    parser.add_argument("--stop-usd", type=float, default=DEFAULT_STOP_USD)
    parser.add_argument("--note", default="", help="one line for the last column of RESULTS.md")
    parser.add_argument("--rescore", metavar="RUN_ID", help="score a saved run's threads again")
    parser.add_argument(
        "--only",
        metavar="IDS",
        help="comma-separated question ids to run (a smoke run); all questions otherwise",
    )
    parser.add_argument("--baseline", metavar="RUN_ID", help="compare with this earlier run")
    args = parser.parse_args(argv)

    if args.rescore:
        saved = (
            Path(__file__).resolve().parents[3] / "data" / "eval" / f"eval_e2e_{args.rescore}.json"
        )
        old = {r["qid"]: r for r in json.loads(saved.read_text(encoding="utf-8"))}
        with open_checkpointer() as saver:
            new = rescore_run(args.rescore, load_golden(), saver)
        print(render_rescore(args.rescore, old, new))
        return 0

    settings = get_llm_settings()
    agent = get_agent_settings()
    graph_settings = get_graph_settings()
    questions = load_golden()
    bad = problems(questions)
    if bad:
        print("golden set problems:\n  " + "\n  ".join(bad), file=sys.stderr)
        return 2
    if args.only:
        wanted = [i.strip() for i in args.only.split(",") if i.strip()]
        unknown = [i for i in wanted if i not in {q.id for q in questions}]
        if unknown:
            print(f"ERROR: unknown question id(s): {unknown}", file=sys.stderr)
            return 2
        questions = [q for q in questions if q.id in wanted]

    sql_tools, docs_tools = SqlMcpTools(agent.sql_mcp_url), DocsMcpTools(agent.docs_mcp_url)
    analytics_tools = AnalyticsMcpTools(agent.analytics_mcp_url)
    try:
        system_prompt = build_system_prompt(sql_tools.schema_context())
        if any(q.type in ("doc", "mixed") for q in questions):
            docs_tools.search("operating band", None, "vector", 1)  # also needs Ollama
        if any(q.type == "stats" for q in questions):
            analytics_tools.menu()  # the analyses themselves also need Docker
    except (SqlMcpUnavailable, DocsMcpUnavailable, DocsMcpError, AnalyticsMcpUnavailable) as e:
        print(f"ERROR: {e}\nstart the servers with: make servers-up", file=sys.stderr)
        return 2

    pricing = Pricing.load(_repo_path(settings.llm_prices_file), settings.llm_usd_to_inr)
    per_q, total = worst_case_usd(
        questions,
        system_prompt,
        settings,
        agent.agent_max_output_tokens,
        graph_settings.orch_writer_max_tokens,
        graph_settings.orch_evidence_chars,
        pricing,
    )
    n = len(questions)
    print(
        f"eval-e2e plan: {n} questions, at most {n * CALLS_PER_QUESTION} LLM calls "
        f"({n} x {CALLS_PER_QUESTION}), model {settings.llm_model} ({settings.llm_service_tier}), "
        f"approval automatic, one thread per question"
    )
    print(
        f"worst-case cost ${total:.4f} (INR {pricing.to_inr(total):.2f}): every call the dearest "
        f"kind with its full output allowance. The run stops at ${args.stop_usd:.2f} (the current "
        f"question is finished first). Budget ${settings.llm_budget_usd:.2f}."
    )
    if args.dry_run:
        print("dry run: no LLM call made")
        return 0
    if not settings.llm_allow_paid:
        print("ERROR: eval-e2e makes paid calls: set LLM_ALLOW_PAID=true", file=sys.stderr)
        return 2

    gateway = build_gateway(eval_settings(settings, n))
    ledger = gateway.ledger
    if ledger.total_usd() + args.stop_usd > settings.llm_budget_usd:
        print(
            f"ERROR: spent ${ledger.total_usd():.4f} plus the stop ${args.stop_usd:.2f} exceeds "
            f"LLM_BUDGET_USD ${settings.llm_budget_usd:.2f}",
            file=sys.stderr,
        )
        return 2

    commit = git_commit()
    by_id = {q.id: q for q in questions}
    print("\n" + HEADER)
    with gateway.run("eval-e2e") as run, open_checkpointer() as saver:
        deps = GraphDeps(
            run,
            sql_tools,
            docs_tools,
            graph_settings,
            approval_mode="auto",
            analytics_tools=analytics_tools,
        )
        graph = build_graph(deps, saver)
        scores, stopped = evaluate(
            questions, run=run, graph=graph, run_id=run.run_id, stop_usd=args.stop_usd
        )
    print("\n" + render_report(scores, by_id))
    if args.baseline:
        base_file = (
            Path(__file__).resolve().parents[3] / "data" / "eval" / f"eval_e2e_{args.baseline}.json"
        )
        base_rows = json.loads(base_file.read_text(encoding="utf-8"))
        print(f"\n--- against the baseline run {args.baseline} ---")
        print(compare_with_baseline(base_rows, scores))
    saved = save(run.run_id, scores)
    print(f"\nfull results saved to {saved}")
    if stopped:
        print(f"STOPPED EARLY: {stopped} ({len(scores)} of {n} questions scored)")
    calls = sum(s.calls for s in scores)
    cost = sum(s.cost_usd for s in scores)
    total_spent = ledger.total_usd()
    print(
        f"Spend: {calls} paid calls this run, cost ${run.cost_usd:.6f} "
        f"(INR {pricing.to_inr(run.cost_usd):.4f}), total spent ${total_spent:.6f}, "
        f"remaining ${max(settings.llm_budget_usd - total_spent, 0):.6f}"
    )
    note = " ".join(p for p in (args.note, f"STOPPED EARLY: {stopped}" if stopped else "") if p)
    append_e2e_row(
        run_id=run.run_id,
        commit=commit,
        model=settings.llm_model,
        tier=settings.llm_service_tier,
        questions_run=len(scores),
        questions_total=n,
        cells=result_cells(scores),
        calls=calls,
        cost_usd=cost,
        note=note,
    )
    print(f"row appended to {RESULTS_PATH}")
    return 1 if stopped else 0


if __name__ == "__main__":
    raise SystemExit(main())
