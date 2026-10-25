"""The orchestrator from a terminal: run a question, resume it in a new process, inspect a thread.

    make orchestrate Q="<question>" THREAD=demo-1 ASK=1     # plan, checkpoint, wait at the prompt
    make orchestrate-status THREAD=demo-1                   # checkpoint and LLM calls of the thread
    make orchestrate-resume THREAD=demo-1 DECISION=approve  # a brand-new process finishes the run

`run` prints the plan and waits for approve / edit / cancel with the checkpoint already saved, so
the process can be killed at that prompt and the thread resumed later (`resume`). The thread id is
the question id: the ledger (`llm_calls`) and the checkpoints are looked up by it.

One-shot: `rflens ask "<question>" [--thread ID] [--ask] [--show-drafts] [--json]` runs a question
to the end in this process (asking at the prompt when approval is needed and a terminal exists).

Offline: with LLM_PROVIDER=fake and RFLENS_FAKE_SCRIPT=<json file with a list of answers> the
model is scripted (tests and demos without a key); the MCP servers are still spoken to over HTTP.
"""

import argparse
import json
import os
import sys
import uuid
from typing import Any

from sqlalchemy import text

from rflens.agents.sql_tools import SqlMcpTools
from rflens.config import get_agent_settings, get_graph_settings
from rflens.db import get_ledger_engine
from rflens.eval.golden import load_golden
from rflens.eval.scorer import ResultSet, compare
from rflens.graph.app import (
    Outcome,
    build_graph,
    continue_thread,
    final_text,
    open_checkpointer,
    start_thread,
    thread_config,
)
from rflens.graph.docs_tools import DocsMcpTools
from rflens.graph.nodes import GraphDeps
from rflens.graph.spec_rules import SPECS
from rflens.graph.view import thread_view
from rflens.llm.base import LLMError
from rflens.llm.fake import FakeProvider
from rflens.llm.gateway import Gateway, build_gateway

PROMPT = "approve / edit / cancel [a/e/c]: "
NODE_OF_PURPOSE = {"planner": "planner", "sql-agent": "sql_agent", "writer": "writer"}


def make_gateway(*, quiet: bool = False) -> Gateway:
    """`quiet` sends the gateway's run summary to stderr (so --json output stays parseable)."""
    gateway = build_gateway()
    if quiet:
        gateway.emit = lambda line: print(line, file=sys.stderr)
    script = os.environ.get("RFLENS_FAKE_SCRIPT")
    if script and gateway.settings.llm_provider == "fake":
        with open(script, encoding="utf-8") as f:
            gateway.providers["fake"] = FakeProvider(script=json.load(f))
    return gateway


def make_deps(run, *, ask: bool) -> GraphDeps:
    agent = get_agent_settings()
    settings = get_graph_settings()
    if ask:
        settings = settings.model_copy(update={"ask": True})
    return GraphDeps(
        run,
        SqlMcpTools(agent.sql_mcp_url),
        DocsMcpTools(agent.docs_mcp_url),
        settings,
        approval_mode="interrupt",
    )


# --- output ---------------------------------------------------------------------------------------


def remaining_budget(gateway: Gateway) -> tuple[float, float]:
    spent = gateway.ledger.total_usd()
    return spent, max(gateway.settings.llm_budget_usd - spent, 0.0)


def print_plan(state: dict[str, Any], gateway: Gateway) -> None:
    plan = state.get("plan") or {}
    spent, remaining = remaining_budget(gateway)
    print("\n--- plan ---")
    print(f"route:         {state.get('route')}")
    print(f"spec:          {state.get('spec')} ({state.get('spec_source')})")
    print(f"bands:         {plan.get('bands')}")
    print(f"doc_query:     {plan.get('doc_query')!r}")
    print(f"sql_question:  {plan.get('sql_question')!r}")
    print(
        f"estimated worst-case cost ${plan.get('planned_worst_case_usd', 0):.4f} "
        f"vs remaining budget ${remaining:.4f} (spent ${spent:.4f})"
    )
    approval = state.get("approval") or {}
    for reason in approval.get("reasons", []):
        print(f"approval needed: {reason}")


def print_nodes(stats: dict[str, dict]) -> None:
    print("\n--- nodes ---")
    for node, stat in stats.items():
        print(
            f"  {node:10} runs={stat.get('runs', 0)} calls={stat.get('calls', 0)} "
            f"cost=${stat.get('cost_usd', 0.0):.6f}"
        )


def print_reference(state: dict[str, Any]) -> None:
    """When the question is a golden question, compare its SQL result with the stored reference."""
    golden = next((q for q in load_golden() if q.question == state.get("question")), None)
    sql = state.get("sql")
    if golden is None or golden.reference_result is None:
        return
    if sql and sql.get("status") == "answered":
        verdict = compare(
            ResultSet(sql["columns"], sql["rows"]), golden.reference_result, golden.ordered
        )
        print(f"\n--- vs golden {golden.id}: match={verdict.match} {verdict.reason} ---")
        print(f"  agent:     {sql['columns']} {sql['rows'][:6]}")
    else:
        print(f"\n--- vs golden {golden.id}: no SQL result to compare ---")
    print(f"  reference: {golden.reference_result.columns} {golden.reference_result.rows[:6]}")


def print_drafts(state: dict[str, Any]) -> None:
    for entry in state.get("draft_history") or []:
        print(f"\n--- writer draft {entry['attempt']} (verified ok: {entry['ok']}) ---")
        print(entry["draft"])
        for failure in entry["failures"]:
            print(
                f"  failure: {failure['type']} (sentence {failure['sentence']}): "
                f"{failure['detail']}"
            )


def print_outcome(outcome: Outcome, *, show_drafts: bool = False) -> None:
    state = outcome.state
    print(f"\nstatus: {outcome.status}   thread: {outcome.question_id}")
    sql = state.get("sql")
    if sql:
        print(f"SQL (query #{sql.get('query_id')}): {sql.get('executed_sql') or sql.get('sql')}")
    print("\n--- final answer ---")
    print(final_text(outcome))
    verification = state.get("verification") or {}
    print(f"\nverification: ok={verification.get('ok')} failures={verification.get('failures')}")
    print_reference(state)
    # the drafts are shown on request, and always when the writer needed more than one
    if show_drafts or len(state.get("draft_history") or []) > 1:
        print_drafts(state)
    print_nodes(outcome.stats)


# --- the commands ---------------------------------------------------------------------------


def prompt_decision(proposed_spec: str | None) -> dict[str, Any] | None:
    """The human's answer at the approval prompt; None when there is no terminal to ask."""
    while True:
        try:
            answer = input(PROMPT).strip().lower()
        except EOFError:
            return None
        if answer in ("a", "approve", ""):
            return {"approve": True}
        if answer in ("c", "cancel"):
            return {"approve": False}
        if answer in ("e", "edit"):
            try:
                spec = input(f"spec filter (36.101 / 38.101-1 / none) [{proposed_spec}]: ").strip()
            except EOFError:
                return None
            if spec in SPECS:
                return {"approve": True, "spec": spec}
            if spec.lower() == "none":
                return {"approve": True, "spec": None}
            return {"approve": True, "spec": proposed_spec}
        print("answer a, e or c")


def cmd_run(args: argparse.Namespace) -> int:
    gateway = make_gateway()
    with gateway.run("orchestrate") as run, open_checkpointer() as saver:
        graph = build_graph(make_deps(run, ask=args.ask), saver)
        if graph.get_state(thread_config(args.thread)).values:
            print(
                f"thread {args.thread!r} already exists: pick another THREAD, or continue it "
                f"with make orchestrate-resume THREAD={args.thread}",
                file=sys.stderr,
            )
            return 2
        outcome = start_thread(graph, args.question, args.thread)
        print(f"thread {args.thread}   pid {os.getpid()}", flush=True)
        print_plan(outcome.state, gateway)
        if outcome.interrupt is None:
            print_outcome(outcome)
            return 0
        print(
            f"\nCheckpoint saved; the thread is waiting at the approval step (pid {os.getpid()})."
        )
        print(
            f"Resume it from any process: make orchestrate-resume THREAD={args.thread} "
            "DECISION=approve",
            flush=True,
        )
        decision = prompt_decision(outcome.interrupt.get("proposed_spec"))
        if decision is None:
            print("\nno input: the thread stays saved at the approval step.", file=sys.stderr)
            return 3
        outcome = continue_thread(graph, args.thread, decision)
        print_outcome(outcome)
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    """One-shot: plan, (approve), run, print. With --json the only stdout is one JSON document."""
    thread = args.thread or f"ask-{uuid.uuid4().hex[:8]}"
    gateway = make_gateway(quiet=args.json)
    with gateway.run("ask") as run, open_checkpointer() as saver:
        graph = build_graph(make_deps(run, ask=args.ask), saver)
        if graph.get_state(thread_config(thread)).values:
            print(f"thread {thread!r} already exists: pick another --thread", file=sys.stderr)
            return 2
        outcome = start_thread(graph, args.question, thread)
        if outcome.interrupt is not None:
            if args.json or not sys.stdin.isatty():
                pending = thread_view(thread, outcome.state, outcome.interrupt, ("approval",))
                if args.json:
                    print(json.dumps(pending, indent=2, default=str))
                else:
                    print_plan(outcome.state, gateway)
                print(
                    f"waiting for approval; resume with: make orchestrate-resume THREAD={thread}",
                    file=sys.stderr,
                )
                return 3
            print_plan(outcome.state, gateway)
            decision = prompt_decision(outcome.interrupt.get("proposed_spec"))
            if decision is None:
                print(
                    f"waiting for approval; resume with: make orchestrate-resume THREAD={thread}",
                    file=sys.stderr,
                )
                return 3
            outcome = continue_thread(graph, thread, decision)
        if args.json:
            view = thread_view(thread, outcome.state, None, (), drafts=args.show_drafts)
            print(json.dumps(view, indent=2, default=str))
        else:
            print(f"thread {thread}")
            print_outcome(outcome, show_drafts=args.show_drafts)
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    decision: dict[str, Any] = {"approve": args.decision == "approve"}
    if args.spec:
        decision["spec"] = None if args.spec.lower() == "none" else args.spec
    gateway = make_gateway()
    with gateway.run("orchestrate-resume") as run, open_checkpointer() as saver:
        graph = build_graph(make_deps(run, ask=False), saver)
        snapshot = graph.get_state(thread_config(args.thread))
        if not snapshot.values:
            print(f"no checkpoint for thread {args.thread!r}", file=sys.stderr)
            return 2
        print(f"thread {args.thread}   pid {os.getpid()}   next: {snapshot.next or 'finished'}")
        outcome = continue_thread(graph, args.thread, decision)
        print_outcome(outcome, show_drafts=args.show_drafts)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    gateway = make_gateway()
    with gateway.run("orchestrate-status") as run, open_checkpointer() as saver:
        graph = build_graph(make_deps(run, ask=False), saver)
        config = thread_config(args.thread)
        snapshot = graph.get_state(config)
        checkpoints = list(saver.list(config))
        print(f"thread {args.thread}")
        if not snapshot.values:
            print("  no checkpoint")
        else:
            state = snapshot.values
            latest = checkpoints[0] if checkpoints else None
            print(f"  checkpoints in checkpoints.checkpoints: {len(checkpoints)}")
            if latest:
                print(f"  latest checkpoint: {latest.config['configurable']['checkpoint_id']}")
                step, source = latest.metadata.get("step"), latest.metadata.get("source")
                print(f"  written by step:   {step} ({source})")
            print(f"  next node(s):      {list(snapshot.next) or 'none (finished)'}")
            waiting = [i.value for t in snapshot.tasks for i in t.interrupts]
            if waiting:
                print(f"  waiting for approval: {waiting[0].get('reasons')}")
            plan = state.get("plan") or {}
            print(f"  route: {state.get('route')}   spec: {state.get('spec')}")
            print(f"  status: {state.get('status')}")
            print(f"  plan: bands={plan.get('bands')} doc_query={plan.get('doc_query')!r}")
            print(f"        sql_question={plan.get('sql_question')!r}")
        print_llm_calls(args.thread)
    return 0


def print_llm_calls(thread: str) -> None:
    """From the ledger: the run ids that made calls for this question, and calls per node."""
    with get_ledger_engine().connect() as c:
        rows = c.execute(
            text(
                "SELECT run_id, purpose, count(*) AS calls, coalesce(sum(cost_usd), 0) AS cost, "
                "min(created_at) AS first FROM llm_calls WHERE question_id = :q "
                "AND status IN ('pending', 'ok', 'error') GROUP BY run_id, purpose "
                "ORDER BY min(created_at), purpose"
            ),
            {"q": thread},
        ).all()
    runs = list(dict.fromkeys(r.run_id for r in rows))
    print(f"  llm_calls for question_id {thread!r}: {len(runs)} run id(s)")
    for run_id in runs:
        print(f"    run {run_id}")
        for r in (r for r in rows if r.run_id == run_id):
            node = NODE_OF_PURPOSE.get(r.purpose, r.purpose)
            print(f"      {node:10} calls={r.calls} cost=${float(r.cost):.6f}")
    if not rows:
        print("    none")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="rflens.graph.cli", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="plan a question, checkpoint, wait at the approval prompt")
    run.add_argument("--question", required=True)
    run.add_argument("--thread", required=True)
    run.add_argument("--ask", action="store_true", help="always ask for approval (ASK=1)")
    run.set_defaults(func=cmd_run)
    ask = sub.add_parser("ask", help="run one question to the end in this process")
    ask.add_argument("question")
    ask.add_argument("--thread", help="thread id (default: a new ask-<hex> id)")
    ask.add_argument("--ask", action="store_true", help="ask for approval first (ASK=1)")
    ask.add_argument("--show-drafts", action="store_true", help="print every writer draft")
    ask.add_argument("--json", action="store_true", help="print one JSON document")
    ask.set_defaults(func=cmd_ask)
    resume = sub.add_parser("resume", help="continue a saved thread in this (new) process")
    resume.add_argument("--thread", required=True)
    resume.add_argument("--decision", choices=["approve", "cancel"], default="approve")
    resume.add_argument("--show-drafts", action="store_true", help="print every writer draft")
    resume.add_argument("--spec", help="36.101, 38.101-1 or none (overrides the proposed spec)")
    resume.set_defaults(func=cmd_resume)
    status = sub.add_parser("status", help="checkpoint and ledger calls of a thread")
    status.add_argument("--thread", required=True)
    status.set_defaults(func=cmd_status)
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except LLMError as e:
        print(
            f"\nstopped: {type(e).__name__}: {e}\nthe thread keeps its last checkpoint.",
            file=sys.stderr,
        )
        return 4


if __name__ == "__main__":
    sys.exit(main())
