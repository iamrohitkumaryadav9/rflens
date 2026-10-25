"""Live smoke of the orchestrator graph on golden questions (paid calls: run it on purpose).

    make sql-mcp         # another terminal
    make docs-mcp        # another terminal (needs Ollama)
    LLM_ALLOW_PAID=true LLM_BUDGET_USD=<spent + cap> \\
        uv run python scripts/graph_smoke.py sql-01 doc-02 mixed-01

Approval is automatic. The gateway's LLM_BUDGET_USD is the hard stop. Prints, per question, the SQL,
the final answer with its Sources section, the result compared with the golden reference, every
writer draft with its failures, the verification result with the evidence map and band edges, and
the calls and cost of every node. The checkpoints of the smoke threads are deleted afterwards.
"""

import sys
import uuid

from rflens.agents.sql_tools import SqlMcpTools
from rflens.config import get_agent_settings, get_graph_settings
from rflens.eval.golden import load_golden
from rflens.eval.scorer import ResultSet, compare
from rflens.graph.app import ask, build_graph, final_text, open_checkpointer
from rflens.graph.docs_tools import DocsMcpTools
from rflens.graph.nodes import GraphDeps
from rflens.llm.gateway import build_gateway


def show_sql_and_reference(state: dict, golden) -> None:
    sql = state.get("sql")
    if sql:
        print(f"\nSQL (query #{sql.get('query_id')}): {sql.get('executed_sql') or sql.get('sql')}")
    else:
        print("\nSQL: none (withheld or not run)")
    if golden is None or golden.reference_result is None:
        return
    reference = golden.reference_result
    if sql and sql.get("status") == "answered":
        verdict = compare(ResultSet(sql["columns"], sql["rows"]), reference, golden.ordered)
        print(f"\n--- vs golden {golden.id}: match={verdict.match} {verdict.reason} ---")
        print(f"  agent:     {sql['columns']} {sql['rows'][:6]}")
    else:
        print(f"\n--- vs golden {golden.id}: no SQL result to compare ---")
    print(f"  reference: {reference.columns} {reference.rows[:6]}")


def show(outcome, golden=None) -> None:
    state = outcome.state
    print(f"\nstatus: {outcome.status}   route: {state.get('route')}   spec: {state.get('spec')}")
    print(f"plan: {state.get('plan')}")
    print(f"approval: {state.get('approval')}")
    show_sql_and_reference(state, golden)
    print("\n--- final answer ---")
    print(final_text(outcome))
    for entry in state.get("draft_history") or []:
        print(f"\n--- writer draft {entry['attempt']} (verified ok: {entry['ok']}) ---")
        print(entry["draft"])
        for failure in entry["failures"]:
            print(
                f"  failure: {failure['type']} (sentence {failure['sentence']}): "
                f"{failure['detail']}"
            )
    v = state.get("verification") or {}
    print(f"\n--- verification: ok={v.get('ok')} failures={v.get('failures')} ---")
    for entry in v.get("evidence_map", []):
        numbers = ", ".join(
            f"{n['number']}<-{n['source']}" + (f"({n['match']})" if n["match"] else "")
            for n in entry["numbers"]
        )
        print(f"  s{entry['sentence']} cites={entry['cites']} numbers=[{numbers}]")
    if state.get("band_lookup"):
        print(f"band lookup: {state['band_lookup']}")
    for chunk in state.get("chunks") or []:
        print(f"  {chunk['ref']} = {chunk['chunk_id']} {chunk.get('band_lines') or ''}")
    print(f"  band joins: {v.get('band_joins')}")
    for edge in v.get("band_edges", []):
        print(
            f"  band edge {edge['column']} = {edge['value']:g} <- {edge['cited_chunk']} "
            f"({edge['chunk_id']})"
        )
    print("\n--- nodes ---")
    total_calls, total_cost = 0, 0.0
    for node, stat in outcome.stats.items():
        calls, cost = stat.get("calls", 0), stat.get("cost_usd", 0.0)
        total_calls, total_cost = total_calls + calls, total_cost + cost
        print(f"  {node:10} runs={stat.get('runs', 0)} calls={calls} cost=${cost:.6f}")
    print(f"  {'total':10} calls={total_calls} cost=${total_cost:.6f}")


def main(argv: list[str]) -> int:
    golden = {q.id: q for q in load_golden()}
    wanted = argv or ["sql-01", "doc-02", "mixed-01"]
    agent = get_agent_settings()
    gateway = build_gateway()
    with gateway.run("graph-smoke") as run, open_checkpointer() as saver:
        deps = GraphDeps(
            run,
            SqlMcpTools(agent.sql_mcp_url),
            DocsMcpTools(agent.docs_mcp_url),
            get_graph_settings(),
            approval_mode="auto",
        )
        graph = build_graph(deps, saver)
        threads = []
        try:
            for gid in wanted:
                thread = f"graph-smoke-{gid}-{uuid.uuid4().hex[:6]}"
                threads.append(thread)
                print(f"\n{'=' * 78}\n{gid}: {golden[gid].question}")
                show(ask(graph, golden[gid].question, thread), golden[gid])
        finally:
            for thread in threads:
                saver.delete_thread(thread)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
