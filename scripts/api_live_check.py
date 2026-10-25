"""One golden question through the running API, with approval (paid calls: run it on purpose).

    make sql-mcp ; make docs-mcp                        # other terminals
    LLM_ALLOW_PAID=true LLM_BUDGET_USD=<spent + cap> make api   # the API process needs these two
    uv run python scripts/api_live_check.py mixed-02

POST /ask with ask=true (the thread waits for approval), then POST /threads/{id}/resume approve,
then GET /threads/{id}; prints each step and compares the SQL result with the golden reference.
"""

import json
import sys

from rflens.eval.golden import load_golden
from rflens.eval.scorer import ResultSet, compare
from rflens.ui.client import ApiClient


def main(argv: list[str]) -> int:
    golden = {q.id: q for q in load_golden()}[argv[0] if argv else "mixed-02"]
    client = ApiClient.from_env()
    print(f"{golden.id}: {golden.question}\n")
    pending = client.ask(golden.question, ask=True)
    thread = pending["thread_id"]
    print(f"POST /ask -> thread {thread}, status {pending['status']} ({pending['badge']})")
    print(f"  route={pending['route']} spec={pending['spec']}")
    print(f"  plan: {json.dumps(pending['plan'], indent=2)}")
    print(f"  pending reasons: {pending['pending']['reasons']}")
    print(f"  cost so far: {pending['cost']}")
    print("\nPOST /threads/{id}/resume approve ...")
    client.resume(thread, "approve")
    view = client.thread(thread)
    print(f"GET /threads/{thread} -> status {view['status']} badge {view['badge']}")
    print(f"\n--- final answer ---\n{view['answer']}\n\nSources:\n{view['sources_text']}")
    print(f"\nverification: {json.dumps(view['verification'], indent=2)[:1500]}")
    print(f"cost: {json.dumps(view['cost'])}")
    print(f"runs: {json.dumps(view['runs'])}")
    sql = next((s for s in view["source_items"] if s["kind"] == "sql"), None)
    if sql and golden.reference_result is not None:
        verdict = compare(
            ResultSet(sql["columns"], sql["rows"]), golden.reference_result, golden.ordered
        )
        print(f"\n--- vs golden {golden.id}: match={verdict.match} {verdict.reason} ---")
        print(f"  agent:     {sql['columns']} {sql['rows'][:5]}")
        print(f"  reference: {golden.reference_result.columns} {golden.reference_result.rows[:5]}")
    for item in view["source_items"]:
        if item["kind"] == "doc":
            chunk = client.chunk(item["chunk_id"])
            print(
                f"\n[{item['tag']}] chunk read through the API: {chunk['chunk_id']} "
                f"clause {chunk['clause']}, {len(chunk['text'])} chars"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
