"""A JSON view of a thread for the API, the UI and `rflens ask --json`.

Pure: it takes the thread's checkpointed state (and the pending approval payload, if any) and
returns plain dicts. Nothing here talks to a database or a model.
"""

from typing import Any

BADGES = {
    "answered": "verified",
    "partial": "partial",
    "refused": "refused",
    "failed": "failed",
    "cancelled": "cancelled",
}
MAX_VIEW_ROWS = 50


def thread_status(state: dict[str, Any], pending: dict[str, Any] | None, next_nodes: Any) -> str:
    if pending is not None:
        return "pending_approval"
    if state.get("status"):
        return state["status"]
    return "running" if next_nodes else "unknown"


def badge(status: str, state: dict[str, Any]) -> str:
    if status == "pending_approval":
        return "pending approval"
    if status == "answered" and not (state.get("verification") or {}).get("ok"):
        return "unverified"  # cannot happen through finalize; kept so the badge never overclaims
    return BADGES.get(status, status)


def source_items(state: dict[str, Any]) -> list[dict[str, Any]]:
    """The sources the answer can cite: S1 (the SQL result) and D1.. (spec chunks)."""
    answer = state.get("answer") or ""
    items: list[dict[str, Any]] = []
    sql = state.get("sql")
    if sql and sql.get("status") == "answered":
        items.append(
            {
                "tag": "S1",
                "kind": "sql",
                "cited": "[S1]" in answer,
                "query_id": sql.get("query_id"),
                "sql": sql.get("executed_sql") or sql.get("sql"),
                "columns": sql.get("columns") or [],
                "rows": (sql.get("rows") or [])[:MAX_VIEW_ROWS],
                "row_count": sql.get("row_count"),
                "truncated": bool(sql.get("truncated")),
                "notes": sql.get("notes") or [],
            }
        )
    for chunk in state.get("chunks") or []:
        items.append(
            {
                "tag": chunk["ref"],
                "kind": "doc",
                "cited": f"[{chunk['ref']}]" in answer,
                "chunk_id": chunk["chunk_id"],
                "spec": chunk["spec"],
                "version": chunk["version"],
                "clause": chunk["clause"],
                "clause_title": chunk.get("clause_title"),
                "table_number": chunk.get("table_number"),
                "band_lines": chunk.get("band_lines") or [],
            }
        )
    return items


def cost_summary(state: dict[str, Any]) -> dict[str, Any]:
    nodes = state.get("node_stats") or {}
    return {
        "nodes": nodes,
        "calls": sum(n.get("calls", 0) for n in nodes.values()),
        "usd": round(sum(n.get("cost_usd", 0.0) for n in nodes.values()), 8),
    }


def thread_view(
    thread_id: str,
    state: dict[str, Any],
    pending: dict[str, Any] | None = None,
    next_nodes: Any = (),
    drafts: bool = True,
) -> dict[str, Any]:
    status = thread_status(state, pending, next_nodes)
    verification = state.get("verification") or {}
    view = {
        "thread_id": thread_id,
        "question": state.get("question"),
        "status": status,
        "badge": badge(status, state),
        "pending": pending,
        "next": list(next_nodes or []),
        "route": state.get("route"),
        "spec": state.get("spec"),
        "spec_source": state.get("spec_source"),
        "plan": state.get("plan"),
        "approval": state.get("approval"),
        "answer": state.get("answer"),
        "sources_text": state.get("sources") or "",
        "source_items": source_items(state),
        "verification": {
            "ok": verification.get("ok"),
            "failures": verification.get("failures") or [],
            "evidence_map": verification.get("evidence_map") or [],
            "band_edges": verification.get("band_edges") or [],
            "band_joins": verification.get("band_joins") or [],
            "redacted": verification.get("redacted") or [],
        },
        "band_lookup": state.get("band_lookup"),
        "halt": state.get("halt"),
        "cost": cost_summary(state),
    }
    if drafts:
        view["draft_history"] = state.get("draft_history") or []
    return view
